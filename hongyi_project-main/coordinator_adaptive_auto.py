import os
import csv
import json
import time
import shutil
import zipfile
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FEATURES = [
    "cpu_util",
    "gpu_util",
    "throughput",
    "memory_headroom",
    "time",
]

DEFAULT_WEIGHTS = {
    "cpu_util": 0.20,
    "gpu_util": 0.20,
    "throughput": 0.20,
    "memory_headroom": 0.20,
    "time": 0.20,
}


def parse_workers(worker_args):
    workers = {}

    for item in worker_args:
        if "=" not in item:
            raise ValueError(f"Bad worker format: {item}. Expected B=http://ip:8000")

        name, url = item.split("=", 1)
        workers[name.strip()] = url.strip().rstrip("/")

    return workers


def parse_ratios(ratio_args, worker_names):
    if ratio_args:
        ratios = {}

        for item in ratio_args:
            if "=" not in item:
                raise ValueError(f"Bad ratio format: {item}. Expected B=33.33")

            name, value = item.split("=", 1)
            ratios[name.strip()] = float(value)

    else:
        equal = 100.0 / len(worker_names)
        ratios = {name: equal for name in worker_names}

    for name in worker_names:
        if name not in ratios:
            raise ValueError(f"Missing ratio for worker {name}")

    total = sum(ratios.values())

    if total <= 0:
        raise ValueError("Ratio sum must be positive.")

    normalized = {
        name: ratios[name] / total for name in worker_names
    }

    return normalized


def check_workers(workers):
    info = {}

    for name, url in workers.items():
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
        info[name] = r.json()

    return info


def scan_dataset(data_root):
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    class_names = sorted([
        p.name for p in data_root.iterdir()
        if p.is_dir()
    ])

    if len(class_names) < 2:
        raise RuntimeError(f"Dataset must contain at least two class folders under {data_root}")

    class_to_idx = {
        name: idx for idx, name in enumerate(class_names)
    }

    rows = []
    sample_id = 0

    for class_name in class_names:
        class_dir = data_root / class_name

        for root, _, files in os.walk(class_dir):
            for filename in files:
                ext = Path(filename).suffix.lower()

                if ext not in IMAGE_EXTENSIONS:
                    continue

                img_path = Path(root) / filename

                rows.append({
                    "id": sample_id,
                    "path": str(img_path),
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                })

                sample_id += 1

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(f"No images found under {data_root}")

    return df, class_to_idx


def split_by_ratios(df, ratios, round_id):
    df = df.sample(frac=1.0, random_state=round_id).reset_index(drop=True)

    names = list(ratios.keys())
    total_n = len(df)

    allocations = {}
    remaining = total_n

    for i, name in enumerate(names):
        if i == len(names) - 1:
            n = remaining
        else:
            n = int(total_n * ratios[name])
            remaining -= n

        allocations[name] = n

    shards = {}
    start = 0

    for name in names:
        n = allocations[name]
        shards[name] = df.iloc[start:start + n].copy()
        start += n

    return shards, allocations


def make_shard_zip(shard_df, out_zip_path):
    out_zip_path = Path(out_zip_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        manifest_rows = []

        for _, row in shard_df.iterrows():
            src_path = Path(row["path"])
            class_name = str(row["class_name"])
            label = int(row["label"])
            sample_id = int(row["id"])

            safe_name = f"{sample_id}_{src_path.name}"

            rel_path = Path("images") / class_name / safe_name
            dst_path = tmpdir / rel_path

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

            manifest_rows.append({
                "rel_path": str(rel_path),
                "label": label,
            })

        manifest_path = tmpdir / "manifest.csv"

        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rel_path", "label"])
            writer.writeheader()
            writer.writerows(manifest_rows)

        with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmpdir):
                for filename in files:
                    full_path = Path(root) / filename
                    arcname = full_path.relative_to(tmpdir)
                    zf.write(full_path, arcname=str(arcname))

    return out_zip_path


def send_stats_job(
    worker_name,
    worker_url,
    job_id,
    round_id,
    shard_zip_path,
    args,
    num_classes,
):
    request_start = time.time()

    with open(shard_zip_path, "rb") as zip_f:
        files = {
            "data_zip": (
                "data.zip",
                zip_f,
                "application/zip",
            ),
        }

        data = {
            "job_id": job_id,
            "round_id": str(round_id),
            "num_classes": str(num_classes),
            "model_name": args.model_name,
            "batch_size": str(args.batch_size),
            "lr": str(args.lr),
            "local_epochs": str(args.local_epochs),
            "num_workers": str(args.num_workers),
            "amp": str(args.amp),
        }

        r = requests.post(
            f"{worker_url}/train_zip_stats",
            data=data,
            files=files,
            timeout=args.worker_timeout,
        )

    r.raise_for_status()

    result = r.json()

    result["worker_name"] = worker_name
    result["worker_url"] = worker_url
    result["request_seconds"] = time.time() - request_start
    result["shard_zip_bytes"] = Path(shard_zip_path).stat().st_size

    return result


def cleanup_worker_job(worker_url, job_id):
    try:
        requests.delete(f"{worker_url}/job/{job_id}", timeout=10)
    except Exception:
        pass


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def extract_feature_row(res):
    """
    Convert raw worker return JSON into algorithm input features.

    Features required by supervisor:
      cpu_util
      gpu_util
      throughput
      memory_headroom
      time
    """

    avg_memory = safe_float(res.get("avg_memory_percent"), 0.0)
    memory_headroom = max(0.0, 100.0 - avg_memory)

    throughput = safe_float(res.get("samples_per_sec_total"), 0.0)

    time_value = safe_float(res.get("total_seconds"), 0.0)

    return {
        "worker_name": res["worker_name"],

        "cpu_util": safe_float(res.get("avg_cpu_percent"), 0.0),
        "gpu_util": safe_float(res.get("avg_gpu_util_percent"), 0.0),
        "throughput": throughput,
        "memory_headroom": memory_headroom,
        "time": time_value,

        "observed_speed": throughput,
    }


def fit_auto_weight_model(history_rows, ridge_alpha=1.0):
    """
    Automatically learn feature weights from all previous profiling results.

    Input features:
      cpu_util, gpu_util, throughput, memory_headroom, time
    """

    if len(history_rows) < 3:
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "coef": {k: 0.0 for k in FEATURES},
            "intercept": 0.0,
            "x_mean": {k: 0.0 for k in FEATURES},
            "x_std": {k: 1.0 for k in FEATURES},
            "y_mean": 0.0,
            "y_std": 1.0,
            "method": "default_equal_weights_not_enough_history",
        }

    df = pd.DataFrame(history_rows)

    X = df[FEATURES].astype(float).values
    y = df["observed_speed"].astype(float).values

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    x_std[x_std < 1e-8] = 1.0

    y_mean = y.mean()
    y_std = y.std()

    if y_std < 1e-8:
        y_std = 1.0

    Xz = (X - x_mean) / x_std
    yz = (y - y_mean) / y_std

    n_features = Xz.shape[1]

    A = Xz.T @ Xz + ridge_alpha * np.eye(n_features)
    b = Xz.T @ yz

    try:
        coef = np.linalg.solve(A, b)
    except Exception:
        coef = np.linalg.pinv(A) @ b

    abs_coef = np.abs(coef)

    if abs_coef.sum() < 1e-8:
        weights = DEFAULT_WEIGHTS.copy()
    else:
        weights = {
            feature: float(abs_coef[i] / abs_coef.sum())
            for i, feature in enumerate(FEATURES)
        }

    return {
        "weights": weights,
        "coef": {
            feature: float(coef[i])
            for i, feature in enumerate(FEATURES)
        },
        "intercept": 0.0,
        "x_mean": {
            feature: float(x_mean[i])
            for i, feature in enumerate(FEATURES)
        },
        "x_std": {
            feature: float(x_std[i])
            for i, feature in enumerate(FEATURES)
        },
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "method": "ridge_regression_auto_weights",
    }


def predict_speed_from_model(model, feature_row):
    """
    Predict worker speed from learned model.
    """

    coef = np.array([model["coef"][f] for f in FEATURES], dtype=float)
    x_mean = np.array([model["x_mean"][f] for f in FEATURES], dtype=float)
    x_std = np.array([model["x_std"][f] for f in FEATURES], dtype=float)

    x = np.array([feature_row[f] for f in FEATURES], dtype=float)
    xz = (x - x_mean) / np.maximum(x_std, 1e-8)

    pred_z = float(xz @ coef)
    pred_speed = model["y_mean"] + pred_z * model["y_std"]

    observed_speed = max(float(feature_row["observed_speed"]), 1e-8)

    if pred_speed <= 0 or np.isnan(pred_speed) or np.isinf(pred_speed):
        pred_speed = observed_speed

    blended = 0.5 * pred_speed + 0.5 * observed_speed

    return max(float(blended), 1e-8)


def compute_next_ratios(
    worker_names,
    current_ratios,
    latest_feature_rows,
    model,
    smoothing=1.0,
    min_ratio=0.05,
):
    """
    Convert learned feature model into data allocation ratios.
    """

    speed_by_worker = {}

    for row in latest_feature_rows:
        name = row["worker_name"]
        speed_by_worker[name] = predict_speed_from_model(model, row)

    total_speed = sum(speed_by_worker.values())

    if total_speed <= 0:
        ideal_ratios = {
            name: 1.0 / len(worker_names)
            for name in worker_names
        }
    else:
        ideal_ratios = {
            name: speed_by_worker[name] / total_speed
            for name in worker_names
        }

    # Smooth the change to avoid unstable jumps.
    new_ratios = {}

    for name in worker_names:
        old_r = current_ratios[name]
        ideal_r = ideal_ratios[name]
        r = (1.0 - smoothing) * old_r + smoothing * ideal_r
        r = max(min_ratio, r)
        new_ratios[name] = r

    # Normalize after min-ratio clipping.
    s = sum(new_ratios.values())

    new_ratios = {
        name: new_ratios[name] / s
        for name in worker_names
    }

    return new_ratios, speed_by_worker, ideal_ratios


def print_weights(weights, coef=None):
    print("Auto parameter weights:")

    for feature in FEATURES:
        if coef is None:
            print(f"  {feature:<16}: weight={weights[feature]:.4f}")
        else:
            sign = "positive" if coef[feature] >= 0 else "negative"
            print(
                f"  {feature:<16}: weight={weights[feature]:.4f}, "
                f"coefficient={coef[feature]:+.4f}, direction={sign}"
            )


def print_ratios(title, ratios):
    print(title)

    for name, ratio in ratios.items():
        print(f"  {name}: {ratio * 100:.2f}%")


def print_round_results(round_id, metric_rows, feature_rows, weights_used):
    print(f"\n========== Round {round_id} Result ==========")

    print("\nWeights used for this round:")
    for feature in FEATURES:
        print(f"  {feature:<16}: {weights_used[feature]:.4f}")

    print("\nWorker profiling results:")
    print(
        "Worker | Samples | Time(s) | CPU Util | GPU Util | Throughput | Memory Headroom"
    )
    print(
        "-------|---------|---------|----------|----------|------------|----------------"
    )

    feature_by_worker = {
        row["worker_name"]: row
        for row in feature_rows
    }

    for row in metric_rows:
        name = row["worker_name"]
        f = feature_by_worker[name]

        print(
            f"{name:>6} | "
            f"{int(row['trained_samples']):>7} | "
            f"{float(row['total_seconds']):>7.3f} | "
            f"{float(f['cpu_util']):>8.2f}% | "
            f"{float(f['gpu_util']):>8.2f}% | "
            f"{float(f['throughput']):>10.3f} | "
            f"{float(f['memory_headroom']):>14.2f}%"
        )

    times = [float(row["total_seconds"]) for row in metric_rows]
    gap = max(times) - min(times)

    print("------------------------------------------")
    print(f"Time gap max-min: {gap:.3f} seconds")

    return gap


def append_csv(path, rows, fieldnames):
    path = Path(path)
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in fieldnames
            })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", nargs="+", required=True)

    parser.add_argument("--initial-ratios", nargs="*", default=None)
    parser.add_argument("--work-dir", default="./adaptive_auto_work")

    parser.add_argument("--model-name", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--target-gap-seconds", type=float, default=3.0)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--smoothing", type=float, default=0.70)
    parser.add_argument("--min-ratio", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)

    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--cleanup-workers", action="store_true")

    args = parser.parse_args()

    workers = parse_workers(args.workers)
    worker_names = list(workers.keys())

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("Checking workers...")
    worker_info = check_workers(workers)

    print("\nWorker status:")
    for name, info in worker_info.items():
        print(
            f"{name}: hostname={info.get('hostname')}, "
            f"device={info.get('device')}, "
            f"cpu_limit={info.get('cpu_limit')}, "
            f"gpu={info.get('gpu_name')}, "
            f"gpu_memory={safe_float(info.get('gpu_total_memory_gb'), 0.0):.2f}GB"
        )

    current_ratios = parse_ratios(args.initial_ratios, worker_names)

    print_ratios("\nInitial data allocation ratios:", current_ratios)

    print("\nScanning dataset on A...")
    df, class_to_idx = scan_dataset(args.data_root)
    num_classes = len(class_to_idx)

    print(f"Total samples: {len(df)}")
    print(f"Classes: {num_classes}")

    with open(work_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    history_feature_rows = []

    current_weights = DEFAULT_WEIGHTS.copy()
    final_gap = None
    converged = False

    metrics_fieldnames = [
        "round",
        "worker_name",
        "hostname",
        "mode",

        "ratio_used",
        "allocated_samples",
        "trained_samples",

        "cpu_util",
        "gpu_util",
        "throughput",
        "memory_headroom",
        "time",

        "train_seconds",
        "total_seconds",
        "request_seconds",

        "samples_per_sec_train",
        "samples_per_sec_total",
        "iteration_time",

        "loss",
        "acc",
        "shard_zip_bytes",

        "avg_cpu_percent",
        "max_cpu_percent",
        "avg_memory_percent",
        "max_memory_percent",
        "avg_gpu_util_percent",
        "max_gpu_util_percent",
        "avg_gpu_memory_used_mb",
        "max_gpu_memory_used_mb",
        "avg_gpu_memory_percent",
        "max_gpu_memory_percent",
    ]

    weight_fieldnames = [
        "round",
        "method",
        "gap_seconds",

        "weight_cpu_util",
        "weight_gpu_util",
        "weight_throughput",
        "weight_memory_headroom",
        "weight_time",

        "coef_cpu_util",
        "coef_gpu_util",
        "coef_throughput",
        "coef_memory_headroom",
        "coef_time",

        "ratio_B",
        "ratio_C",
        "ratio_D",
    ]

    for round_id in range(1, args.max_rounds + 1):
        print("\n\n=================================================")
        print(f"Starting adaptive round {round_id}")
        print("=================================================")

        print_ratios("\nData allocation ratios used in this round:", current_ratios)

        print("\nParameter weights used in this round:")
        for feature in FEATURES:
            print(f"  {feature:<16}: {current_weights[feature]:.4f}")

        round_dir = work_dir / f"round_{round_id}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print("\nSplitting data by current ratios...")
        shards, allocations = split_by_ratios(
            df=df,
            ratios=current_ratios,
            round_id=round_id,
        )

        for name in worker_names:
            print(f"  {name}: {current_ratios[name] * 100:.2f}% -> {allocations[name]} samples")

        print("\nCreating shard zip files on A...")
        shard_zip_paths = {}

        for name, shard_df in shards.items():
            zip_path = round_dir / f"shard_{name}.zip"
            make_shard_zip(shard_df, zip_path)
            shard_zip_paths[name] = zip_path

        print("\nSending data shards only.")
        print("No model weights are exchanged.")

        round_results = []
        futures = []

        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            for name, url in workers.items():
                job_id = f"adaptive_auto_round_{round_id}_{name}"

                fut = executor.submit(
                    send_stats_job,
                    name,
                    url,
                    job_id,
                    round_id,
                    shard_zip_paths[name],
                    args,
                    num_classes,
                )

                futures.append((name, url, job_id, fut))

            for name, url, job_id, fut in futures:
                res = fut.result()
                round_results.append(res)

        feature_rows = []
        metric_rows = []

        for res in round_results:
            name = res["worker_name"]
            f = extract_feature_row(res)
            feature_rows.append(f)

            metric_rows.append({
                "round": round_id,
                "worker_name": name,
                "hostname": res.get("hostname"),
                "mode": res.get("mode"),

                "ratio_used": current_ratios[name],
                "allocated_samples": allocations[name],
                "trained_samples": res.get("samples"),

                "cpu_util": f["cpu_util"],
                "gpu_util": f["gpu_util"],
                "throughput": f["throughput"],
                "memory_headroom": f["memory_headroom"],
                "time": f["time"],

                "train_seconds": res.get("train_seconds"),
                "total_seconds": res.get("total_seconds"),
                "request_seconds": res.get("request_seconds"),

                "samples_per_sec_train": res.get("samples_per_sec_train"),
                "samples_per_sec_total": res.get("samples_per_sec_total"),
                "iteration_time": res.get("iteration_time"),

                "loss": res.get("loss"),
                "acc": res.get("acc"),
                "shard_zip_bytes": res.get("shard_zip_bytes"),

                "avg_cpu_percent": res.get("avg_cpu_percent"),
                "max_cpu_percent": res.get("max_cpu_percent"),
                "avg_memory_percent": res.get("avg_memory_percent"),
                "max_memory_percent": res.get("max_memory_percent"),
                "avg_gpu_util_percent": res.get("avg_gpu_util_percent"),
                "max_gpu_util_percent": res.get("max_gpu_util_percent"),
                "avg_gpu_memory_used_mb": res.get("avg_gpu_memory_used_mb"),
                "max_gpu_memory_used_mb": res.get("max_gpu_memory_used_mb"),
                "avg_gpu_memory_percent": res.get("avg_gpu_memory_percent"),
                "max_gpu_memory_percent": res.get("max_gpu_memory_percent"),
            })

        gap = print_round_results(
            round_id=round_id,
            metric_rows=metric_rows,
            feature_rows=feature_rows,
            weights_used=current_weights,
        )

        final_gap = gap

        append_csv(
            work_dir / "adaptive_auto_metrics.csv",
            metric_rows,
            metrics_fieldnames,
        )

        history_feature_rows.extend(feature_rows)

        learned_model = fit_auto_weight_model(
            history_feature_rows,
            ridge_alpha=args.ridge_alpha,
        )

        learned_weights = learned_model["weights"]
        learned_coef = learned_model["coef"]

        print("\nAuto-learned parameter weights after this round:")
        print_weights(learned_weights, learned_coef)

        weight_row = {
            "round": round_id,
            "method": learned_model["method"],
            "gap_seconds": gap,

            "weight_cpu_util": learned_weights["cpu_util"],
            "weight_gpu_util": learned_weights["gpu_util"],
            "weight_throughput": learned_weights["throughput"],
            "weight_memory_headroom": learned_weights["memory_headroom"],
            "weight_time": learned_weights["time"],

            "coef_cpu_util": learned_coef["cpu_util"],
            "coef_gpu_util": learned_coef["gpu_util"],
            "coef_throughput": learned_coef["throughput"],
            "coef_memory_headroom": learned_coef["memory_headroom"],
            "coef_time": learned_coef["time"],

            "ratio_B": current_ratios.get("B", ""),
            "ratio_C": current_ratios.get("C", ""),
            "ratio_D": current_ratios.get("D", ""),
        }

        append_csv(
            work_dir / "adaptive_auto_weights.csv",
            [weight_row],
            weight_fieldnames,
        )

        if gap <= args.target_gap_seconds:
            converged = True
            print("\nConvergence reached.")
            print(f"B/C/D completion time gap <= {args.target_gap_seconds:.3f} seconds.")
            break

        next_ratios, estimated_speed, ideal_ratios = compute_next_ratios(
            worker_names=worker_names,
            current_ratios=current_ratios,
            latest_feature_rows=feature_rows,
            model=learned_model,
            smoothing=args.smoothing,
            min_ratio=args.min_ratio,
        )

        print("\nEstimated worker speed for next allocation:")
        for name in worker_names:
            print(f"  {name}: {estimated_speed[name]:.3f} samples/sec")

        print_ratios("\nIdeal ratios from learned model:", ideal_ratios)
        print_ratios("\nSmoothed ratios for next round:", next_ratios)

        current_weights = learned_weights
        current_ratios = next_ratios

        if args.cleanup_workers:
            for res in round_results:
                cleanup_worker_job(res["worker_url"], res["job_id"])

    print("\n\n=================================================")
    print("Adaptive auto allocation finished")
    print("=================================================")

    if converged:
        print(f"Final status: converged, final time gap = {final_gap:.3f} seconds")
    else:
        print(f"Final status: max rounds reached, final time gap = {final_gap:.3f} seconds")

    print("\nSaved files:")
    print(f"  {work_dir / 'adaptive_auto_metrics.csv'}")
    print(f"  {work_dir / 'adaptive_auto_weights.csv'}")
    print(f"  {work_dir / 'class_to_idx.json'}")

    print("\nReminder:")
    print("This is stats-only adaptive data allocation.")
    print("No global model weights were sent.")
    print("No local model weights were returned.")
    print("No FedAvg aggregation was performed.")


if __name__ == "__main__":
    main()
