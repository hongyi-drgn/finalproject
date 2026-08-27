import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from coordinator_adaptive_auto import (
    FEATURES,
    parse_workers,
    check_workers,
    scan_dataset,
    split_by_ratios,
    make_shard_zip,
    send_stats_job,
    cleanup_worker_job,
    safe_float,
    append_csv,
    predict_speed_from_model,
    compute_next_ratios,
)


DEFAULT_DIRECTIONS = {
    "cpu_util": -1.0,          # lower CPU pressure is better
    "gpu_util": 1.0,           # higher GPU utilisation is better
    "throughput": 1.0,         # higher throughput is better
    "memory_headroom": 1.0,    # higher memory headroom is better
    "time": -1.0,              # lower completion time is better
}


def parse_weights(weight_args):
    weights = {}

    for item in weight_args:
        if "=" not in item:
            raise ValueError(f"Bad weight format: {item}. Expected throughput=0.5")

        name, value = item.split("=", 1)
        name = name.strip()
        value = float(value)

        if name not in FEATURES:
            raise ValueError(f"Unknown weight name: {name}. Allowed: {', '.join(FEATURES)}")

        weights[name] = value

    for feature in FEATURES:
        if feature not in weights:
            raise ValueError(f"Missing weight for {feature}")

    total = sum(weights.values())

    if total <= 0:
        raise ValueError("Weight sum must be positive.")

    return {
        feature: weights[feature] / total
        for feature in FEATURES
    }


def parse_ratios(ratio_args, worker_names):
    if ratio_args is None:
        return {
            name: 1.0 / len(worker_names)
            for name in worker_names
        }

    ratios = {}

    for item in ratio_args:
        if "=" not in item:
            raise ValueError(f"Bad ratio format: {item}. Expected B=33.33")

        name, value = item.split("=", 1)
        ratios[name.strip()] = float(value)

    for name in worker_names:
        if name not in ratios:
            raise ValueError(f"Missing ratio for worker {name}")

    total = sum(ratios.values())

    if total <= 0:
        raise ValueError("Ratio sum must be positive.")

    return {
        name: ratios[name] / total
        for name in worker_names
    }


def row_to_feature(row):
    avg_memory = safe_float(row.get("avg_memory_percent"), 0.0)

    memory_headroom = safe_float(
        row.get("memory_headroom"),
        max(0.0, 100.0 - avg_memory),
    )

    return {
        "worker_name": row["worker_name"],

        "cpu_util": safe_float(row.get("cpu_util", row.get("avg_cpu_percent")), 0.0),
        "gpu_util": safe_float(row.get("gpu_util", row.get("avg_gpu_util_percent")), 0.0),
        "throughput": safe_float(row.get("throughput", row.get("samples_per_sec_total")), 0.0),
        "memory_headroom": memory_headroom,
        "time": safe_float(row.get("time", row.get("total_seconds")), 0.0),

        "observed_speed": safe_float(
            row.get("throughput", row.get("samples_per_sec_total")),
            0.0,
        ),
    }


def load_profile_history(profile_csv, worker_names, profile_round):
    profile_csv = Path(profile_csv)

    if not profile_csv.exists():
        raise FileNotFoundError(f"Profile CSV not found: {profile_csv}")

    df = pd.read_csv(profile_csv)

    if "worker_name" not in df.columns:
        raise RuntimeError("profile_csv must contain worker_name column")

    if "round" not in df.columns:
        raise RuntimeError("profile_csv must contain round column")

    history_rows = []

    for _, row in df.iterrows():
        row = row.to_dict()

        if row.get("worker_name") in worker_names:
            history_rows.append(row_to_feature(row))

    if len(history_rows) < len(worker_names):
        raise RuntimeError("Not enough profiling history rows found.")

    if profile_round == "latest":
        selected_round = int(df["round"].max())
    else:
        selected_round = int(profile_round)

    df_round = df[df["round"].astype(int) == selected_round].copy()

    latest_rows = []

    for name in worker_names:
        worker_df = df_round[df_round["worker_name"] == name]

        if len(worker_df) == 0:
            raise RuntimeError(f"No profiling row found for worker {name} in round {selected_round}")

        latest_rows.append(row_to_feature(worker_df.iloc[-1].to_dict()))

    return selected_round, history_rows, latest_rows


def build_manual_model_from_weights(history_rows, manual_weights):
    """
    Build a model object with the same structure as coordinator_adaptive_auto.py.

    Difference:
      adaptive version learns coefficient values from ridge regression.
      this manual version uses user-provided weight magnitudes and fixed directions.

    Same:
      x_mean, x_std, y_mean, y_std
      predict_speed_from_model()
      compute_next_ratios()
    """

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

    coef = {}

    for feature in FEATURES:
        coef[feature] = DEFAULT_DIRECTIONS[feature] * manual_weights[feature]

    model = {
        "weights": manual_weights,

        "coef": coef,

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

        "method": "manual_weights_auto_style_allocation",
    }

    return model


def print_manual_model(model):
    print("\nManual weights using auto-style allocation:")

    for feature in FEATURES:
        coef = model["coef"][feature]
        direction = "positive" if coef >= 0 else "negative"

        print(
            f"  {feature:<16}: "
            f"weight={model['weights'][feature]:.4f}, "
            f"coefficient={coef:+.4f}, "
            f"direction={direction}"
        )


def print_profile_inputs(latest_rows):
    print("\nProfiling inputs used for allocation:")
    print("Worker | CPU Util | GPU Util | Throughput | Memory Headroom | Time")
    print("-------|----------|----------|------------|-----------------|---------")

    for row in latest_rows:
        print(
            f"{row['worker_name']:>6} | "
            f"{row['cpu_util']:>8.2f}% | "
            f"{row['gpu_util']:>8.2f}% | "
            f"{row['throughput']:>10.3f} | "
            f"{row['memory_headroom']:>15.2f}% | "
            f"{row['time']:>7.3f}s"
        )


def print_ratios(title, ratios):
    print(title)

    for name, ratio in ratios.items():
        print(f"  {name}: {ratio * 100:.2f}%")


def print_estimated_speed(worker_names, model, latest_rows):
    print("\nEstimated worker speed using manual weights:")

    row_by_worker = {
        row["worker_name"]: row
        for row in latest_rows
    }

    speeds = {}

    for name in worker_names:
        speed = predict_speed_from_model(model, row_by_worker[name])
        speeds[name] = speed
        print(f"  {name}: {speed:.3f} samples/sec")

    return speeds


def print_training_result(round_results, ratios):
    print("\n========== One-Round Manual Auto-Style Training Result ==========")
    print("Worker | Ratio | Samples | Time(s) | CPU Util | GPU Util | Throughput | Memory Headroom")
    print("-------|-------|---------|---------|----------|----------|------------|----------------")

    rows = []

    for res in round_results:
        name = res["worker_name"]

        avg_memory = safe_float(res.get("avg_memory_percent"), 0.0)
        memory_headroom = max(0.0, 100.0 - avg_memory)

        row = {
            "worker_name": name,
            "ratio": ratios[name],
            "samples": safe_float(res.get("samples"), 0.0),
            "time": safe_float(res.get("total_seconds"), 0.0),
            "cpu_util": safe_float(res.get("avg_cpu_percent"), 0.0),
            "gpu_util": safe_float(res.get("avg_gpu_util_percent"), 0.0),
            "throughput": safe_float(res.get("samples_per_sec_total"), 0.0),
            "memory_headroom": memory_headroom,
        }

        rows.append(row)

        print(
            f"{name:>6} | "
            f"{row['ratio'] * 100:>5.2f}% | "
            f"{int(row['samples']):>7} | "
            f"{row['time']:>7.3f} | "
            f"{row['cpu_util']:>8.2f}% | "
            f"{row['gpu_util']:>8.2f}% | "
            f"{row['throughput']:>10.3f} | "
            f"{row['memory_headroom']:>14.2f}%"
        )

    times = [row["time"] for row in rows]
    gap = max(times) - min(times)

    print("------------------------------------------")
    print(f"Time gap max-min: {gap:.3f} seconds")

    return rows, gap


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", nargs="+", required=True)

    parser.add_argument("--profile-csv", required=True)
    parser.add_argument("--profile-round", default="latest")

    parser.add_argument("--weights", nargs="+", required=True)

    parser.add_argument("--previous-ratios", nargs="*", default=None)
    parser.add_argument("--smoothing", type=float, default=0.70)
    parser.add_argument("--min-ratio", type=float, default=0.05)

    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--work-dir", default="./manual_weight_auto_style_tests")

    parser.add_argument("--model-name", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--cleanup-workers", action="store_true")

    args = parser.parse_args()

    workers = parse_workers(args.workers)
    worker_names = list(workers.keys())

    manual_weights = parse_weights(args.weights)
    previous_ratios = parse_ratios(args.previous_ratios, worker_names)

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
            f"gpu={info.get('gpu_name')}"
        )

    selected_profile_round, history_rows, latest_rows = load_profile_history(
        profile_csv=args.profile_csv,
        worker_names=worker_names,
        profile_round=args.profile_round,
    )

    print(f"\nProfile source: {args.profile_csv}")
    print(f"Profile round used: {selected_profile_round}")

    print_profile_inputs(latest_rows)

    model = build_manual_model_from_weights(
        history_rows=history_rows,
        manual_weights=manual_weights,
    )

    print_manual_model(model)

    print_ratios("\nPrevious ratios used for smoothing:", previous_ratios)

    print_estimated_speed(worker_names, model, latest_rows)

    ratios, estimated_speed, ideal_ratios = compute_next_ratios(
        worker_names=worker_names,
        current_ratios=previous_ratios,
        latest_feature_rows=latest_rows,
        model=model,
        smoothing=args.smoothing,
        min_ratio=args.min_ratio,
    )

    print_ratios("\nIdeal ratios from manual-weight speed model:", ideal_ratios)
    print_ratios("\nFinal smoothed ratios used for this one-round test:", ratios)

    print("\nScanning dataset on A...")
    df, class_to_idx = scan_dataset(args.data_root)
    num_classes = len(class_to_idx)

    print(f"Total samples: {len(df)}")
    print(f"Classes: {num_classes}")

    with open(work_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    round_dir = work_dir / f"round_{args.round_id}"
    round_dir.mkdir(parents=True, exist_ok=True)

    print("\nSplitting data by manual auto-style ratios...")
    shards, allocations = split_by_ratios(
        df=df,
        ratios=ratios,
        round_id=args.round_id,
    )

    for name in worker_names:
        print(f"  {name}: {ratios[name] * 100:.2f}% -> {allocations[name]} samples")

    print("\nCreating shard zip files on A...")
    shard_zip_paths = {}

    for name, shard_df in shards.items():
        zip_path = round_dir / f"shard_{name}.zip"
        make_shard_zip(shard_df, zip_path)
        shard_zip_paths[name] = zip_path

    print("\nSending one-round stats-only training jobs.")
    print("No model weights are exchanged.")

    round_results = []
    futures = []

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        for name, url in workers.items():
            job_id = f"manual_auto_style_round_{args.round_id}_{name}"

            fut = executor.submit(
                send_stats_job,
                name,
                url,
                job_id,
                args.round_id,
                shard_zip_paths[name],
                args,
                num_classes,
            )

            futures.append((name, url, job_id, fut))

        for name, url, job_id, fut in futures:
            res = fut.result()
            round_results.append(res)

    training_rows, gap = print_training_result(round_results, ratios)

    metric_rows = []

    for res in round_results:
        name = res["worker_name"]

        avg_memory = safe_float(res.get("avg_memory_percent"), 0.0)
        memory_headroom = max(0.0, 100.0 - avg_memory)

        metric_rows.append({
            "round": args.round_id,
            "profile_round_used": selected_profile_round,
            "worker_name": name,

            "weight_cpu_util": manual_weights["cpu_util"],
            "weight_gpu_util": manual_weights["gpu_util"],
            "weight_throughput": manual_weights["throughput"],
            "weight_memory_headroom": manual_weights["memory_headroom"],
            "weight_time": manual_weights["time"],

            "coef_cpu_util": model["coef"]["cpu_util"],
            "coef_gpu_util": model["coef"]["gpu_util"],
            "coef_throughput": model["coef"]["throughput"],
            "coef_memory_headroom": model["coef"]["memory_headroom"],
            "coef_time": model["coef"]["time"],

            "previous_ratio": previous_ratios[name],
            "ideal_ratio": ideal_ratios[name],
            "final_ratio": ratios[name],

            "allocated_samples": allocations[name],
            "trained_samples": res.get("samples"),

            "cpu_util": safe_float(res.get("avg_cpu_percent"), 0.0),
            "gpu_util": safe_float(res.get("avg_gpu_util_percent"), 0.0),
            "throughput": safe_float(res.get("samples_per_sec_total"), 0.0),
            "memory_headroom": memory_headroom,
            "time": safe_float(res.get("total_seconds"), 0.0),

            "time_gap": gap,

            "train_seconds": res.get("train_seconds"),
            "total_seconds": res.get("total_seconds"),
            "request_seconds": res.get("request_seconds"),
            "samples_per_sec_train": res.get("samples_per_sec_train"),
            "samples_per_sec_total": res.get("samples_per_sec_total"),
            "iteration_time": res.get("iteration_time"),

            "avg_cpu_percent": res.get("avg_cpu_percent"),
            "avg_gpu_util_percent": res.get("avg_gpu_util_percent"),
            "avg_memory_percent": res.get("avg_memory_percent"),
        })

        if args.cleanup_workers:
            cleanup_worker_job(res["worker_url"], res["job_id"])

    fieldnames = [
        "round",
        "profile_round_used",
        "worker_name",

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

        "previous_ratio",
        "ideal_ratio",
        "final_ratio",

        "allocated_samples",
        "trained_samples",

        "cpu_util",
        "gpu_util",
        "throughput",
        "memory_headroom",
        "time",

        "time_gap",

        "train_seconds",
        "total_seconds",
        "request_seconds",
        "samples_per_sec_train",
        "samples_per_sec_total",
        "iteration_time",

        "avg_cpu_percent",
        "avg_gpu_util_percent",
        "avg_memory_percent",
    ]

    append_csv(
        work_dir / "manual_weight_auto_style_results.csv",
        metric_rows,
        fieldnames,
    )

    print("\nSaved:")
    print(f"  {work_dir / 'manual_weight_auto_style_results.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
