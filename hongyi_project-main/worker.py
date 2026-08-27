import os

CPU_LIMIT = int(os.environ.get("CPU_LIMIT", os.environ.get("cpu_limit", "0")) or "0")

if CPU_LIMIT > 0:
    os.environ["OMP_NUM_THREADS"] = str(CPU_LIMIT)
    os.environ["MKL_NUM_THREADS"] = str(CPU_LIMIT)
    os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_LIMIT)
    os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_LIMIT)

import csv
import json
import time
import shutil
import zipfile
import socket
import threading
import subprocess
from pathlib import Path

import psutil
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from model import build_model
from dataset import build_loader


app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOSTNAME = socket.gethostname()

WORK_ROOT = os.environ.get("WORKER_ROOT", str(Path.home() / "dist_resnet_worker"))
Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)


def apply_cpu_limit():
    if CPU_LIMIT <= 0:
        return

    try:
        available = sorted(list(os.sched_getaffinity(0)))
        selected = set(available[:min(CPU_LIMIT, len(available))])
        os.sched_setaffinity(0, selected)
    except Exception:
        pass

    try:
        torch.set_num_threads(CPU_LIMIT)
        torch.set_num_interop_threads(max(1, min(CPU_LIMIT, 4)))
    except Exception:
        pass


apply_cpu_limit()


def effective_num_workers(requested_num_workers: int):
    """
    If A sends num_workers=0, worker decides automatically.
    If CPU_LIMIT is set, use CPU_LIMIT.
    If CPU_LIMIT is not set, use current CPU affinity count.
    """

    if requested_num_workers is None or requested_num_workers <= 0:
        if CPU_LIMIT > 0:
            return CPU_LIMIT

        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            return psutil.cpu_count(logical=True) or 1

    return requested_num_workers


def save_upload_file(upload_file: UploadFile, destination: Path):
    with open(destination, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)


def get_gpu_info():
    if not torch.cuda.is_available():
        return {
            "cuda": False,
            "gpu_name": "CPU",
            "gpu_total_memory_gb": 0.0,
        }

    props = torch.cuda.get_device_properties(0)

    return {
        "cuda": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gb": props.total_memory / (1024 ** 3),
    }


def read_nvidia_smi():
    """
    Read current GPU utilisation and memory usage using nvidia-smi.
    """

    if not torch.cuda.is_available():
        return {
            "gpu_util_percent": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_memory_percent": 0.0,
        }

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]

        out = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("utf-8").strip()

        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]

        gpu_util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
        mem_percent = 100.0 * mem_used / max(mem_total, 1e-8)

        return {
            "gpu_util_percent": gpu_util,
            "gpu_memory_used_mb": mem_used,
            "gpu_memory_total_mb": mem_total,
            "gpu_memory_percent": mem_percent,
        }

    except Exception:
        return {
            "gpu_util_percent": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_memory_percent": 0.0,
        }


class UtilisationMonitor:
    """
    Monitor CPU, memory, GPU utilisation during local training.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None

        self.cpu_values = []
        self.memory_values = []
        self.gpu_util_values = []
        self.gpu_memory_used_values = []
        self.gpu_memory_percent_values = []

    def _run(self):
        psutil.cpu_percent(interval=None)

        while not self.stop_event.is_set():
            try:
                self.cpu_values.append(float(psutil.cpu_percent(interval=None)))
            except Exception:
                pass

            try:
                self.memory_values.append(float(psutil.virtual_memory().percent))
            except Exception:
                pass

            gpu = read_nvidia_smi()
            self.gpu_util_values.append(float(gpu["gpu_util_percent"]))
            self.gpu_memory_used_values.append(float(gpu["gpu_memory_used_mb"]))
            self.gpu_memory_percent_values.append(float(gpu["gpu_memory_percent"]))

            time.sleep(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=5)

    @staticmethod
    def _avg(values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _max(values):
        if not values:
            return 0.0
        return max(values)

    def summary(self):
        return {
            "avg_cpu_percent": self._avg(self.cpu_values),
            "max_cpu_percent": self._max(self.cpu_values),

            "avg_memory_percent": self._avg(self.memory_values),
            "max_memory_percent": self._max(self.memory_values),

            "avg_gpu_util_percent": self._avg(self.gpu_util_values),
            "max_gpu_util_percent": self._max(self.gpu_util_values),

            "avg_gpu_memory_used_mb": self._avg(self.gpu_memory_used_values),
            "max_gpu_memory_used_mb": self._max(self.gpu_memory_used_values),

            "avg_gpu_memory_percent": self._avg(self.gpu_memory_percent_values),
            "max_gpu_memory_percent": self._max(self.gpu_memory_percent_values),
        }


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def make_autocast(use_amp: bool):
    try:
        return torch.amp.autocast("cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.autocast(enabled=use_amp)


@app.get("/health")
def health():
    mem = psutil.virtual_memory()
    gpu = get_gpu_info()
    gpu_runtime = read_nvidia_smi()

    try:
        affinity = sorted(list(os.sched_getaffinity(0)))
    except Exception:
        affinity = []

    return {
        "hostname": HOSTNAME,
        "device": DEVICE,

        "cpu_limit": CPU_LIMIT,
        "cpu_affinity": affinity,
        "cpu_count": psutil.cpu_count(logical=True),

        "memory_total_gb": mem.total / (1024 ** 3),
        "memory_available_gb": mem.available / (1024 ** 3),
        "memory_percent": mem.percent,

        "work_root": WORK_ROOT,

        **gpu,
        **gpu_runtime,
    }


@app.post("/train_zip_stats")
async def train_zip_stats(
    job_id: str = Form(...),
    round_id: int = Form(...),
    num_classes: int = Form(...),
    model_name: str = Form("resnet18"),
    batch_size: int = Form(64),
    lr: float = Form(0.01),
    local_epochs: int = Form(1),
    num_workers: int = Form(0),
    amp: bool = Form(True),
    data_zip: UploadFile = File(...),
):
    """
    Stats-only training endpoint.

    A sends only data.zip.
    Worker trains a local ResNet from scratch.
    Worker returns only runtime statistics.
    No global model is received.
    No local model is returned.
    No FedAvg is performed.
    """

    job_dir = Path(WORK_ROOT) / job_id

    if job_dir.exists():
        shutil.rmtree(job_dir)

    job_dir.mkdir(parents=True, exist_ok=True)

    zip_path = job_dir / "data.zip"
    data_dir = job_dir / "data"

    save_upload_file(data_zip, zip_path)

    unzip_start = time.time()

    data_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    unzip_seconds = time.time() - unzip_start

    manifest_csv = data_dir / "manifest.csv"

    if not manifest_csv.exists():
        raise RuntimeError(f"manifest.csv not found in uploaded zip: {zip_path}")

    actual_num_workers = effective_num_workers(num_workers)

    loader = build_loader(
        root_dir=str(data_dir),
        manifest_csv=str(manifest_csv),
        batch_size=batch_size,
        num_workers=actual_num_workers,
        shuffle=True,
    )

    model = build_model(
        num_classes=num_classes,
        model_name=model_name,
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    use_amp = DEVICE == "cuda" and amp
    scaler = make_grad_scaler(use_amp)

    model.train()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0
    total_batches = 0

    monitor = UtilisationMonitor(interval=0.5)

    train_start = time.time()
    monitor.start()

    try:
        for _ in range(local_epochs):
            for x, y in loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with make_autocast(use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = x.size(0)

                total_samples += bs
                total_batches += 1
                total_loss += loss.item() * bs
                total_correct += (logits.argmax(dim=1) == y).sum().item()

    finally:
        monitor.stop()

    train_seconds = time.time() - train_start

    total_seconds = unzip_seconds + train_seconds
    util_summary = monitor.summary()

    result = {
        "job_id": job_id,
        "round_id": round_id,
        "hostname": HOSTNAME,
        "device": DEVICE,
        "mode": "stats_only_no_weight_sync",

        "cpu_limit": CPU_LIMIT,
        "actual_num_workers": actual_num_workers,

        "samples": total_samples,
        "batches": total_batches,

        "unzip_seconds": unzip_seconds,
        "train_seconds": train_seconds,
        "total_seconds": total_seconds,

        "samples_per_sec_train": total_samples / max(train_seconds, 1e-8),
        "samples_per_sec_total": total_samples / max(total_seconds, 1e-8),
        "iteration_time": train_seconds / max(total_batches, 1),

        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),

        **util_summary,
    }

    with open(job_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return JSONResponse(result)


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    job_dir = Path(WORK_ROOT) / job_id

    if job_dir.exists():
        shutil.rmtree(job_dir)

    return {
        "deleted": job_id
    }
