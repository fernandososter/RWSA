from __future__ import annotations

import csv
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import psutil
import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResourceMonitor:
    """Coleta periodicamente uso de CPU, RAM e GPU em um CSV."""

    FIELDNAMES = (
        "timestamp_utc",
        "elapsed_sec",
        "cpu_percent",
        "process_cpu_percent",
        "system_memory_used_mb",
        "system_memory_percent",
        "process_rss_mb",
        "gpu_available",
        "gpu_index",
        "gpu_name",
        "gpu_utilization_percent",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "torch_cuda_allocated_mb",
        "torch_cuda_reserved_mb",
        "torch_cuda_max_allocated_mb",
    )

    def __init__(
        self,
        output_path: str | Path,
        *,
        device: torch.device,
        interval_sec: float = 30.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.device = device
        self.interval_sec = max(float(interval_sec), 1.0)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._process = psutil.Process()
        self._gpu_index = device.index if device.type == "cuda" and device.index is not None else 0
        self._gpu_name = None
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                self._gpu_name = torch.cuda.get_device_name(self._gpu_index)
            except Exception:
                self._gpu_name = None

    def _query_gpu(self) -> dict[str, Any]:
        payload = {
            "gpu_available": int(torch.cuda.is_available() and self.device.type == "cuda"),
            "gpu_index": self._gpu_index if self.device.type == "cuda" else None,
            "gpu_name": self._gpu_name,
            "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "torch_cuda_allocated_mb": None,
            "torch_cuda_reserved_mb": None,
            "torch_cuda_max_allocated_mb": None,
        }
        if not (torch.cuda.is_available() and self.device.type == "cuda"):
            return payload

        try:
            gpu_device = torch.device("cuda", self._gpu_index)
            payload["torch_cuda_allocated_mb"] = round(torch.cuda.memory_allocated(gpu_device) / (1024 ** 2), 3)
            payload["torch_cuda_reserved_mb"] = round(torch.cuda.memory_reserved(gpu_device) / (1024 ** 2), 3)
            payload["torch_cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated(gpu_device) / (1024 ** 2), 3)
            total = torch.cuda.get_device_properties(gpu_device).total_memory / (1024 ** 2)
            payload["gpu_memory_total_mb"] = round(float(total), 3)
        except Exception:
            return payload

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            first_line = result.stdout.strip().splitlines()[0]
            util_str, mem_used_str, mem_total_str = [item.strip() for item in first_line.split(",")]
            payload["gpu_utilization_percent"] = float(util_str)
            payload["gpu_memory_used_mb"] = float(mem_used_str)
            payload["gpu_memory_total_mb"] = float(mem_total_str)
        except Exception:
            pass
        return payload

    def _sample(self) -> dict[str, Any]:
        if self._start_time is None:
            self._start_time = monotonic()
        vm = psutil.virtual_memory()
        row = {
            "timestamp_utc": _utc_now(),
            "elapsed_sec": round(monotonic() - self._start_time, 3),
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "process_cpu_percent": float(self._process.cpu_percent(interval=None)),
            "system_memory_used_mb": round(float(vm.used) / (1024 ** 2), 3),
            "system_memory_percent": float(vm.percent),
            "process_rss_mb": round(float(self._process.memory_info().rss) / (1024 ** 2), 3),
        }
        row.update(self._query_gpu())
        return row

    def _write_header(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            return
        with self.output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.FIELDNAMES))
            writer.writeheader()

    def _append_row(self, row: dict[str, Any]) -> None:
        with self.output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.FIELDNAMES))
            writer.writerow(row)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._append_row(self._sample())
            if self._stop_event.wait(self.interval_sec):
                break

    def start(self) -> "ResourceMonitor":
        self._write_header()
        self._start_time = monotonic()
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        self._append_row(self._sample())
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_sec + 5.0)
        self._thread = None

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
