from __future__ import annotations

import csv
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _system_metadata(device: torch.device) -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        cuda["device_count"] = torch.cuda.device_count()
        cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return {
        "created_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "selected_device": str(device),
        "cuda": cuda,
        "pid": os.getpid(),
    }


class ExperimentLogger:
    """Persiste tudo que uma execução precisa para comparação e ablation.

    Cada execução cria uma pasta independente contendo configuração, divisão de
    sujeitos, descrição do modelo, métricas por época, log textual e resumo final.
    """

    def __init__(
        self,
        *,
        task: str,
        experiment_name: str,
        root_dir: str | Path,
        device: torch.device,
        args: Mapping[str, Any],
        project_root: str | Path | None = None,
        notes: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in experiment_name)
        self.run_id = f"{timestamp}_{safe_name}"
        self.task = task
        self.root_dir = Path(root_dir)
        self.run_dir = self.root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._rows: list[dict[str, Any]] = []
        self._start = perf_counter()
        self._best: dict[str, Any] | None = None
        self._closed = False

        self.logger = logging.getLogger(f"sleep_rswa.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        file_handler = logging.FileHandler(self.run_dir / "training.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)

        project_root_path = Path(project_root or Path.cwd()).resolve()
        self.write_json(
            "run.json",
            {
                "run_id": self.run_id,
                "task": task,
                "experiment_name": experiment_name,
                "notes": notes,
                "tags": list(tags or []),
                "arguments": dict(args),
                "system": _system_metadata(device),
                "git": _git_metadata(project_root_path),
            },
        )
        self.info(f"Run criada em: {self.run_dir}")

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def write_json(self, filename: str, payload: Any) -> None:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )

    def log_model(self, model: torch.nn.Module, *, name: str = "model") -> None:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        payload = {
            "name": name,
            "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "total_parameters": total,
            "trainable_parameters": trainable,
            "representation": str(model),
        }
        self.write_json(f"models/{name}.json", payload)
        (self.run_dir / "models" / f"{name}.txt").write_text(str(model), encoding="utf-8")

    def log_subject_split(self, train_subjects: Sequence[Any], val_subjects: Sequence[Any], *, filename: str = "data_split.json") -> None:
        def describe(subject: Any) -> dict[str, Any]:
            return {
                "subject_id": getattr(subject, "subject_id", None),
                "n_epochs": getattr(subject, "n_epochs", None),
            }

        train = [describe(s) for s in train_subjects]
        val = [describe(s) for s in val_subjects]
        self.write_json(
            filename,
            {
                "train": train,
                "validation": val,
                "n_train_subjects": len(train),
                "n_validation_subjects": len(val),
                "n_train_epochs": sum(int(x["n_epochs"] or 0) for x in train),
                "n_validation_epochs": sum(int(x["n_epochs"] or 0) for x in val),
            },
        )

    def log_epoch(self, row: Mapping[str, Any]) -> None:
        clean = {str(k): _jsonable(v) for k, v in row.items()}
        clean.setdefault("timestamp_utc", _utc_now())
        clean.setdefault("elapsed_total_sec", perf_counter() - self._start)
        self._rows.append(clean)

        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(clean, ensure_ascii=False, allow_nan=True) + "\n")

        fieldnames = sorted({key for item in self._rows for key in item.keys()})
        with (self.run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        self.write_json("history.json", self._rows)

    def mark_best(self, *, epoch: int, monitor: str, value: float) -> None:
        self._best = {"epoch": epoch, "monitor": monitor, "value": float(value)}
        self.write_json("best.json", self._best)
        self.info(f"Novo melhor resultado: {monitor}={value:.6f} na época {epoch}")

    def finalize(self, *, status: str, summary: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        payload = {
            "status": status,
            "task": self.task,
            "finished_at_utc": _utc_now(),
            "duration_sec": perf_counter() - self._start,
            "epochs_logged": len(self._rows),
            "best": self._best,
            "summary": dict(summary or {}),
        }
        self.write_json("summary.json", payload)
        self.info(f"Execução finalizada com status={status}")
        self._closed = True

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is None:
            self.finalize(status="completed")
        else:
            self.logger.exception("Execução interrompida por erro", exc_info=(exc_type, exc, traceback))
            self.finalize(status="failed", summary={"error": repr(exc)})
        return False
