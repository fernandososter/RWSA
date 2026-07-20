from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


class ValidationPredictionLogger:
    """Grava expected/prediction de todas as validações de um fold.

    Durante o treinamento, cada época é persistida como um fragmento Parquet.
    No fechamento do logger, os fragmentos são consolidados, em streaming, em
    um único ``validation_predictions.parquet``.

    Cada linha representa uma mini-época válida de 3 segundos.
    """

    COLUMNS = (
        "fold",
        "epoch",
        "subject_id",
        "mini_epoch_index",
        "expected",
        "prediction",
    )

    SCHEMA = pa.schema(
        [
            pa.field("fold", pa.int16(), nullable=False),
            pa.field("epoch", pa.int16(), nullable=False),
            pa.field("subject_id", pa.string(), nullable=False),
            pa.field("mini_epoch_index", pa.int32(), nullable=False),
            pa.field("expected", pa.int8(), nullable=False),
            pa.field("prediction", pa.int8(), nullable=False),
        ]
    )

    def __init__(
        self,
        fold_dir: str | Path,
        *,
        fold: int,
        compression: str = "zstd",
        keep_epoch_parts: bool = False,
    ) -> None:
        self.fold = int(fold)
        self.fold_dir = Path(fold_dir)
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.fold_dir / "validation_predictions.parquet"
        self.parts_dir = self.fold_dir / ".validation_prediction_parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.keep_epoch_parts = keep_epoch_parts
        self._closed = False
        self._active_epoch: int | None = None
        self._chunks: list[dict[str, np.ndarray | list[str]]] = []

        if self.output_path.exists():
            raise FileExistsError(
                f"O arquivo de predições já existe: {self.output_path}. "
                "Use uma nova run para evitar misturar execuções."
            )

    def start_epoch(self, epoch: int) -> None:
        if self._active_epoch is not None:
            raise RuntimeError(
                f"A época {self._active_epoch} ainda não foi finalizada."
            )
        self._active_epoch = int(epoch)
        self._chunks.clear()

    def log_staging_batch(
        self,
        *,
        subject_ids: Sequence[str],
        valid_mask: torch.Tensor,
        expected: torch.Tensor,
        prediction: torch.Tensor,
    ) -> None:
        """Adiciona as predições válidas de um batch de staging."""
        if self._active_epoch is None:
            raise RuntimeError("Chame start_epoch(epoch) antes de registrar batches.")

        valid = valid_mask.detach().to(device="cpu", dtype=torch.bool)
        y_true = expected.detach().to(device="cpu")
        y_pred = prediction.detach().to(device="cpu")

        if valid.ndim != 2:
            raise ValueError(f"valid_mask deve ter shape [B,T], recebeu {tuple(valid.shape)}")
        if y_true.shape != valid.shape or y_pred.shape != valid.shape:
            raise ValueError(
                "expected, prediction e valid_mask devem possuir o mesmo shape [B,T]."
            )
        if len(subject_ids) != valid.shape[0]:
            raise ValueError("subject_ids deve possuir um item por elemento do batch.")

        fold_values: list[np.ndarray] = []
        epoch_values: list[np.ndarray] = []
        subjects: list[str] = []
        indices: list[np.ndarray] = []
        expected_values: list[np.ndarray] = []
        prediction_values: list[np.ndarray] = []

        for batch_index, subject_id in enumerate(subject_ids):
            mini_indices = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            if mini_indices.numel() == 0:
                continue
            n = int(mini_indices.numel())
            idx_np = mini_indices.numpy().astype(np.int32, copy=False)
            fold_values.append(np.full(n, self.fold, dtype=np.int16))
            epoch_values.append(np.full(n, self._active_epoch, dtype=np.int16))
            subjects.extend([str(subject_id)] * n)
            indices.append(idx_np)
            expected_values.append(
                y_true[batch_index, mini_indices].numpy().astype(np.int8, copy=False)
            )
            prediction_values.append(
                y_pred[batch_index, mini_indices].numpy().astype(np.int8, copy=False)
            )

        if not indices:
            return

        self._chunks.append(
            {
                "fold": np.concatenate(fold_values),
                "epoch": np.concatenate(epoch_values),
                "subject_id": subjects,
                "mini_epoch_index": np.concatenate(indices),
                "expected": np.concatenate(expected_values),
                "prediction": np.concatenate(prediction_values),
            }
        )

    def end_epoch(self) -> Path:
        """Persiste a validação da época atual em um fragmento Parquet."""
        if self._active_epoch is None:
            raise RuntimeError("Nenhuma época de validação está ativa.")
        if not self._chunks:
            raise RuntimeError(
                f"Nenhuma predição foi registrada para a época {self._active_epoch}."
            )

        arrays: dict[str, Any] = {}
        for column in self.COLUMNS:
            values = [chunk[column] for chunk in self._chunks]
            if column == "subject_id":
                arrays[column] = [item for group in values for item in group]
            else:
                arrays[column] = np.concatenate(values)

        table = pa.Table.from_pydict(arrays, schema=self.SCHEMA)
        part_path = self.parts_dir / f"epoch_{self._active_epoch:04d}.parquet"
        if part_path.exists():
            raise FileExistsError(
                f"Já existem predições para fold={self.fold}, epoch={self._active_epoch}: "
                f"{part_path}"
            )
        pq.write_table(table, part_path, compression=self.compression)

        self._active_epoch = None
        self._chunks.clear()
        return part_path

    def consolidate(self) -> Path:
        """Consolida todos os fragmentos em um único Parquet por fold."""
        if self._active_epoch is not None:
            raise RuntimeError(
                f"Finalize a época {self._active_epoch} antes de consolidar."
            )

        parts = sorted(self.parts_dir.glob("epoch_*.parquet"))
        if not parts:
            raise RuntimeError(f"Nenhum fragmento de validação encontrado em {self.parts_dir}")

        temporary_output = self.output_path.with_suffix(".parquet.tmp")
        writer: pq.ParquetWriter | None = None
        try:
            writer = pq.ParquetWriter(
                temporary_output,
                self.SCHEMA,
                compression=self.compression,
            )
            for part in parts:
                parquet_file = pq.ParquetFile(part)
                for batch in parquet_file.iter_batches():
                    writer.write_table(pa.Table.from_batches([batch], schema=self.SCHEMA))
        finally:
            if writer is not None:
                writer.close()

        temporary_output.replace(self.output_path)
        if not self.keep_epoch_parts:
            shutil.rmtree(self.parts_dir)
        return self.output_path

    def close(self) -> Path:
        if self._closed:
            return self.output_path
        path = self.consolidate()
        self._closed = True
        return path

    def __enter__(self) -> "ValidationPredictionLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Em caso de falha, preserva os fragmentos já concluídos. O arquivo final
        # é criado somente quando o fold termina corretamente.
        if exc_type is None:
            self.close()
