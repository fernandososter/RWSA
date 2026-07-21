from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
)

def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"

    if isinstance(value, int):
        return f"{value:,}"

    if value is None:
        return "None"

    return str(value)


def print_section(
    title: str,
    values: Mapping[str, Any],
    *,
    width: int = 80,
) -> None:
    print(title)
    print("-" * width)

    if not values:
        print("Nenhuma informação disponível.")
        print()
        return

    key_width = max(len(str(key)) for key in values)

    for key, value in values.items():
        print(
            f"{str(key):<{key_width}} : "
            f"{_format_value(value)}"
        )

    print()


def print_experiment_summary(
    *,
    model: nn.Module,
    model_name: str,
    experiment_name: str,
    device: str,
    training_config: Mapping[str, Any] | None = None,
    dataset_config: Mapping[str, Any] | None = None,
    cross_validation_config: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    show_structure: bool = True,
    width: int = 80,
) -> None:
    print()
    print("=" * width)
    print("EXPERIMENT SUMMARY")
    print("=" * width)
    print()

    if hasattr(model, "model_summary"):
        model_information = model.model_summary()
    else:
        model_information = {
            "model_name": model_name,
            "class_name": model.__class__.__name__,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
        }

    general_information = {
        "experiment": experiment_name,
        "device": device,
    }

    print_section(
        "General",
        general_information,
        width=width,
    )

    print_section(
        "Model",
        model_information,
        width=width,
    )

    if model_config:
        print_section(
            "Model configuration",
            model_config,
            width=width,
        )

    if training_config:
        print_section(
            "Training configuration",
            training_config,
            width=width,
        )

    if dataset_config:
        print_section(
            "Dataset",
            dataset_config,
            width=width,
        )

    if cross_validation_config:
        print_section(
            "Cross-validation",
            cross_validation_config,
            width=width,
        )

    if show_structure:
        print("Model structure")
        print("-" * width)
        print(model)
        print()

    print("=" * width)

def print_split_summary(
    *,
    split_name,
    subjects,
    dataset,
    loader,
    width=80,
):
    values = {
        "subjects": len(subjects),
        **dataset.summary(),
        "batches": len(loader),
        "batch_size": loader.batch_size,
        "shuffle": isinstance(
            loader.sampler,
            RandomSampler,
        ),
        "num_workers": loader.num_workers,
    }

    print_section(
        f"{split_name} split",
        values,
        width=width,
    )

def print_stage_distribution(
    title: str,
    distribution: Mapping[
        str,
        Mapping[str, Any],
    ],
    *,
    width: int = 80,
) -> None:
    print(title)
    print("-" * width)

    total = sum(
        int(values["count"])
        for values in distribution.values()
    )

    print(f"{'Stage':<10} {'Count':>15} {'Percentage':>15}")
    print("-" * 42)

    for stage, values in distribution.items():
        count = int(values["count"])
        percentage = float(values["percentage"])

        print(
            f"{stage:<10} "
            f"{count:>15,} "
            f"{percentage:>14.2f}%"
        )

    print("-" * 42)
    print(f"{'Total':<10} {total:>15,}")
    print()