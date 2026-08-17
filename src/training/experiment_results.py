"""Shared helpers for reading training artifacts and publishing best models."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from src.config.paths import normalize_project_paths, project_relative_path, resolve_project_path
from src.training.encoder_data import VALID_LANDMARK_SUBSETS


TUNABLE_KEYS = (
    "d_model", "num_heads", "num_layers", "dim_feedforward", "dropout",
    "batch_size", "learning_rate", "weight_decay", "label_smoothing", "seed",
    "scheduler", "warmup_epochs", "use_class_weight", "balanced_sampler",
    "augment_train", "augment_noise_std", "standardize", "num_workers",
    "cpu_threads", "prefetch_factor", "pin_memory", "use_amp", "compile_model",
)

PUBLISHED_ARTIFACTS = (
    "best.pt", "config.json", "runtime.json", "data_context.json",
    "summary.json", "metrics.csv",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Atomically write portable project metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(normalize_project_paths(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def rebuild_best(dataset_root: Path) -> list[dict[str, Any]]:
    """Select and publish the highest-validation run for every landmark subset."""
    grouped: dict[str, list[dict[str, Any]]] = {
        subset: [] for subset in VALID_LANDMARK_SUBSETS
    }
    for summary_path in dataset_root.glob("*/runs/*/summary.json"):
        config_path = summary_path.parent / "config.json"
        if not config_path.exists():
            continue
        subset = read_json(config_path).get("landmark_subset")
        if subset not in grouped:
            continue
        summary = read_json(summary_path)
        summary["source_run_dir"] = project_relative_path(summary_path.parent)
        grouped[subset].append(summary)

    missing = [subset for subset, candidates in grouped.items() if not candidates]
    if missing:
        raise RuntimeError(f"No completed runs found for: {', '.join(missing)}")

    winners: list[dict[str, Any]] = []
    for subset in VALID_LANDMARK_SUBSETS:
        winner = max(grouped[subset], key=lambda row: float(row["best_val_macro_f1"]))
        source = resolve_project_path(winner["source_run_dir"])
        target = dataset_root / "best_models" / subset
        target.mkdir(parents=True, exist_ok=True)
        for filename in PUBLISHED_ARTIFACTS:
            shutil.copy2(source / filename, target / filename)
        selected = {
            "landmark_subset": subset,
            "best_val_macro_f1": winner["best_val_macro_f1"],
            "test_macro_f1": winner.get("test_macro_f1"),
            "source_run_dir": project_relative_path(source),
            "best_model_dir": project_relative_path(target),
        }
        write_json(target / "selection.json", selected)
        winners.append(selected)

    write_json(dataset_root / "best_by_landmark_subset.json", winners)
    csv_path = dataset_root / "best_by_landmark_subset.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(winners[0]))
        writer.writeheader()
        writer.writerows(winners)
    temporary.replace(csv_path)
    return winners
