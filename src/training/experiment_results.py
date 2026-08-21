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
    "config.json", "runtime.json", "data_context.json", "metrics.csv",
    "summary.json", "best.pt",
)
RUN_CHECKPOINT_GLOB = "*/runs/*/*.pt"


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


def prune_run_checkpoints(dataset_root: Path) -> dict[str, int]:
    """Delete run checkpoints after their selected models have been published.

    Run metadata remains in place so completed trials can still be skipped and
    adaptive tuning can reconstruct its search state.
    """
    checkpoint_paths = list(dataset_root.glob(RUN_CHECKPOINT_GLOB))
    removed_bytes = sum(path.stat().st_size for path in checkpoint_paths)
    affected_run_dirs = {path.parent for path in checkpoint_paths}
    for path in checkpoint_paths:
        path.unlink()

    for run_dir in affected_run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        artifacts = summary.get("artifacts")
        if isinstance(artifacts, list):
            summary["artifacts"] = [
                artifact for artifact in artifacts
                if Path(str(artifact)).suffix.casefold() != ".pt"
            ]
        summary["checkpoint_retention"] = "metadata_only_after_publication"
        write_json(summary_path, summary)

    return {"removed_files": len(checkpoint_paths), "removed_bytes": removed_bytes}


def _can_reuse_published_checkpoint(target: Path, source: Path) -> bool:
    selection_path = target / "selection.json"
    checkpoint_path = target / "best.pt"
    if not selection_path.exists() or not checkpoint_path.exists():
        return False
    try:
        selected_source = resolve_project_path(read_json(selection_path)["source_run_dir"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return selected_source.resolve() == source.resolve()


def rebuild_best(
    dataset_root: Path,
    *,
    prune_checkpoints: bool = True,
) -> list[dict[str, Any]]:
    """Select and publish each best run, then retain metadata-only run history."""
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

    publication_plan: list[tuple[str, dict[str, Any], Path, Path, bool]] = []
    for subset in VALID_LANDMARK_SUBSETS:
        winner = max(grouped[subset], key=lambda row: float(row["best_val_macro_f1"]))
        source = resolve_project_path(winner["source_run_dir"])
        target = dataset_root / "best_models" / subset
        missing_metadata = [
            filename for filename in PUBLISHED_ARTIFACTS
            if filename != "best.pt" and not (source / filename).exists()
        ]
        if missing_metadata:
            raise FileNotFoundError(
                f"Winner {source} is missing required metadata: {', '.join(missing_metadata)}"
            )
        reuse_published_checkpoint = not (source / "best.pt").exists()
        if reuse_published_checkpoint and not _can_reuse_published_checkpoint(target, source):
            raise FileNotFoundError(
                f"Winner checkpoint is unavailable at {source / 'best.pt'} and the "
                f"published checkpoint at {target / 'best.pt'} does not match this run"
            )
        publication_plan.append((subset, winner, source, target, reuse_published_checkpoint))

    winners: list[dict[str, Any]] = []
    for subset, winner, source, target, reuse_published_checkpoint in publication_plan:
        target.mkdir(parents=True, exist_ok=True)
        for filename in PUBLISHED_ARTIFACTS:
            if filename == "best.pt" and reuse_published_checkpoint:
                continue
            shutil.copy2(source / filename, target / filename)
        published_summary = read_json(target / "summary.json")
        published_summary["artifacts"] = [
            filename for filename in PUBLISHED_ARTIFACTS
            if filename != "summary.json" and (target / filename).exists()
        ]
        published_summary["checkpoint_retention"] = "published_best_only"
        write_json(target / "summary.json", published_summary)
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
    if prune_checkpoints:
        prune_run_checkpoints(dataset_root)
    return winners
