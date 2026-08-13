"""Continue WLASL landmark-subset tuning from a completed batch's best models.

For each of the seven landmark subsets, this runs five new variants derived
from that subset's previous winning config, then selects across the previous
winner and all new trials by validation macro F1.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.encoder_data import VALID_LANDMARK_SUBSETS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def trial_config(landmark_subset: str, old_result: dict[str, Any], old_config: dict[str, Any], variant: str, updates: dict[str, Any]) -> dict[str, Any]:
    epochs = min(140, max(70, int(old_result["best_epoch"]) + 35))
    config: dict[str, Any] = {
        "name": f"{landmark_subset}__phase2_{variant}",
        "landmark_subset": landmark_subset,
        "epochs": epochs,
        "early_stopping_patience": min(32, max(18, round(epochs * 0.25))),
        "tuning_basis": {
            "previous_best_run": old_result["source_run_dir"],
            "previous_best_val_macro_f1": old_result["best_val_macro_f1"],
            "previous_config": {key: old_config[key] for key in ("d_model", "num_heads", "num_layers", "dim_feedforward", "dropout", "batch_size", "learning_rate", "weight_decay", "label_smoothing", "seed") if key in old_config},
            "selection_rule": "Second tuning phase: refine LR/regularization, capacity/depth, label smoothing and initialization around the prior winner.",
        },
        **{key: old_config[key] for key in ("d_model", "num_heads", "num_layers", "dim_feedforward", "dropout", "batch_size", "learning_rate", "weight_decay", "label_smoothing", "seed", "scheduler", "use_class_weight", "balanced_sampler", "augment_train", "augment_noise_std", "standardize", "num_workers", "cpu_threads", "prefetch_factor", "pin_memory", "use_amp", "compile_model") if key in old_config},
        **updates,
    }
    return config


def make_trials(previous_batch: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    previous = read_json(previous_batch / "best_by_landmark_subset.json")
    by_subset = {row["landmark_subset"]: row for row in previous}
    prior_results: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for landmark_subset in VALID_LANDMARK_SUBSETS:
        old_result = {**by_subset[landmark_subset], **read_json(Path(by_subset[landmark_subset]["source_run_dir"]) / "summary.json")}
        old_config = read_json(Path(old_result["best_model_dir"]) / "config.json")
        prior_results.append(old_result)
        lr = float(old_config["learning_rate"])
        dropout = float(old_config["dropout"])
        depth = int(old_config["num_layers"])
        d_model = int(old_config["d_model"])
        batch = int(old_config["batch_size"])
        variants = (
            ("fine_lr", {"learning_rate": lr * 0.65, "weight_decay": max(3e-4, float(old_config["weight_decay"]) * 0.7)}),
            ("higher_regularization", {"learning_rate": lr * 0.8, "dropout": min(0.45, dropout + 0.10), "weight_decay": min(5e-3, float(old_config["weight_decay"]) * 2)}),
            ("label_smoothing", {"learning_rate": lr * 0.8, "label_smoothing": 0.05, "dropout": min(0.40, dropout + 0.05)}),
            ("more_capacity", {"d_model": min(512, d_model + 128), "num_heads": 8, "dim_feedforward": min(1536, max(int(old_config["dim_feedforward"]), (min(512, d_model + 128) * 2))), "batch_size": min(batch, 16), "learning_rate": min(lr, 1.25e-4)}),
            ("deeper_new_seed", {"num_layers": min(5, depth + 1), "dim_feedforward": min(1536, max(int(old_config["dim_feedforward"]), d_model * 3)), "batch_size": min(batch, 16), "seed": int(old_config["seed"]) + 701, "learning_rate": lr * 0.75}),
        )
        trials.extend(trial_config(landmark_subset, old_result, old_config, name, values) for name, values in variants)
    return prior_results, trials


def collect(dataset_root: Path) -> list[dict[str, Any]]:
    candidates = []
    for path in dataset_root.glob("*/runs/*/summary.json"):
        summary = read_json(path)
        summary["source_run_dir"] = summary["run_dir"]
        candidates.append(summary)
    winners: list[dict[str, Any]] = []
    best_root = dataset_root / "best_models"
    for landmark_subset in VALID_LANDMARK_SUBSETS:
        selected = [row for row in candidates if read_json(Path(row["source_run_dir"]) / "config.json").get("landmark_subset") == landmark_subset]
        winner = max(selected, key=lambda row: (float(row["best_val_macro_f1"]), float(row.get("test_macro_f1", -1))))
        source = Path(winner["source_run_dir"])
        target = best_root / landmark_subset
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("best.pt", "config.json", "runtime.json", "data_context.json", "summary.json", "metrics.csv"):
            shutil.copy2(source / filename, target / filename)
        result = {"landmark_subset": landmark_subset, "best_val_macro_f1": winner["best_val_macro_f1"], "test_macro_f1": winner.get("test_macro_f1"), "source_run_dir": str(source), "best_model_dir": str(target)}
        write_json(target / "selection.json", result)
        winners.append(result)
    write_json(dataset_root / "best_by_landmark_subset.json", winners)
    with (dataset_root / "best_by_landmark_subset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(winners[0])); writer.writeheader(); writer.writerows(winners)
    return winners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, choices=("wlasl100", "wlasl300", "wlasl1000", "wlasl2000"))
    parser.add_argument("--previous-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "encoder_only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    previous_batch = args.previous_batch.resolve()
    if not (previous_batch / "best_by_landmark_subset.json").exists():
        raise FileNotFoundError(f"Not a completed batch: {previous_batch}")
    phase_id = f"phase2_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_root = args.output_dir / args.subset
    experiments_dir = dataset_root / "experiments"
    prior_results, trials = make_trials(previous_batch)
    if args.dry_run:
        print(f"Would write experiment metadata to: {experiments_dir / (phase_id + '_summary.json')}")
        print(f"Would run {len(trials)} new trials (five per landmark subset).")
        return
    experiments_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiments_dir / f"{phase_id}_trials.json"
    write_json(config_path, {"defaults": {}, "trials": trials})
    write_json(experiments_dir / f"{phase_id}_previous_best_models.json", prior_results)
    command = [sys.executable, "scripts/train_encoder_only.py", "--subset", args.subset, "--config", str(config_path), "--device", args.device, "--output-dir", str(args.output_dir)]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    winners = collect(dataset_root)
    write_json(experiments_dir / f"{phase_id}_summary.json", {"subset": args.subset, "previous_batch": str(previous_batch), "new_trial_count": len(trials), "winners": winners})
    print(f"Phase 2 completed: {dataset_root}")


if __name__ == "__main__":
    main()
