"""Run baseline and five result-informed tuning trials for every landmark subset.

The runner is intentionally sequential: a laptop GPU is shared by one trial at
a time. All generated files remain below ``output/encoder_only/batches`` and
each child training run still writes its own detailed JSON/CSV artifacts.
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


def read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary_path"] = str(path)
    return payload


def latest_summary(root: Path, subset: str, landmark_subset: str) -> Path:
    candidates = sorted((root / subset / landmark_subset / "runs").glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No summary for {landmark_subset} below {root}")
    return candidates[-1]


def tuning_trials(baselines: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Create five safe model/LR/regularization variants per baseline result."""
    variants = (
        ("low_lr", {"learning_rate": 1.0e-4, "dropout": 0.20, "weight_decay": 5.0e-4, "d_model": 256, "num_heads": 8, "num_layers": 3, "dim_feedforward": 512, "batch_size": 32}),
        ("regularized", {"learning_rate": 1.5e-4, "dropout": 0.35, "weight_decay": 2.0e-3, "d_model": 256, "num_heads": 8, "num_layers": 3, "dim_feedforward": 512, "batch_size": 32}),
        ("wider", {"learning_rate": 1.5e-4, "dropout": 0.25, "weight_decay": 1.0e-3, "d_model": 384, "num_heads": 8, "num_layers": 3, "dim_feedforward": 768, "batch_size": 16}),
        ("deeper", {"learning_rate": 1.5e-4, "dropout": 0.25, "weight_decay": 1.0e-3, "d_model": 256, "num_heads": 8, "num_layers": 4, "dim_feedforward": 1024, "batch_size": 24}),
        ("wide_deep", {"learning_rate": 1.0e-4, "dropout": 0.30, "weight_decay": 1.0e-3, "d_model": 384, "num_heads": 8, "num_layers": 4, "dim_feedforward": 1024, "batch_size": 16}),
    )
    trials: list[dict[str, Any]] = []
    for landmark_subset, baseline in baselines.items():
        baseline_epoch = int(baseline["best_epoch"])
        epochs = min(120, max(30, baseline_epoch + 20))
        patience = min(30, max(12, round(epochs * 0.25)))
        for variant_name, parameters in variants:
            trials.append({
                "name": f"{landmark_subset}__tuned_{variant_name}",
                "landmark_subset": landmark_subset,
                "epochs": epochs,
                "early_stopping_patience": patience,
                "tuning_basis": {
                    "baseline_summary": baseline["summary_path"],
                    "baseline_best_epoch": baseline_epoch,
                    "baseline_best_val_macro_f1": baseline["best_val_macro_f1"],
                    "selection_rule": "Epoch budget follows the baseline best epoch; five variants sweep learning rate, regularization, width and depth.",
                },
                **parameters,
            })
    return trials


def run(command: list[str], *, dry_run: bool) -> None:
    print("\n+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_best(dataset_root: Path) -> list[dict[str, Any]]:
    all_summaries = [read_summary(path) for path in dataset_root.glob("*/runs/*/summary.json")]
    results: list[dict[str, Any]] = []
    best_root = dataset_root / "best_models"
    for landmark_subset in VALID_LANDMARK_SUBSETS:
        candidates = [summary for summary in all_summaries if (Path(summary["run_dir"]) / "config.json").exists() and read_summary(Path(summary["run_dir"]) / "config.json").get("landmark_subset") == landmark_subset]
        if not candidates:
            raise RuntimeError(f"No completed candidate for {landmark_subset}")
        winner = max(candidates, key=lambda summary: (float(summary["best_val_macro_f1"]), float(summary.get("test_macro_f1", -1))))
        source = Path(winner["run_dir"])
        target = best_root / landmark_subset
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("best.pt", "config.json", "runtime.json", "data_context.json", "summary.json", "metrics.csv"):
            shutil.copy2(source / filename, target / filename)
        result = {
            "landmark_subset": landmark_subset,
            "best_val_macro_f1": winner["best_val_macro_f1"],
            "test_macro_f1": winner.get("test_macro_f1"),
            "source_run_dir": str(source),
            "best_model_dir": str(target),
        }
        write_json(target / "selection.json", result)
        results.append(result)
    write_json(dataset_root / "best_by_landmark_subset.json", results)
    with (dataset_root / "best_by_landmark_subset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, choices=("wlasl100", "wlasl300", "wlasl1000", "wlasl2000"))
    parser.add_argument("--baseline-epochs", type=int, default=90)
    parser.add_argument("--existing-pose-summary", type=Path, required=True, help="Completed pose summary used as the existing baseline.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "encoder_only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.baseline_epochs < 2:
        raise ValueError("baseline-epochs must be at least 2")
    batch_id = f"all_landmark_subsets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_root = args.output_dir / args.subset
    experiments_dir = dataset_root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    pose_summary = read_summary(args.existing_pose_summary)
    if Path(pose_summary["run_dir"]).parent.parent.name != "pose":
        raise ValueError("--existing-pose-summary must be a pose run")
    baselines: dict[str, dict[str, Any]] = {"pose": pose_summary}
    remaining = [name for name in VALID_LANDMARK_SUBSETS if name != "pose"]
    baseline_command = [sys.executable, "scripts/train_encoder_only.py", "--subset", args.subset, "--landmark-subsets", *remaining, "--epochs", str(args.baseline_epochs), "--device", args.device, "--output-dir", str(args.output_dir)]
    run(baseline_command, dry_run=args.dry_run)
    if args.dry_run:
        return
    for landmark_subset in remaining:
        baselines[landmark_subset] = read_summary(latest_summary(args.output_dir, args.subset, landmark_subset))
    write_json(experiments_dir / f"{batch_id}_baseline_summaries.json", baselines)
    tune_config = {"defaults": {}, "trials": tuning_trials(baselines)}
    tune_config_path = experiments_dir / f"{batch_id}_tuned_trials.json"
    write_json(tune_config_path, tune_config)
    run([sys.executable, "scripts/train_encoder_only.py", "--subset", args.subset, "--config", str(tune_config_path), "--device", args.device, "--output-dir", str(args.output_dir)], dry_run=False)
    winners = collect_best(dataset_root)
    write_json(experiments_dir / f"{batch_id}_summary.json", {"batch_id": batch_id, "subset": args.subset, "baseline_epochs": args.baseline_epochs, "baseline_pose_summary": str(args.existing_pose_summary), "num_tuned_trials": len(tune_config["trials"]), "winners": winners})
    print(f"Batch completed: {dataset_root}")


if __name__ == "__main__":
    main()
