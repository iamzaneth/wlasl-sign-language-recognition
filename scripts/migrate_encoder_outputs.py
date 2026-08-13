"""One-time migration from legacy batch/phase folders to the stable output tree."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def all_run_dirs(root: Path) -> list[Path]:
    return [
        path.parent for path in root.rglob("summary.json")
        if path.parent.parent.name == "runs" and (path.parent / "best.pt").exists() and (path.parent / "config.json").exists()
    ]


def rebuild_best(dataset_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for run_dir in all_run_dirs(dataset_root):
        summary = read_json(run_dir / "summary.json")
        summary["run_dir"] = str(run_dir)
        summaries.append(summary)
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        subset = read_json(Path(summary["run_dir"]) / "config.json")["landmark_subset"]
        by_subset.setdefault(subset, []).append(summary)
    results = []
    for subset, candidates in sorted(by_subset.items()):
        winner = max(candidates, key=lambda row: (float(row["best_val_macro_f1"]), float(row.get("test_macro_f1", -1))))
        source = Path(winner["run_dir"])
        target = dataset_root / "best_models" / subset
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("best.pt", "config.json", "runtime.json", "data_context.json", "summary.json", "metrics.csv"):
            shutil.copy2(source / filename, target / filename)
        result = {"landmark_subset": subset, "best_val_macro_f1": winner["best_val_macro_f1"], "test_macro_f1": winner.get("test_macro_f1"), "source_run_dir": str(source), "best_model_dir": str(target)}
        write_json(target / "selection.json", result)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "encoder_only")
    parser.add_argument("--apply", action="store_true", help="Actually move runs and remove the legacy batches folder.")
    args = parser.parse_args()
    dataset_root = args.output_root / args.subset
    legacy_root = args.output_root / "batches"
    legacy_runs = []
    for batch in legacy_root.glob(f"{args.subset}_*"):
        legacy_runs.extend(all_run_dirs(batch))
    print(f"Found {len(legacy_runs)} legacy runs; target: {dataset_root}")
    if not args.apply:
        return
    experiments = dataset_root / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for source in legacy_runs:
        config = read_json(source / "config.json")
        target = dataset_root / config["landmark_subset"] / "runs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.resolve() != source.resolve():
                raise FileExistsError(f"Refusing to overwrite existing run: {target}")
            continue
        shutil.move(str(source), str(target))
        moved.append({"run_id": source.name, "from": str(source), "to": str(target)})
    # Preserve compact experiment provenance before removing the redundant tree.
    for batch in legacy_root.glob(f"{args.subset}_*"):
        metadata = {"legacy_batch_path": str(batch), "files": {}}
        for file in batch.glob("*.json"):
            metadata["files"][file.name] = read_json(file)
        write_json(experiments / f"{batch.name}.json", metadata)
    results = rebuild_best(dataset_root)
    write_json(dataset_root / "best_by_landmark_subset.json", results)
    with (dataset_root / "best_by_landmark_subset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    write_json(experiments / "migration.json", {"moved_run_count": len(moved), "moved_runs": moved})
    # All run artifacts and batch metadata now exist in the stable layout.
    for batch in legacy_root.glob(f"{args.subset}_*"):
        shutil.rmtree(batch)
    if legacy_root.exists() and not any(legacy_root.iterdir()):
        legacy_root.rmdir()
    print(f"Moved {len(moved)} runs. Best models: {dataset_root / 'best_models'}")


if __name__ == "__main__":
    main()
