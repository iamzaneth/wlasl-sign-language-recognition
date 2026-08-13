"""Generate shoulder-normalised XYZ data and recover only short hand gaps."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config.paths import INTERIM_LANDMARK_DIR, METADATA_DIR, PROCESSED_LANDMARK_DIR, SUBSETS, split_dir
from src.preprocessing.hand_recovery import recover_hands
from src.preprocessing.landmark_io import is_valid_npz, load_npz, save_npz
from src.preprocessing.normalize_landmarks import shoulder_normalize


def _update_report(video_id: str, payload: dict[str, Any]) -> None:
    """Enrich extraction rows with post-processing quality instead of replacing them."""
    from src.preprocessing.extract_landmarks import REPORT_FIELDS
    path = METADATA_DIR / "landmark_extraction_report.csv"
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = {row["video_id"]: row for row in csv.DictReader(handle)}
    row = rows.get(video_id, {key: "" for key in REPORT_FIELDS})
    row.update({"video_id": video_id, "left_hand_imputed_frames": int(np.asarray(payload["left_hand_imputed"]).sum()), "right_hand_imputed_frames": int(np.asarray(payload["right_hand_imputed"]).sum()), "left_hand_unresolved_frames": int(np.asarray(payload["left_hand_unresolved"]).sum()), "right_hand_unresolved_frames": int(np.asarray(payload["right_hand_unresolved"]).sum()), "normalization_invalid_frames": int((~np.asarray(payload["normalization_valid"])).sum())})
    rows[video_id] = row
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS); writer.writeheader(); writer.writerows(rows[key] for key in sorted(rows))


def _ids(subset: str) -> list[str]:
    result: set[str] = set()
    for partition in ("train", "val", "test"):
        with (split_dir(subset) / f"{partition}.csv").open(newline="", encoding="utf-8") as handle:
            result.update(row["video_id"] for row in csv.DictReader(handle))
    return sorted(result)


def preprocess_video(video_id: str, *, max_gap_frames: int = 2, overwrite: bool = False) -> str:
    source, target = INTERIM_LANDMARK_DIR / f"{video_id}.npz", PROCESSED_LANDMARK_DIR / f"{video_id}.npz"
    if target.exists() and is_valid_npz(target, processed=True) and not overwrite:
        return "skipped_existing"
    payload = load_npz(source)
    normalized, valid, scale = shoulder_normalize(payload)
    processed = recover_hands(normalized, max_gap_frames)
    raw_pose = np.asarray(payload["pose_xyz"])
    centers = (raw_pose[:, 1, :2] + raw_pose[:, 2, :2]) / 2.0
    processed.update({"normalization_valid": valid, "normalization_center_xy": centers.astype(np.float32), "xy_normalized": np.asarray(True), "z_normalized": np.asarray(False), "normalization": np.asarray("shoulder_norm_v1"), "hand_recovery": np.asarray("wrist_anchored_linear"), "shoulder_scale": np.asarray(scale, dtype=np.float32)})
    save_npz(target, processed, processed=True)
    _update_report(video_id, processed)
    return "processed"


def run_preprocessing(video_ids: list[str], *, max_gap_frames: int = 2, overwrite: bool = False, show_progress: bool = True) -> dict[str, int]:
    """Preprocess an ID list while allowing a bad NPZ to fail independently."""
    from src.preprocessing.extract_landmarks import _progress
    total, processed, skipped, failed = len(video_ids), 0, 0, 0
    started = time.monotonic()
    print(f"Starting preprocessing for {total:,} video(s). Existing schema-valid outputs will be skipped.")
    for done, video_id in enumerate(video_ids, start=1):
        try:
            outcome = preprocess_video(video_id, max_gap_frames=max_gap_frames, overwrite=overwrite)
            if outcome == "processed": processed += 1
            else: skipped += 1
        except Exception as exc:
            failed += 1
            print(f"\nPreprocess error for {video_id}: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        if show_progress:
            _progress(done, total, processed + skipped, failed, skipped, started, stage="Preprocessing")
    if show_progress:
        print()
    print(f"Finished {total:,} video(s): processed={processed:,}, failed={failed:,}, skipped={skipped:,}.")
    return {"total": total, "processed": processed, "failed": failed, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=SUBSETS, required=True); parser.add_argument("--normalization", default="shoulder_norm_v1", choices=("shoulder_norm_v1",)); parser.add_argument("--hand-recovery", default="wrist_anchored_linear", choices=("wrist_anchored_linear",)); parser.add_argument("--max-gap-frames", type=int, default=2); parser.add_argument("--limit", type=int); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ids = _ids(args.subset); ids = ids if args.limit is None else ids[:args.limit]
    run_preprocessing(ids, max_gap_frames=args.max_gap_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
