"""Create WLASL inventory, valid/invalid manifests, and dataset metadata.

The command reads immutable files below ``data/raw`` and writes only generated
artefacts. It intentionally audits every annotation instance but never decodes
or extracts landmarks unless explicitly requested by another command.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.config.paths import CLASS_LIST_FILE, INTERIM_MANIFEST_DIR, METADATA_DIR, SUBSETS, ensure_data_directories
from src.data.wlasl import annotations_by_video, canonical_split, official_subset, video_path

INVENTORY_FIELDS = ["video_id", "gloss", "signer_id", "split", "frame_start", "frame_end", "source", "url", "path", "file_exists", "readable", "fps", "num_frames", "width", "height", "duration", "error"]
INVALID_FIELDS = ["video_id", "path", "reason"]


def _opencv() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for an audit. Install requirements.txt first.") from exc
    return cv2


def inspect_video(path: Path, cv2: Any) -> tuple[bool, dict[str, Any], str]:
    """Open one video and return no exception outside of a per-video failure."""
    if not path.exists():
        return False, {}, "missing_file"
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False, {}, "cannot_open"
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps > 0:
            return False, {}, "invalid_fps"
        if frames <= 0:
            return False, {}, "zero_frames"
        if width <= 0 or height <= 0:
            return False, {}, "invalid_dimensions"
        # Decode one frame: container metadata alone is not enough to call usable.
        ok, _ = capture.read()
        if not ok:
            return False, {}, "decode_error"
        return True, {"fps": fps, "num_frames": frames, "width": width, "height": height, "duration": frames / fps}, ""
    except Exception as exc:  # OpenCV errors must not abort a large audit.
        return False, {}, f"other:{type(exc).__name__}"
    finally:
        capture.release()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_class_metadata(instances: dict[str, Any]) -> dict[str, int]:
    """Preserve WLASL's supplied class IDs; fall back deterministically if absent."""
    mapping: dict[str, int] = {}
    if CLASS_LIST_FILE.exists():
        for line in CLASS_LIST_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label, gloss = line.split(maxsplit=1)
            mapping[gloss] = int(label)
    if not mapping:
        glosses = sorted({item.gloss for item in instances.values()})
        mapping = {gloss: index for index, gloss in enumerate(glosses)}
    _write_csv(METADATA_DIR / "classes.csv", [{"label": label, "gloss": gloss} for gloss, label in mapping.items()], ["label", "gloss"])
    (METADATA_DIR / "gloss_to_id.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return mapping


def build_summary(rows: list[dict[str, Any]], instances: dict[str, Any]) -> dict[str, Any]:
    def counts(values: list[dict[str, Any]]) -> dict[str, int | float]:
        expected = len(values)
        available = sum(row["file_exists"] for row in values)
        usable = sum(row["readable"] for row in values)
        invalid = available - usable
        return {"expected": expected, "available": available, "missing": expected - available, "invalid": invalid, "usable": usable, "coverage": round(100 * usable / expected, 4) if expected else 0.0}

    result: dict[str, Any] = counts(rows)
    result.update({
        "expected_instances": result.pop("expected"), "available_videos": result.pop("available"),
        "missing_videos": result.pop("missing"), "invalid_videos": result.pop("invalid"),
        "usable_videos": result.pop("usable"), "coverage_percent": result.pop("coverage"),
        "by_split": {split: counts([row for row in rows if row["split"] == split]) for split in ("train", "val", "test")},
        "subsets": {},
    })
    rows_by_id = {row["video_id"]: row for row in rows}
    for subset in SUBSETS:
        selected = [rows_by_id[video_id] for video_id in official_subset(subset) if video_id in rows_by_id]
        result["subsets"][subset] = counts(selected)
    return result


def audit_dataset() -> dict[str, Any]:
    ensure_data_directories()
    cv2 = _opencv()
    instances = annotations_by_video()
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for video_id in sorted(instances):
        item = instances[video_id]
        path = video_path(video_id)
        readable, metadata, error = inspect_video(path, cv2)
        row: dict[str, Any] = {
            "video_id": video_id, "gloss": item.gloss, "signer_id": item.signer_id or "", "split": canonical_split(item.split),
            "frame_start": item.frame_start if item.frame_start is not None else "", "frame_end": item.frame_end if item.frame_end is not None else "",
            "source": item.source or "", "url": item.url or "", "path": path.relative_to(path.parents[2]).as_posix(),
            "file_exists": path.exists(), "readable": readable, "fps": metadata.get("fps", ""), "num_frames": metadata.get("num_frames", ""),
            "width": metadata.get("width", ""), "height": metadata.get("height", ""), "duration": metadata.get("duration", ""), "error": error,
        }
        rows.append(row)
        if not readable:
            invalid.append({"video_id": video_id, "path": row["path"], "reason": error})
    _write_csv(INTERIM_MANIFEST_DIR / "video_inventory.csv", rows, INVENTORY_FIELDS)
    _write_csv(INTERIM_MANIFEST_DIR / "valid_videos.csv", [row for row in rows if row["readable"]], INVENTORY_FIELDS)
    _write_csv(INTERIM_MANIFEST_DIR / "invalid_videos.csv", invalid, INVALID_FIELDS)
    build_class_metadata(instances)
    summary = build_summary(rows, instances)
    (METADATA_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = audit_dataset()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
