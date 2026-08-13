"""Fixed-length feature cache and modality selection for encoder-only models."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config.paths import PROCESSED_LANDMARK_DIR, SPLITS_DIR
from src.preprocessing.landmark_io import load_npz


# The order is stable and saved in the cache manifest.  Do not reorder it after
# models have been trained: checkpoints use column offsets from this layout.
MODALITY_KEYS: dict[str, tuple[str, ...]] = {
    "pose": ("pose_xyz",),
    "hand": ("left_hand_xyz", "right_hand_xyz"),
    "mouth": ("mouth_xyz",),
    "eye": ("left_eye_xyz", "right_eye_xyz"),
    "eyebrow": ("left_eyebrow_xyz", "right_eyebrow_xyz"),
}
MODALITY_ALIASES = {"hands": "hand", "eyes": "eye", "eyebrows": "eyebrow", "month": "mouth"}
VALID_MODALITIES = tuple(MODALITY_KEYS)

# Public experiment contract.  These are the exact landmark components in each
# user-facing subset key.  The cache's internal groups combine paired sides.
LANDMARK_SUBSET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "pose": ("pose",),
    "hands": ("left_hand", "right_hand"),
    "pose_hands": ("pose", "left_hand", "right_hand"),
    "pose_hands_mouth": ("pose", "left_hand", "right_hand", "mouth"),
    "pose_hands_eyes": ("pose", "left_hand", "right_hand", "left_eye", "right_eye"),
    "pose_hands_eyebrows": ("pose", "left_hand", "right_hand", "left_eyebrow", "right_eyebrow"),
    "pose_hands_face": ("pose", "left_hand", "right_hand", "mouth", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow"),
}
COMPONENT_TO_MODALITY = {"pose": "pose", "left_hand": "hand", "right_hand": "hand", "mouth": "mouth", "left_eye": "eye", "right_eye": "eye", "left_eyebrow": "eyebrow", "right_eyebrow": "eyebrow"}
LANDMARK_SUBSETS = {name: canonical for name, canonical in ((name, tuple(dict.fromkeys(COMPONENT_TO_MODALITY[component] for component in components))) for name, components in LANDMARK_SUBSET_COMPONENTS.items())}
VALID_LANDMARK_SUBSETS = tuple(LANDMARK_SUBSETS)


def canonical_modalities(values: Iterable[str]) -> tuple[str, ...]:
    """Validate and put a modality collection in the canonical feature order."""
    requested = {MODALITY_ALIASES.get(str(value).strip().lower(), str(value).strip().lower()) for value in values}
    unknown = requested.difference(MODALITY_KEYS)
    if unknown:
        raise ValueError(f"Unknown modalities: {', '.join(sorted(unknown))}. Choose from {', '.join(VALID_MODALITIES)}")
    if not requested:
        raise ValueError("At least one modality is required")
    return tuple(name for name in VALID_MODALITIES if name in requested)


def parse_feature_set(value: str) -> tuple[str, ...]:
    """Parse a CLI feature set such as ``pose,hand,mouth``."""
    return canonical_modalities(part for part in value.replace("+", ",").split(",") if part.strip())


def landmark_subset_modalities(name: str) -> tuple[str, ...]:
    """Resolve one named experiment subset to its canonical modality blocks."""
    normalized = name.strip().lower()
    try:
        return LANDMARK_SUBSETS[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown landmark subset {name!r}. Choose from {', '.join(VALID_LANDMARK_SUBSETS)}") from error


def landmark_subset_components(name: str) -> tuple[str, ...]:
    """Return the exact left/right landmark components in a named subset."""
    normalized = name.strip().lower()
    try:
        return LANDMARK_SUBSET_COMPONENTS[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown landmark subset {name!r}. Choose from {', '.join(VALID_LANDMARK_SUBSETS)}") from error


def feature_layout() -> dict[str, tuple[int, int]]:
    """Return inclusive/exclusive coordinate-column spans for every modality."""
    cursor = 0
    layout: dict[str, tuple[int, int]] = {}
    for modality, keys in MODALITY_KEYS.items():
        width = sum(_point_count(key) * 3 for key in keys)
        layout[modality] = (cursor, cursor + width)
        cursor += width
    return layout


def feature_columns(modalities: Iterable[str]) -> np.ndarray:
    layout = feature_layout()
    columns: list[int] = []
    for modality in canonical_modalities(modalities):
        start, end = layout[modality]
        columns.extend(range(start, end))
    return np.asarray(columns, dtype=np.int64)


def _point_count(key: str) -> int:
    return {
        "pose_xyz": 9,
        "left_hand_xyz": 21,
        "right_hand_xyz": 21,
        "mouth_xyz": 40,
        "left_eye_xyz": 16,
        "right_eye_xyz": 16,
        "left_eyebrow_xyz": 10,
        "right_eyebrow_xyz": 10,
    }[key]


def _read_split_rows(subset: str, splits_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        path = splits_dir / subset / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    ids = [row["video_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate video IDs in splits for {subset}")
    return rows


def _resize_sequence(x: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample long clips; right-pad short clips with zero frames."""
    frames, width = x.shape
    result = np.zeros((seq_len, width), dtype=np.float32)
    mask = np.zeros(seq_len, dtype=bool)
    if frames == 0:
        return result, mask
    if frames <= seq_len:
        result[:frames] = x
        mask[:frames] = True
    else:
        indices = np.rint(np.linspace(0, frames - 1, seq_len)).astype(np.int64)
        result[:] = x[indices]
        mask[:] = True
    return result, mask


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "features": cache_dir / "features.npy",
        "frame_mask": cache_dir / "frame_mask.npy",
        "labels": cache_dir / "labels.npy",
        "manifest": cache_dir / "manifest.json",
        "samples": cache_dir / "samples.csv",
    }


def build_feature_cache(
    subset: str,
    *,
    seq_len: int,
    cache_dir: Path,
    landmark_dir: Path = PROCESSED_LANDMARK_DIR,
    splits_dir: Path = SPLITS_DIR,
    overwrite: bool = False,
) -> Path:
    """Create an mmap-friendly full-modality cache from processed NPZ files."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    rows = _read_split_rows(subset, splits_dir)
    paths = cache_paths(cache_dir)
    if paths["manifest"].exists() and not overwrite:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("subset") == subset and manifest.get("seq_len") == seq_len:
            return cache_dir
        raise FileExistsError(f"Cache exists with a different configuration: {cache_dir}. Pass --overwrite to rebuild it.")
    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [row["video_id"] for row in rows if not (landmark_dir / f"{row['video_id']}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} processed landmark files are missing; first IDs: {', '.join(missing[:8])}")

    total_width = max(end for _, end in feature_layout().values())
    features = np.lib.format.open_memmap(paths["features"], mode="w+", dtype=np.float32, shape=(len(rows), seq_len, total_width))
    masks = np.lib.format.open_memmap(paths["frame_mask"], mode="w+", dtype=bool, shape=(len(rows), seq_len))
    labels = np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=np.int64, shape=(len(rows),))
    layout = feature_layout()

    try:
        for index, row in enumerate(rows):
            payload = load_npz(landmark_dir / f"{row['video_id']}.npz", processed=True)
            components = [np.asarray(payload[key], dtype=np.float32).reshape(int(payload["num_frames"].item()), -1) for keys in MODALITY_KEYS.values() for key in keys]
            raw = np.concatenate(components, axis=1)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            sequence, frame_mask = _resize_sequence(raw, seq_len)
            features[index] = sequence
            masks[index] = frame_mask
            labels[index] = int(row["label"])
    finally:
        del features, masks, labels

    with paths["samples"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_idx", "video_id", "gloss", "label", "split"))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"sample_idx": index, **{key: row[key] for key in ("video_id", "gloss", "label", "split")}})
    paths["manifest"].write_text(json.dumps({
        "subset": subset,
        "seq_len": seq_len,
        "num_samples": len(rows),
        "feature_dim": total_width,
        "modalities": {name: list(feature_layout()[name]) for name in VALID_MODALITIES},
        "source": str(landmark_dir),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_dir


def load_cache(cache_dir: Path) -> dict[str, object]:
    paths = cache_paths(cache_dir)
    required = ("features", "frame_mask", "labels", "manifest", "samples")
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError("Missing feature-cache files: " + ", ".join(missing))
    with paths["samples"].open("r", newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    return {
        "features": np.load(paths["features"], mmap_mode="r"),
        "frame_mask": np.load(paths["frame_mask"], mmap_mode="r"),
        "labels": np.load(paths["labels"], mmap_mode="r"),
        "samples": samples,
        "manifest": json.loads(paths["manifest"].read_text(encoding="utf-8")),
        "paths": paths,
    }
