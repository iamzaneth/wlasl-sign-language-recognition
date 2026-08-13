"""Central, repository-relative paths for the data lifecycle."""
from __future__ import annotations

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_VIDEO_DIR = RAW_DIR / "videos"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_MANIFEST_DIR = INTERIM_DIR / "manifests"
INTERIM_LANDMARK_DIR = INTERIM_DIR / "landmarks" / "mediapipe_v1"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_LANDMARK_DIR = PROCESSED_DIR / "landmarks" / "shoulder_norm_v1"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
MEDIAPIPE_MODEL_DIR = Path(os.environ.get("WLASL_MEDIAPIPE_MODEL_DIR", PROJECT_ROOT / "models" / "mediapipe"))

ANNOTATION_FILE = RAW_DIR / "WLASL_v0.3.json"
CLASS_LIST_FILE = RAW_DIR / "wlasl_class_list.txt"
SUBSETS = ("wlasl100", "wlasl300", "wlasl1000", "wlasl2000")


def split_source(subset: str) -> Path:
    normalized = subset.lower()
    if normalized not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; choose one of {', '.join(SUBSETS)}")
    return RAW_DIR / f"nslt_{normalized.removeprefix('wlasl')}.json"


def split_dir(subset: str) -> Path:
    split_source(subset)  # validates the subset
    return SPLITS_DIR / subset.lower()


def ensure_data_directories() -> None:
    """Create only generated-data directories; raw input is never touched."""
    for path in (
        INTERIM_MANIFEST_DIR,
        INTERIM_LANDMARK_DIR,
        PROCESSED_LANDMARK_DIR,
        METADATA_DIR,
        SPLITS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
