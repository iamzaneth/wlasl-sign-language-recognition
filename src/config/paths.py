"""Central, repository-relative paths for the data lifecycle."""
from __future__ import annotations

from pathlib import Path
import os
from typing import Any

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


def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative metadata paths and remap paths from an old clone."""
    path = Path(value)
    if not path.is_absolute():
        return PROJECT_ROOT / path
    try:
        path.relative_to(PROJECT_ROOT)
        return path
    except ValueError:
        pass
    matches = [
        index for index, part in enumerate(path.parts)
        if part.casefold() == PROJECT_ROOT.name.casefold()
    ]
    if matches:
        return PROJECT_ROOT.joinpath(*path.parts[matches[-1] + 1:])
    return path


def project_relative_path(value: str | Path) -> str:
    """Serialize a repository-owned path without a machine-specific prefix."""
    path = resolve_project_path(value)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def normalize_project_paths(value: Any) -> Any:
    """Recursively make absolute repository paths portable in metadata."""
    if isinstance(value, dict):
        return {key: normalize_project_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_project_paths(item) for item in value]
    if isinstance(value, Path):
        return project_relative_path(value)
    if isinstance(value, str) and Path(value).is_absolute():
        resolved = resolve_project_path(value)
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            return value
        return project_relative_path(resolved)
    return value


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
