"""Atomic NPZ input/output with schema validation for resumable stages."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.landmark_schema import validate_landmark_payload


def load_npz(path: Path, *, processed: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    validate_landmark_payload(payload, processed=processed)
    return payload


def is_valid_npz(path: Path, *, processed: bool = False) -> bool:
    try:
        load_npz(path, processed=processed)
    except (OSError, ValueError, KeyError):
        return False
    return True


def save_npz(path: Path, payload: dict[str, Any], *, processed: bool = False) -> None:
    arrays = {key: np.asarray(value) for key, value in payload.items()}
    validate_landmark_payload(arrays, processed=processed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
