"""Shoulder-midpoint, median-video-scale normalization that preserves Z."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.preprocessing.landmark_schema import LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, XYZ_KEYS


def shoulder_normalize(payload: dict[str, Any], epsilon: float = 1e-6) -> tuple[dict[str, Any], np.ndarray, float]:
    pose = np.asarray(payload["pose_xyz"], dtype=np.float32)
    left, right = pose[:, LEFT_SHOULDER_INDEX, :2], pose[:, RIGHT_SHOULDER_INDEX, :2]
    distances = np.linalg.norm(left - right, axis=1)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1) & (distances > epsilon)
    scale = float(np.median(distances[valid])) if valid.any() else float("nan")
    result = dict(payload)
    if not valid.any() or not np.isfinite(scale):
        for key in XYZ_KEYS:
            result[key] = np.asarray(payload[key], dtype=np.float32).copy()
        return result, valid, scale
    centers = (left + right) / 2.0
    for key in XYZ_KEYS:
        xyz = np.asarray(payload[key], dtype=np.float32).copy()
        # Only transform valid-anchor frames; Z is deliberately untouched.
        xyz[valid, :, :2] = (xyz[valid, :, :2] - centers[valid, None, :]) / (scale + epsilon)
        result[key] = xyz
    return result, valid, scale
