"""Two-sided wrist-anchored XY and direct-Z interpolation for short hand gaps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.preprocessing.landmark_schema import LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX


@dataclass(frozen=True)
class Gap:
    start: int
    end: int  # exclusive
    @property
    def length(self) -> int:
        return self.end - self.start


def internal_missing_gaps(observed: np.ndarray) -> list[Gap]:
    """Return missing runs with observed endpoints; boundary gaps are excluded."""
    observed = np.asarray(observed, dtype=bool)
    gaps: list[Gap] = []
    index = 0
    while index < len(observed):
        if observed[index]:
            index += 1; continue
        start = index
        while index < len(observed) and not observed[index]:
            index += 1
        if start > 0 and index < len(observed) and observed[start - 1] and observed[index]:
            gaps.append(Gap(start, index))
    return gaps


def recover_hand(hand_xyz: np.ndarray, observed: np.ndarray, wrist_xy: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover only short internal gaps with finite endpoint and target wrists."""
    hand = np.asarray(hand_xyz, dtype=np.float32).copy()
    observed = np.asarray(observed, dtype=bool)
    imputed = np.zeros(len(observed), dtype=bool)
    for gap in internal_missing_gaps(observed):
        if gap.length > max_gap_frames:
            continue
        left, right = gap.start - 1, gap.end
        endpoints_ok = np.isfinite(hand[[left, right]]).all() and np.isfinite(wrist_xy[[left, right]]).all()
        targets_ok = np.isfinite(wrist_xy[gap.start:gap.end]).all()
        if not (endpoints_ok and targets_ok):
            continue
        relative_left = hand[left, :, :2] - wrist_xy[left]
        relative_right = hand[right, :, :2] - wrist_xy[right]
        for frame in range(gap.start, gap.end):
            alpha = (frame - left) / (right - left)
            relative_xy = (1 - alpha) * relative_left + alpha * relative_right
            hand[frame, :, :2] = wrist_xy[frame] + relative_xy
            hand[frame, :, 2] = (1 - alpha) * hand[left, :, 2] + alpha * hand[right, :, 2]
            imputed[frame] = True
    unresolved = ~observed & ~imputed
    return hand, imputed, unresolved


def recover_hands(payload: dict[str, object], max_gap_frames: int) -> dict[str, object]:
    result = dict(payload)
    pose = np.asarray(payload["pose_xyz"])
    for side, wrist_index in (("left", LEFT_WRIST_INDEX), ("right", RIGHT_WRIST_INDEX)):
        hand, imputed, unresolved = recover_hand(np.asarray(payload[f"{side}_hand_xyz"]), np.asarray(payload[f"{side}_hand_observed"]), pose[:, wrist_index, :2], max_gap_frames)
        result[f"{side}_hand_xyz"] = hand
        result[f"{side}_hand_imputed"] = imputed
        result[f"{side}_hand_unresolved"] = unresolved
    return result
