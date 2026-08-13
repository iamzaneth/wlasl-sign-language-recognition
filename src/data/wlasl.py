"""Small, dependency-light readers for immutable WLASL source files."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.config.paths import ANNOTATION_FILE, RAW_VIDEO_DIR, split_source


@dataclass(frozen=True)
class WLASLInstance:
    video_id: str
    gloss: str
    signer_id: int | None
    split: str
    frame_start: int | None
    frame_end: int | None
    source: str | None
    url: str | None


def load_annotations(path: Path = ANNOTATION_FILE) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected an annotation list in {path}")
    return data


def iter_instances(annotations: Iterable[dict[str, Any]]) -> Iterable[WLASLInstance]:
    for entry in annotations:
        gloss = str(entry["gloss"])
        for item in entry.get("instances", []):
            yield WLASLInstance(
                video_id=str(item["video_id"]), gloss=gloss,
                signer_id=item.get("signer_id"), split=canonical_split(item.get("split")),
                frame_start=item.get("frame_start"), frame_end=item.get("frame_end"),
                source=item.get("source"), url=item.get("url"),
            )


def canonical_split(value: object) -> str:
    value = str(value or "").lower()
    return "val" if value in {"val", "validation", "valid"} else value


def annotations_by_video(path: Path = ANNOTATION_FILE) -> dict[str, WLASLInstance]:
    result: dict[str, WLASLInstance] = {}
    for instance in iter_instances(load_annotations(path)):
        # The official dataset has unique video IDs. Do not silently choose if it changes.
        if instance.video_id in result and result[instance.video_id] != instance:
            raise ValueError(f"Duplicate annotation for video {instance.video_id}")
        result[instance.video_id] = instance
    return result


def official_subset(subset: str) -> dict[str, dict[str, Any]]:
    with split_source(subset).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in official {subset} split")
    return {str(video_id): details for video_id, details in data.items()}


def video_path(video_id: str) -> Path:
    return RAW_VIDEO_DIR / f"{video_id}.mp4"
