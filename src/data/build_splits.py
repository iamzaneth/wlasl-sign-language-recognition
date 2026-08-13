"""Build usable subsets without changing an official WLASL split assignment."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.config.paths import INTERIM_MANIFEST_DIR, SUBSETS, split_dir
from src.data.wlasl import annotations_by_video, canonical_split, official_subset


def usable_video_ids(manifest: Path = INTERIM_MANIFEST_DIR / "valid_videos.csv") -> set[str]:
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is absent; run `python -m src.data.audit_dataset` first.")
    with manifest.open(newline="", encoding="utf-8") as handle:
        return {row["video_id"] for row in csv.DictReader(handle) if row.get("readable", "").lower() == "true"}


def build_splits(subset: str) -> dict[str, int]:
    instances, official, usable = annotations_by_video(), official_subset(subset), usable_video_ids()
    rows = {"train": [], "val": [], "test": []}
    for video_id, details in sorted(official.items()):
        if video_id not in usable:
            continue
        item = instances.get(video_id)
        if item is None:
            continue  # official source and annotations disagree; never fabricate a sample.
        split = canonical_split(details.get("subset", item.split))
        if split not in rows:
            continue
        action = details.get("action", [])
        if not action:
            continue
        rows[split].append({"video_id": video_id, "gloss": item.gloss, "label": int(action[0]), "split": split})
    target = split_dir(subset)
    target.mkdir(parents=True, exist_ok=True)
    fields = ["video_id", "gloss", "label", "split"]
    for split, values in rows.items():
        with (target / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(values)
    return {name: len(values) for name, values in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=SUBSETS, default=None, help="Build one subset (default: all).")
    args = parser.parse_args()
    for subset in (args.subset,) if args.subset else SUBSETS:
        print(f"{subset}: {build_splits(subset)}")


if __name__ == "__main__":
    main()
