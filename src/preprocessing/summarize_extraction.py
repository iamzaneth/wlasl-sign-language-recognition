"""Print aggregate landmark detection and recovery quality statistics."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter

from src.config.paths import METADATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=METADATA_DIR / "landmark_extraction_report.csv")
    args = parser.parse_args()
    with open(args.report, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if row["extraction_success"].lower() == "true"]
    rates = {name: [float(row[f"{name}_detection_rate"]) for row in successful] for name in ("pose", "left_hand", "right_hand", "face")}
    gap_counts = Counter()
    for row in successful:
        gap_counts.update({"left_missing": int(row["left_hand_missing_frames"]), "right_missing": int(row["right_hand_missing_frames"]), "imputed": int(row["left_hand_imputed_frames"]) + int(row["right_hand_imputed_frames"]), "unresolved": int(row["left_hand_unresolved_frames"]) + int(row["right_hand_unresolved_frames"])})
    print({"total_videos": len(rows), "successful_videos": len(successful), "failed_videos": len(rows) - len(successful), "detection_rate_mean_median": {name: {"mean": statistics.fmean(values) if values else 0.0, "median": statistics.median(values) if values else 0.0} for name, values in rates.items()}, "hand_counts": dict(gap_counts), "note": "Per-video short-gap counts are recorded in the report; full gap-length distribution can be derived without re-extraction."})


if __name__ == "__main__":
    main()
