"""Create a reproducible, dependency-free EDA report for the local WLASL data.

Usage (from the repository root):
    .venv\\Scripts\\python src/data/analyze_dataset.py

The script reads only ``data/raw`` and writes a self-contained HTML dashboard,
Markdown summary, CSV tables, and a JSON machine-readable summary under
``reports/dataset_analysis``.  No data files or videos are modified.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "dataset_analysis"


@dataclass
class DatasetStats:
    glosses: int
    instances: int
    annotated_video_ids: int
    local_videos: int
    matched_videos: int
    missing_annotated_videos: int
    orphan_local_videos: int
    zero_byte_videos: int
    local_video_bytes: int


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile without a numerical dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fmt_int(value: int | float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}".replace(",", " ")


def fmt_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_label(value: Any) -> str:
    return html.escape(str(value))


def bar_chart_svg(
    title: str,
    data: list[tuple[str, int | float]],
    value_label: str = "mẫu",
    width: int = 760,
    max_items: int = 15,
) -> str:
    """Return a compact, accessible horizontal bar chart SVG."""
    data = data[:max_items]
    if not data:
        return f"<section class='chart'><h3>{safe_label(title)}</h3><p>Không có dữ liệu.</p></section>"
    height = max(180, 52 + 32 * len(data))
    left, right, top = 185, 70, 35
    plot_width = width - left - right
    maximum = max(float(value) for _, value in data) or 1.0
    rows = []
    for index, (label, value) in enumerate(data):
        y = top + index * 32
        bar_width = plot_width * float(value) / maximum
        rows.append(
            f"<text x='{left - 10}' y='{y + 16}' class='axis-label' text-anchor='end'>{safe_label(label)}</text>"
            f"<rect x='{left}' y='{y + 3}' width='{bar_width:.2f}' height='20' rx='4' class='bar'/>"
            f"<text x='{left + bar_width + 8:.2f}' y='{y + 17}' class='value-label'>{fmt_int(value)}</text>"
        )
    return (
        f"<section class='chart'><h3>{safe_label(title)}</h3>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{safe_label(title)}; đơn vị {safe_label(value_label)}'>"
        + "".join(rows)
        + "</svg></section>"
    )


def histogram_svg(title: str, values: list[float], unit: str, bins: int = 14, width: int = 760) -> str:
    if not values:
        return f"<section class='chart'><h3>{safe_label(title)}</h3><p>Không có dữ liệu.</p></section>"
    low, high = min(values), max(values)
    if math.isclose(low, high):
        high = low + 1
    step = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / step), bins - 1)
        counts[index] += 1
    left, right, top, bottom, height = 58, 24, 28, 42, 270
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(counts) or 1
    bar_width = plot_width / bins
    bars = []
    for index, count in enumerate(counts):
        bar_height = plot_height * count / maximum
        x, y = left + index * bar_width, top + plot_height - bar_height
        bars.append(f"<rect x='{x + 1:.2f}' y='{y:.2f}' width='{bar_width - 2:.2f}' height='{bar_height:.2f}' rx='2' class='bar alt'/>")
    ticks = []
    for index in range(5):
        value = low + (high - low) * index / 4
        x = left + plot_width * index / 4
        ticks.append(f"<text x='{x:.2f}' y='{height - 14}' class='tick' text-anchor='middle'>{fmt_float(value, 1)}</text>")
    return (
        f"<section class='chart'><h3>{safe_label(title)}</h3><p class='chart-note'>Trục x: {safe_label(unit)} · n = {fmt_int(len(values))}</p>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{safe_label(title)}'>"
        f"<line x1='{left}' y1='{top + plot_height}' x2='{width-right}' y2='{top + plot_height}' class='axis'/>"
        + "".join(bars + ticks)
        + "</svg></section>"
    )


def table_html(headers: list[str], rows: list[list[Any]], css_class: str = "") -> str:
    head = "".join(f"<th>{safe_label(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{safe_label(item)}</td>" for item in row) + "</tr>" for row in rows
    )
    return f"<div class='table-wrap {css_class}'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def analyze(raw_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    annotations_path = raw_dir / "WLASL_v0.3.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Không tìm thấy annotation: {annotations_path}")
    annotations: list[dict[str, Any]] = load_json(annotations_path)
    videos_dir = raw_dir / "videos"
    local_files = {item.stem: item for item in videos_dir.glob("*.mp4")} if videos_dir.exists() else {}

    instances: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        gloss = str(annotation.get("gloss", ""))
        labels[gloss] = annotation
        for instance in annotation.get("instances", []):
            row = dict(instance)
            row["gloss"] = gloss
            row["is_local"] = str(row.get("video_id", "")) in local_files
            instances.append(row)

    annotated_ids = {str(row.get("video_id", "")) for row in instances}
    local_ids = set(local_files)
    matched_ids = annotated_ids & local_ids
    missing_ids = sorted(annotated_ids - local_ids)
    orphan_ids = sorted(local_ids - annotated_ids)
    file_sizes = [item.stat().st_size for item in local_files.values()]
    stats = DatasetStats(
        glosses=len(labels), instances=len(instances), annotated_video_ids=len(annotated_ids),
        local_videos=len(local_files), matched_videos=len(matched_ids),
        missing_annotated_videos=len(missing_ids), orphan_local_videos=len(orphan_ids),
        zero_byte_videos=sum(size == 0 for size in file_sizes), local_video_bytes=sum(file_sizes),
    )

    split_counts = Counter(str(row.get("split", "unknown")) for row in instances)
    split_local = Counter(str(row.get("split", "unknown")) for row in instances if row["is_local"])
    source_counts = Counter(str(row.get("source", "unknown")) for row in instances)
    source_local = Counter(str(row.get("source", "unknown")) for row in instances if row["is_local"])
    fps_counts = Counter(str(row.get("fps", "unknown")) for row in instances)
    class_counts = Counter(row["gloss"] for row in instances)
    class_train_counts = Counter(row["gloss"] for row in instances if row.get("split") == "train")

    signer_splits: dict[str, set[str]] = defaultdict(set)
    for row in instances:
        signer = row.get("signer_id")
        if signer is not None:
            signer_splits[str(signer)].add(str(row.get("split", "unknown")))
    signer_pattern_counts = Counter(" + ".join(sorted(value)) for value in signer_splits.values())
    signers_per_split = {
        split: len({str(row["signer_id"]) for row in instances if row.get("split") == split and row.get("signer_id") is not None})
        for split in sorted(split_counts)
    }

    bbox_widths, bbox_heights, bbox_areas, bbox_aspect = [], [], [], []
    valid_bbox, invalid_bbox = 0, 0
    for row in instances:
        bbox = row.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1
            if width > 0 and height > 0:
                valid_bbox += 1
                bbox_widths.append(float(width))
                bbox_heights.append(float(height))
                bbox_areas.append(float(width * height))
                bbox_aspect.append(float(width / height))
            else:
                invalid_bbox += 1
        else:
            invalid_bbox += 1

    clipped_frames = []
    full_video_markers = 0
    invalid_frame_ranges = 0
    for row in instances:
        start, end = row.get("frame_start"), row.get("frame_end")
        if end == -1:
            full_video_markers += 1
        elif isinstance(start, int) and isinstance(end, int) and end >= start:
            clipped_frames.append(end - start + 1)
        else:
            invalid_frame_ranges += 1

    split_label_counts: dict[str, int] = {}
    labels_without_split: dict[str, list[str]] = {}
    for split in sorted(split_counts):
        present = {row["gloss"] for row in instances if row.get("split") == split}
        split_label_counts[split] = len(present)
        labels_without_split[split] = sorted(set(labels) - present)

    manifest_stats = []
    class_list = (raw_dir / "wlasl_class_list.txt").read_text(encoding="utf-8").splitlines()
    for size in (100, 300, 1000, 2000):
        manifest_path = raw_dir / f"nslt_{size}.json"
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        class_ids = {entry["action"][0] for entry in manifest.values() if "action" in entry}
        manifest_ids = set(manifest)
        manifest_stats.append({
            "subset": f"NSLT-{size}", "manifest_videos": len(manifest), "classes": len(class_ids),
            "local_videos": len(manifest_ids & local_ids), "coverage_pct": round(100 * len(manifest_ids & local_ids) / len(manifest), 2),
            "class_list_matches": len(class_list) >= size,
        })

    duration_seconds = [frames / row["fps"] for row, frames in ((row, row.get("frame_end", -1) - row.get("frame_start", 0) + 1) for row in instances) if isinstance(row.get("fps"), (int, float)) and row.get("fps", 0) > 0 and row.get("frame_end") != -1 and isinstance(frames, int) and frames > 0]

    analysis = {
        "dataset": asdict(stats),
        "coverage_pct": round(100 * stats.matched_videos / stats.annotated_video_ids, 2) if stats.annotated_video_ids else 0,
        "splits": {key: {"annotated": split_counts[key], "local": split_local[key]} for key in sorted(split_counts)},
        "sources": {key: {"annotated": source_counts[key], "local": source_local[key]} for key in sorted(source_counts)},
        "fps": dict(sorted(fps_counts.items(), key=lambda item: (-item[1], item[0]))),
        "classes": {
            "all_count": len(class_counts), "train_count": len(class_train_counts), "by_split": split_label_counts,
            "single_instance": sum(count == 1 for count in class_counts.values()),
            "at_least_10": sum(count >= 10 for count in class_counts.values()),
            "at_least_20": sum(count >= 20 for count in class_counts.values()),
            "at_least_50": sum(count >= 50 for count in class_counts.values()),
            "median_instances": percentile([float(value) for value in class_counts.values()], 50),
            "p90_instances": percentile([float(value) for value in class_counts.values()], 90),
            "max_instances": max(class_counts.values()),
        },
        "signers": {
            "unique": len(signer_splits), "per_split": signers_per_split,
            "split_membership": dict(sorted(signer_pattern_counts.items())),
            "multi_split": sum(len(value) > 1 for value in signer_splits.values()),
        },
        "bbox": {
            "valid": valid_bbox, "invalid_or_missing": invalid_bbox,
            "width": distribution_summary(bbox_widths), "height": distribution_summary(bbox_heights),
            "area": distribution_summary(bbox_areas), "aspect_ratio": distribution_summary(bbox_aspect),
        },
        "temporal": {
            "full_video_marker": full_video_markers, "trimmed": len(clipped_frames), "invalid_range": invalid_frame_ranges,
            "trimmed_frames": distribution_summary([float(item) for item in clipped_frames]),
            "trimmed_seconds": distribution_summary(duration_seconds),
        },
        "files": {"size": distribution_summary([float(item) for item in file_sizes])},
        "nslt_manifests": manifest_stats,
        "quality": {
            "missing_annotated_video_ids": missing_ids,
            "orphan_local_video_ids": orphan_ids,
            "labels_missing_validation": labels_without_split.get("val", []),
            "labels_missing_test": labels_without_split.get("test", []),
        },
        "top_classes": [{"gloss": gloss, "instances": count, "train_instances": class_train_counts[gloss]} for gloss, count in class_counts.most_common(30)],
        "source_counts": source_counts, "class_counts": class_counts,
        "bbox_aspect_values": bbox_aspect, "trimmed_frame_values": clipped_frames,
        "file_size_mb_values": [size / (1024 * 1024) for size in file_sizes],
    }
    return analysis, instances, manifest_stats


def distribution_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": round(min(values), 3), "p25": round(percentile(values, 25), 3),
        "median": round(percentile(values, 50), 3), "mean": round(statistics.fmean(values), 3),
        "p75": round(percentile(values, 75), 3), "p95": round(percentile(values, 95), 3), "max": round(max(values), 3),
    }


def build_html(analysis: dict[str, Any]) -> str:
    dataset, classes, signers = analysis["dataset"], analysis["classes"], analysis["signers"]
    split_rows = [[split, fmt_int(values["annotated"]), fmt_int(values["local"]), f"{100 * values['local'] / values['annotated']:.1f}%"] for split, values in analysis["splits"].items()]
    source_rows = [[source, fmt_int(values["annotated"]), fmt_int(values["local"])] for source, values in sorted(analysis["sources"].items(), key=lambda item: -item[1]["annotated"])]
    top_rows = [[index + 1, item["gloss"], fmt_int(item["instances"]), fmt_int(item["train_instances"])] for index, item in enumerate(analysis["top_classes"])]
    manifest_rows = [[row["subset"], fmt_int(row["classes"]), fmt_int(row["manifest_videos"]), fmt_int(row["local_videos"]), f"{row['coverage_pct']:.1f}%"] for row in analysis["nslt_manifests"]]
    overlap_rows = [[pattern, fmt_int(count)] for pattern, count in analysis["signers"]["split_membership"].items()]
    temporal = analysis["temporal"]
    bbox = analysis["bbox"]
    cards = [
        ("Gloss / lớp", fmt_int(dataset["glosses"]), "Tất cả nhãn trong WLASL v0.3"),
        ("Instances", fmt_int(dataset["instances"]), "Mẫu video được annotation"),
        ("Video cục bộ", fmt_int(dataset["local_videos"]), f"{analysis['coverage_pct']:.1f}% annotation có tệp"),
        ("Dung lượng video", fmt_bytes(dataset["local_video_bytes"]), "11.980 tệp MP4"),
        ("Signer", fmt_int(signers["unique"]), f"{signers['multi_split']} xuất hiện ở ≥2 split"),
        ("BBox hợp lệ", f"{100*bbox['valid']/dataset['instances']:.1f}%", "x2>x1, y2>y1"),
    ]
    cards_html = "".join(f"<article class='card'><p>{label}</p><strong>{value}</strong><small>{note}</small></article>" for label, value, note in cards)
    top_chart = bar_chart_svg("30 gloss có nhiều instances nhất", [(item["gloss"], item["instances"]) for item in analysis["top_classes"]], max_items=15)
    split_chart = bar_chart_svg("Phân bố annotation theo split", [(name, values["annotated"]) for name, values in analysis["splits"].items()])
    source_chart = bar_chart_svg("Nguồn video", [(name, values["annotated"]) for name, values in sorted(analysis["sources"].items(), key=lambda item: -item[1]["annotated"])], max_items=12)
    file_chart = histogram_svg("Phân bố kích thước tệp video", analysis["file_size_mb_values"], "MB")
    frame_chart = histogram_svg("Độ dài đoạn đã cắt", [float(value) for value in analysis["trimmed_frame_values"]], "frames")
    aspect_chart = histogram_svg("Tỉ lệ khung BBox (rộng / cao)", analysis["bbox_aspect_values"], "tỉ lệ")
    quality_notes = [
        f"{fmt_int(dataset['missing_annotated_videos'])} video có trong annotation nhưng chưa thấy tệp cục bộ.",
        f"{fmt_int(dataset['orphan_local_videos'])} video cục bộ không có trong annotation v0.3.",
        f"{fmt_int(dataset['zero_byte_videos'])} tệp video 0 byte.",
        f"{fmt_int(temporal['full_video_marker'])} instances dùng frame_end = -1 (dùng toàn video); {fmt_int(temporal['trimmed'])} instances có đoạn cắt xác định.",
        f"{fmt_int(temporal['invalid_range'])} range frame không hợp lệ; {fmt_int(bbox['invalid_or_missing'])} bbox thiếu/không hợp lệ.",
        f"{fmt_int(len(analysis['quality']['labels_missing_validation']))} lớp không có mẫu validation; {fmt_int(len(analysis['quality']['labels_missing_test']))} lớp không có mẫu test.",
    ]
    class_notes = [
        ["Median instances / gloss", fmt_float(classes["median_instances"])],
        ["P90 instances / gloss", fmt_float(classes["p90_instances"])],
        ["Gloss có đúng 1 mẫu", fmt_int(classes["single_instance"])],
        ["Gloss có ≥10 / ≥20 / ≥50 mẫu", f"{fmt_int(classes['at_least_10'])} / {fmt_int(classes['at_least_20'])} / {fmt_int(classes['at_least_50'])}"],
        ["Lớp có mặt train / val / test", " / ".join(f"{key}: {fmt_int(value)}" for key, value in classes["by_split"].items())],
    ]
    dist_rows = []
    for name, details, unit in [("BBox width", bbox["width"], "px"), ("BBox height", bbox["height"], "px"), ("BBox area", bbox["area"], "px²"), ("Trimmed frames", temporal["trimmed_frames"], "frames"), ("Trimmed duration", temporal["trimmed_seconds"], "s"), ("File size", analysis["files"]["size"], "bytes")]:
        if details.get("count", 0):
            dist_rows.append([name, fmt_int(details["count"]), f"{fmt_float(float(details['median']))} {unit}", f"{fmt_float(float(details['p95']))} {unit}", f"{fmt_float(float(details['max']))} {unit}"])
    return f"""<!doctype html>
<html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>WLASL — Dataset Analysis</title>
<style>
:root{{--ink:#172033;--muted:#5d6a7c;--line:#dce3eb;--bg:#f6f8fb;--card:#fff;--blue:#2764d8;--teal:#009e8b;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}} main{{max-width:1440px;margin:auto;padding:32px 24px 56px}} h1{{font-size:30px;margin:0 0 4px}} h2{{margin:38px 0 14px;font-size:21px}} h3{{font-size:16px;margin:0 0 4px}} .subtitle,.chart-note{{color:var(--muted);margin:0}} .meta{{color:var(--muted);font-size:13px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-top:22px}} .card,.chart,.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 1px 2px #17203308}} .card p{{margin:0;color:var(--muted);font-size:13px}} .card strong{{display:block;font-size:25px;margin:5px 0 3px}} .card small{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}} .chart svg{{display:block;width:100%;height:auto;margin-top:8px}} .axis{{stroke:#b9c4d0;stroke-width:1}} .bar{{fill:var(--blue)}} .bar.alt{{fill:var(--teal)}} .axis-label{{font-size:12px;fill:#3f4c5d}} .value-label{{font-size:12px;fill:#3f4c5d;font-weight:600}} .tick{{font-size:11px;fill:#5d6a7c}} .table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}} table{{border-collapse:collapse;width:100%;min-width:480px}} th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}} th{{background:#f0f4f9;color:#344255;font-size:12px;text-transform:uppercase;letter-spacing:.03em}} tr:last-child td{{border-bottom:0}} .two-col{{columns:2;gap:24px;padding-left:24px}} li{{break-inside:avoid;margin:5px 0}} .callout{{border-left:4px solid var(--blue);padding:12px 15px;background:#edf4ff;border-radius:0 8px 8px 0}} @media(max-width:600px){{main{{padding:22px 12px}}.grid{{grid-template-columns:1fr}}.two-col{{columns:1}}}}
</style></head><body><main>
<header><h1>WLASL v0.3 — Phân tích dataset cục bộ</h1><p class='subtitle'>Nguồn: <code>data/raw/WLASL_v0.3.json</code> và <code>data/raw/videos/</code></p><p class='meta'>Dashboard được tạo từ metadata và trạng thái file hiện có. Thời lượng thực / độ phân giải video không được đo vì bộ công cụ video không có trong môi trường.</p></header>
<section class='cards'>{cards_html}</section>
<h2>1. Phạm vi dữ liệu và split</h2><div class='grid'>{split_chart}<section class='panel'><h3>Độ phủ video cục bộ theo split</h3>{table_html(['Split','Annotation','Có tệp','Độ phủ'], split_rows)}</section></div>
<h2>2. Nhãn và mất cân bằng lớp</h2><p class='callout'>WLASL là long-tail: median chỉ {fmt_float(classes['median_instances'])} mẫu/gloss, trong khi gloss lớn nhất có {fmt_int(classes['max_instances'])} mẫu. Khi train, nên báo cáo macro-F1/top-k ngoài accuracy và dùng sampler hoặc class-weight.</p><div class='grid'>{top_chart}<section class='panel'><h3>Chỉ số phân phối lớp</h3>{table_html(['Chỉ số','Giá trị'], class_notes)}</section></div><h3>Top 30 lớp</h3>{table_html(['#','Gloss','Tổng instances','Train instances'], top_rows)}
<h2>3. Nguồn dữ liệu và signer</h2><div class='grid'>{source_chart}<section class='panel'><h3>Signer giữa các split</h3><p class='chart-note'>Các signer xuất hiện trong nhiều split là rủi ro nhận diện danh tính thay vì động tác nếu protocol không yêu cầu signer-independent.</p>{table_html(['Tổ hợp split','Số signer'], overlap_rows)}</section></div><h3>Nguồn video chi tiết</h3>{table_html(['Nguồn','Annotation','Có tệp cục bộ'], source_rows)}
<h2>4. Hình học, thời gian và tệp video</h2><div class='grid'>{aspect_chart}{frame_chart}{file_chart}</div>{table_html(['Đại lượng','n','Median','P95','Max'], dist_rows)}
<h2>5. Kiểm tra chất lượng & khuyến nghị</h2><section class='panel'><ul class='two-col'>{''.join(f'<li>{safe_label(note)}</li>' for note in quality_notes)}</ul></section>
<p class='callout'><strong>Khuyến nghị pipeline:</strong> chỉ dùng giao của annotation và video cục bộ; lưu danh sách ID thiếu; giữ split gốc; áp dụng crop bbox (có padding) khi bbox hợp lệ; xử lý riêng frame_end = -1; với lớp hiếm dùng weighted sampler / augmentation theo thời gian và đánh giá macro metrics.</p>
<h2>6. Các subset NSLT</h2>{table_html(['Subset','Lớp','Video theo manifest','Có tệp','Độ phủ'], manifest_rows)}
<footer class='meta'><p>Các tệp đi kèm: <code>analysis_summary.json</code>, <code>class_distribution.csv</code>, <code>data_quality.csv</code>, <code>nslt_coverage.csv</code>.</p></footer>
</main></body></html>"""


def write_outputs(analysis: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_json = dict(analysis)
    for key in ("source_counts", "class_counts", "bbox_aspect_values", "trimmed_frame_values", "file_size_mb_values"):
        compact_json.pop(key, None)
    (output_dir / "analysis_summary.json").write_text(json.dumps(compact_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dashboard.html").write_text(build_html(analysis), encoding="utf-8")
    class_rows = [{"gloss": gloss, "instances": count} for gloss, count in analysis["class_counts"].most_common()]
    write_csv(output_dir / "class_distribution.csv", class_rows, ["gloss", "instances"])
    quality_rows = [
        {"check": "annotation_video_missing_locally", "count": analysis["dataset"]["missing_annotated_videos"]},
        {"check": "local_video_missing_annotation", "count": analysis["dataset"]["orphan_local_videos"]},
        {"check": "zero_byte_local_videos", "count": analysis["dataset"]["zero_byte_videos"]},
        {"check": "invalid_or_missing_bbox", "count": analysis["bbox"]["invalid_or_missing"]},
        {"check": "invalid_frame_range", "count": analysis["temporal"]["invalid_range"]},
        {"check": "labels_without_validation", "count": len(analysis["quality"]["labels_missing_validation"])},
        {"check": "labels_without_test", "count": len(analysis["quality"]["labels_missing_test"])},
    ]
    write_csv(output_dir / "data_quality.csv", quality_rows, ["check", "count"])
    write_csv(output_dir / "nslt_coverage.csv", analysis["nslt_manifests"], ["subset", "classes", "manifest_videos", "local_videos", "coverage_pct", "class_list_matches"])
    ds = analysis["dataset"]
    markdown = f"""# WLASL v0.3 — Dataset analysis

Dashboard trực quan: [dashboard.html](dashboard.html)

## Tóm tắt

- {fmt_int(ds['glosses'])} gloss, {fmt_int(ds['instances'])} instances annotation và {fmt_int(ds['local_videos'])} video MP4 cục bộ ({analysis['coverage_pct']:.1f}% ID annotation).
- {fmt_int(ds['missing_annotated_videos'])} ID annotation thiếu video cục bộ; {fmt_int(ds['orphan_local_videos'])} video cục bộ không có annotation.
- Dataset mất cân bằng mạnh: median {fmt_float(analysis['classes']['median_instances'])} instances/gloss, P90 {fmt_float(analysis['classes']['p90_instances'])}, cao nhất {fmt_int(analysis['classes']['max_instances'])}.
- {fmt_int(analysis['signers']['multi_split'])}/{fmt_int(analysis['signers']['unique'])} signer có mặt trong hơn một split.

Các số liệu chi tiết, bảng phân phối lớp và kiểm tra chất lượng nằm trong dashboard và CSV/JSON đi kèm.
"""
    (output_dir / "README.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WLASL metadata and locally available videos.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    analysis, _, _ = analyze(args.raw_dir)
    write_outputs(analysis, args.out_dir)
    # Keep CLI output ASCII-only: some Windows terminals still use cp1252.
    print(f"Report created at: {args.out_dir}")
    print(f"Instances: {analysis['dataset']['instances']:,}; local videos: {analysis['dataset']['local_videos']:,}; coverage: {analysis['coverage_pct']:.2f}%")


if __name__ == "__main__":
    main()
