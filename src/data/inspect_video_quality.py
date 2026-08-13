"""Audit technical quality of locally available WLASL MP4 videos.

This module deliberately has no third-party dependency. It reads ISO-BMFF/MP4
container metadata (not video pixels) and creates an HTML dashboard and CSV
inventory. It therefore detects malformed/missing MP4 metadata, duration,
dimensions, timing, codec and bitrate anomalies without modifying videos.

Run from repository root:
    .venv\\Scripts\\python src/data/inspect_video_quality.py
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "video_quality"

# Only these boxes contain child boxes that are relevant to a media track.
CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta", b"meta"}


@dataclass(frozen=True)
class Box:
    kind: bytes
    offset: int
    payload_offset: int
    payload_size: int


@dataclass
class VideoRecord:
    video_id: str
    filename: str
    size_bytes: int
    status: str
    issue: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frames: int | None = None
    codec: str | None = None
    video_tracks: int = 0
    audio_tracks: int = 0
    moov_before_mdat: bool | None = None
    mdat_size_mismatch: bool = False
    bitrate_mbps: float | None = None


def fmt_int(value: int | float | None) -> str:
    return "—" if value is None else f"{value:,.0f}".replace(",", " ")


def fmt_float(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}".replace(",", " ")


def fmt_bytes(value: int | float) -> str:
    value = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": round(min(values), 4), "p25": round(percentile(values, 25), 4),
        "median": round(percentile(values, 50), 4), "mean": round(statistics.fmean(values), 4),
        "p75": round(percentile(values, 75), 4), "p95": round(percentile(values, 95), 4), "max": round(max(values), 4),
    }


def read_at(handle: BinaryIO, offset: int, size: int) -> bytes:
    handle.seek(offset)
    return handle.read(size)


def child_boxes(handle: BinaryIO, start: int, size: int, allow_truncated_mdat: bool = False) -> Iterator[Box]:
    """Yield ISO-BMFF boxes in one bounded byte range, safely handling size forms."""
    end = start + size
    position = start
    while position + 8 <= end:
        header = read_at(handle, position, 16)
        if len(header) < 8:
            return
        declared_size = int.from_bytes(header[:4], "big")
        kind = header[4:8]
        header_size = 8
        if declared_size == 1:
            if len(header) < 16:
                raise ValueError("truncated extended-size box")
            declared_size = int.from_bytes(header[8:16], "big")
            header_size = 16
        elif declared_size == 0:
            declared_size = end - position
        if declared_size < header_size:
            raise ValueError(f"invalid {kind.decode('latin-1', 'replace')} box size")
        if position + declared_size > end:
            # A small number of local WLASL files declare an mdat atom extending
            # beyond EOF. Their moov metadata is intact, so retain it for the
            # inventory but mark the file as a container warning.
            if allow_truncated_mdat and kind == b"mdat":
                yield Box(kind, position, position + header_size, end - position - header_size)
                return
            raise ValueError(f"invalid {kind.decode('latin-1', 'replace')} box size")
        yield Box(kind, position, position + header_size, declared_size - header_size)
        position += declared_size


def descendants(handle: BinaryIO, root: Box, kind: bytes) -> Iterator[Box]:
    """Find target boxes by descending only into known container boxes."""
    for child in child_boxes(handle, root.payload_offset, root.payload_size):
        if child.kind == kind:
            yield child
        if child.kind in CONTAINER_BOXES:
            yield from descendants(handle, child, kind)


def first_descendant(handle: BinaryIO, root: Box, kind: bytes) -> Box | None:
    return next(descendants(handle, root, kind), None)


def payload(handle: BinaryIO, box: Box, maximum: int = 4096) -> bytes:
    return read_at(handle, box.payload_offset, min(box.payload_size, maximum))


def parse_mdhd(data: bytes) -> tuple[int, int]:
    if len(data) < 20:
        raise ValueError("short mdhd")
    version = data[0]
    if version == 1:
        if len(data) < 32:
            raise ValueError("short version-1 mdhd")
        return int.from_bytes(data[20:24], "big"), int.from_bytes(data[24:32], "big")
    return int.from_bytes(data[12:16], "big"), int.from_bytes(data[16:20], "big")


def parse_track_dimensions(data: bytes) -> tuple[int, int]:
    # The width and height 16.16 fixed-point fields are the final 8 bytes of tkhd.
    if len(data) < 8:
        raise ValueError("short tkhd")
    return int.from_bytes(data[-8:-4], "big") >> 16, int.from_bytes(data[-4:], "big") >> 16


def parse_handler(data: bytes) -> bytes:
    if len(data) < 12:
        raise ValueError("short hdlr")
    return data[8:12]


def parse_stsd_codec(data: bytes) -> str | None:
    # FullBox header (4), entry count (4), first SampleEntry: size (4), type (4).
    if len(data) < 16 or int.from_bytes(data[4:8], "big") < 1:
        return None
    return data[12:16].decode("latin-1", "replace").strip() or None


def parse_stts(data: bytes) -> tuple[int, int] | tuple[None, None]:
    if len(data) < 8:
        return None, None
    entries = int.from_bytes(data[4:8], "big")
    if len(data) < 8 + entries * 8:
        return None, None
    samples, ticks = 0, 0
    for position in range(8, 8 + entries * 8, 8):
        count = int.from_bytes(data[position:position + 4], "big")
        delta = int.from_bytes(data[position + 4:position + 8], "big")
        samples += count
        ticks += count * delta
    return samples, ticks


def inspect_file(path: Path) -> VideoRecord:
    record = VideoRecord(video_id=path.stem, filename=path.name, size_bytes=path.stat().st_size, status="ok", issue="")
    if record.size_bytes == 0:
        record.status, record.issue = "error", "empty file"
        return record
    try:
        with path.open("rb") as handle:
            top = list(child_boxes(handle, 0, record.size_bytes, allow_truncated_mdat=True))
            kinds = [item.kind for item in top]
            if b"ftyp" not in kinds or b"moov" not in kinds or b"mdat" not in kinds:
                raise ValueError("missing required ftyp/moov/mdat box")
            moov = next(item for item in top if item.kind == b"moov")
            mdat = next(item for item in top if item.kind == b"mdat")
            record.moov_before_mdat = moov.offset < mdat.offset
            declared_mdat_size = int.from_bytes(read_at(handle, mdat.offset, 4), "big")
            record.mdat_size_mismatch = declared_mdat_size not in (0, 1) and mdat.offset + declared_mdat_size > record.size_bytes
            video_track_data: list[tuple[float, int, int, float | None, int | None, str | None]] = []
            for track in descendants(handle, moov, b"trak"):
                hdlr = first_descendant(handle, track, b"hdlr")
                if hdlr is None:
                    continue
                handler_type = parse_handler(payload(handle, hdlr))
                if handler_type == b"soun":
                    record.audio_tracks += 1
                    continue
                if handler_type != b"vide":
                    continue
                record.video_tracks += 1
                mdhd = first_descendant(handle, track, b"mdhd")
                tkhd = first_descendant(handle, track, b"tkhd")
                if mdhd is None or tkhd is None:
                    raise ValueError("video track lacks mdhd or tkhd")
                timescale, duration_ticks = parse_mdhd(payload(handle, mdhd))
                if timescale <= 0 or duration_ticks <= 0:
                    raise ValueError("non-positive video duration/timescale")
                width, height = parse_track_dimensions(payload(handle, tkhd))
                stsd = first_descendant(handle, track, b"stsd")
                stts = first_descendant(handle, track, b"stts")
                frames, decode_ticks = parse_stts(payload(handle, stts, 128 * 1024)) if stts else (None, None)
                duration = duration_ticks / timescale
                if decode_ticks and decode_ticks > 0:
                    duration = decode_ticks / timescale
                fps = frames / duration if frames and duration else None
                codec = parse_stsd_codec(payload(handle, stsd)) if stsd else None
                video_track_data.append((duration, width, height, fps, frames, codec))
            if not video_track_data:
                raise ValueError("no video track")
            # WLASL clips normally have one video track. Use the longest if a file has more.
            duration, width, height, fps, frames, codec = max(video_track_data, key=lambda item: item[0])
            record.duration_seconds, record.width, record.height = duration, width, height
            record.fps, record.frames, record.codec = fps, frames, codec
            record.bitrate_mbps = record.size_bytes * 8 / duration / 1_000_000
            anomalies = ["mdat declares bytes beyond EOF"] if record.mdat_size_mismatch else []
            if width <= 0 or height <= 0:
                anomalies.append("invalid dimensions")
            if fps is not None and not 1 <= fps <= 120:
                anomalies.append("unusual fps")
            if duration < 0.1:
                anomalies.append("very short duration")
            if record.bitrate_mbps < 0.02 or record.bitrate_mbps > 50:
                anomalies.append("unusual bitrate")
            if anomalies:
                record.status, record.issue = "warning", "; ".join(anomalies)
    except (OSError, ValueError, OverflowError) as error:
        record.status, record.issue = "error", str(error)
    return record


def bar_chart_svg(title: str, data: list[tuple[str, int]], width: int = 700, max_items: int = 12) -> str:
    data = data[:max_items]
    if not data:
        return ""
    height, left, right, top = max(180, 48 + len(data) * 32), 170, 62, 32
    plot = width - left - right
    maximum = max(value for _, value in data) or 1
    rows = []
    for index, (label, value) in enumerate(data):
        y, bar_width = top + index * 32, plot * value / maximum
        rows.append(f"<text x='{left - 10}' y='{y+16}' class='label' text-anchor='end'>{html.escape(label)}</text><rect class='bar' x='{left}' y='{y+3}' width='{bar_width:.1f}' height='20' rx='3'/><text class='value' x='{left+bar_width+7:.1f}' y='{y+17}'>{fmt_int(value)}</text>")
    return f"<section class='chart'><h3>{html.escape(title)}</h3><svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>{''.join(rows)}</svg></section>"


def histogram_svg(title: str, values: list[float], unit: str, bins: int = 14, width: int = 700) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    high = high if high > low else low + 1
    step, counts = (high - low) / bins, [0] * bins
    for value in values:
        counts[min(int((value - low) / step), bins - 1)] += 1
    height, left, right, top, bottom = 255, 52, 20, 25, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum, bar_width = max(counts) or 1, plot_width / bins
    bars = "".join(f"<rect class='bar alt' x='{left+i*bar_width+1:.1f}' y='{top+plot_height-plot_height*count/maximum:.1f}' width='{bar_width-2:.1f}' height='{plot_height*count/maximum:.1f}' rx='2'/>" for i, count in enumerate(counts))
    ticks = "".join(f"<text class='tick' x='{left+plot_width*i/4:.1f}' y='{height-13}' text-anchor='middle'>{fmt_float(low+(high-low)*i/4, 1)}</text>" for i in range(5))
    return f"<section class='chart'><h3>{html.escape(title)}</h3><p>Trục x: {html.escape(unit)} · n = {fmt_int(len(values))}</p><svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'><line class='axis' x1='{left}' y1='{top+plot_height}' x2='{width-right}' y2='{top+plot_height}'/>{bars}{ticks}</svg></section>"


def table_html(headers: list[str], rows: list[list[str]]) -> str:
    return "<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{html.escape(item)}</th>" for item in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def analyze(video_dir: Path) -> tuple[list[VideoRecord], dict]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")
    records = [inspect_file(path) for path in sorted(video_dir.glob("*.mp4"))]
    okay = [record for record in records if record.status != "error" and record.duration_seconds]
    durations = [record.duration_seconds for record in okay if record.duration_seconds is not None]
    widths = [float(record.width) for record in okay if record.width]
    heights = [float(record.height) for record in okay if record.height]
    fps_values = [record.fps for record in okay if record.fps is not None]
    bitrates = [record.bitrate_mbps for record in okay if record.bitrate_mbps is not None]
    resolutions = Counter(f"{record.width}×{record.height}" for record in okay)
    codecs = Counter(record.codec or "unknown" for record in okay)
    statuses = Counter(record.status for record in records)
    fps_buckets = Counter(f"{record.fps:.3f}" for record in okay if record.fps is not None)
    audio = Counter("with audio" if record.audio_tracks else "no audio" for record in okay)
    total_duration = sum(durations)
    return records, {
        "files_scanned": len(records), "status_counts": dict(statuses), "total_bytes": sum(record.size_bytes for record in records),
        "total_duration_seconds": round(total_duration, 3), "total_duration_hours": round(total_duration / 3600, 3),
        "duration_seconds": describe(durations), "width": describe(widths), "height": describe(heights),
        "fps": describe(fps_values), "bitrate_mbps": describe(bitrates),
        "resolutions": dict(resolutions.most_common()), "codecs": dict(codecs.most_common()),
        "fps_buckets": dict(fps_buckets.most_common()), "audio": dict(audio),
        "faststart": {"moov_before_mdat": sum(record.moov_before_mdat is True for record in okay), "moov_after_mdat": sum(record.moov_before_mdat is False for record in okay)},
        "warnings": sum(record.status == "warning" for record in records), "errors": sum(record.status == "error" for record in records),
    }


def write_outputs(records: list[VideoRecord], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = [asdict(record) for record in records]
    fields = list(inventory[0]) if inventory else list(VideoRecord.__dataclass_fields__)
    with (output_dir / "video_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    issues = [row for row in inventory if row["status"] != "ok"]
    with (output_dir / "video_issues.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)
    (output_dir / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = [record for record in records if record.status != "error" and record.duration_seconds]
    resolution_rows = [[resolution, fmt_int(count), f"{count / len(valid) * 100:.2f}%"] for resolution, count in Counter(f"{record.width}×{record.height}" for record in valid).most_common(25)]
    codec_rows = [[codec, fmt_int(count), f"{count / len(valid) * 100:.2f}%"] for codec, count in Counter(record.codec or "unknown" for record in valid).most_common()]
    status_rows = [[name, fmt_int(count)] for name, count in sorted(summary["status_counts"].items())]
    numeric_rows = [["Duration", "seconds", *[fmt_float(summary['duration_seconds'].get(key)) for key in ('min','p25','median','mean','p75','p95','max')]], ["FPS", "frames/s", *[fmt_float(summary['fps'].get(key)) for key in ('min','p25','median','mean','p75','p95','max')]], ["Bitrate", "Mb/s", *[fmt_float(summary['bitrate_mbps'].get(key)) for key in ('min','p25','median','mean','p75','p95','max')]], ["Width", "px", *[fmt_float(summary['width'].get(key)) for key in ('min','p25','median','mean','p75','p95','max')]], ["Height", "px", *[fmt_float(summary['height'].get(key)) for key in ('min','p25','median','mean','p75','p95','max')]]]
    duration_values = [record.duration_seconds for record in valid if record.duration_seconds is not None]
    bitrate_values = [record.bitrate_mbps for record in valid if record.bitrate_mbps is not None]
    charts = "".join([
        histogram_svg("Phân bố thời lượng clip", duration_values, "giây"),
        histogram_svg("Phân bố bitrate container", bitrate_values, "Mb/s"),
        bar_chart_svg("Độ phân giải phổ biến", list(Counter(f"{r.width}×{r.height}" for r in valid).most_common(12))),
        bar_chart_svg("FPS đo từ timeline (top)", list(Counter(f"{r.fps:.3f}" for r in valid if r.fps is not None).most_common(12))),
        bar_chart_svg("Codec video", list(Counter(r.codec or "unknown" for r in valid).most_common(12))),
    ])
    cards = [("Video đã quét", fmt_int(summary["files_scanned"])), ("Không lỗi container", fmt_int(len(valid))), ("Tổng thời lượng", f"{fmt_float(summary['total_duration_hours'])} giờ"), ("Dung lượng", fmt_bytes(summary["total_bytes"])), ("Cảnh báo", fmt_int(summary["warnings"])), ("Lỗi metadata", fmt_int(summary["errors"]))]
    card_html = "".join(f"<article class='card'><p>{name}</p><strong>{value}</strong></article>" for name, value in cards)
    html_doc = f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>WLASL video quality audit</title><style>
:root{{--ink:#162033;--muted:#647084;--line:#dbe2eb;--bg:#f5f8fc;--blue:#285fda;--teal:#009987}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1420px;margin:auto;padding:32px 24px 52px}}h1{{font-size:30px;margin:0 0 4px}}h2{{margin:36px 0 13px;font-size:21px}}h3{{font-size:16px;margin:0}}p{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:22px 0}}.card,.chart,.panel{{background:white;border:1px solid var(--line);border-radius:12px;padding:15px;box-shadow:0 1px 2px #17203309}}.card p{{font-size:13px;margin:0}}.card strong{{font-size:25px;display:block;margin-top:5px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:15px}}.chart svg{{display:block;width:100%;height:auto;margin-top:7px}}.label{{font-size:12px;fill:#3d4a5c}}.value{{font-size:12px;fill:#3d4a5c;font-weight:600}}.bar{{fill:var(--blue)}}.bar.alt{{fill:var(--teal)}}.axis{{stroke:#b8c2d0}}.tick{{font-size:11px;fill:#647084}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}}table{{border-collapse:collapse;width:100%;min-width:560px}}th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}}th{{font-size:12px;background:#eff4fa;text-transform:uppercase;letter-spacing:.03em}}tr:last-child td{{border-bottom:0}}.callout{{background:#eaf3ff;border-left:4px solid var(--blue);padding:13px 16px;border-radius:0 8px 8px 0}}@media(max-width:600px){{main{{padding:22px 12px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>WLASL — Kiểm tra chất lượng video cục bộ</h1><p>Quét toàn bộ MP4 trong <code>data/raw/videos/</code>. Metadata được đọc trực tiếp từ MP4/ISO-BMFF, không tái mã hóa hay chỉnh sửa dữ liệu.</p></header><section class='cards'>{card_html}</section>
<section class='callout'><strong>Phạm vi kiểm tra:</strong> cấu trúc ftyp/moov/mdat, video track, codec, độ phân giải, thời lượng/FPS/frame count từ timing tables, bitrate tệp và audio track. <strong>Không bao gồm:</strong> giải mã từng frame nên không khẳng định được lỗi hình, rung, tối/nhòe hay audio bị rè bên trong container hợp lệ.</section>
<h2>1. Trạng thái và thống kê số</h2><div class='grid'><section class='panel'><h3>Trạng thái container</h3>{table_html(['Trạng thái','Số video'], status_rows)}</section><section class='panel'><h3>Phân vị</h3>{table_html(['Đại lượng','Đơn vị','Min','P25','Median','Mean','P75','P95','Max'], numeric_rows)}</section></div>
<h2>2. Phân phối kỹ thuật</h2><div class='grid'>{charts}</div>
<h2>3. Resolution, codec và streaming layout</h2><div class='grid'><section class='panel'><h3>Top độ phân giải</h3>{table_html(['Resolution','Số video','Tỉ trọng'], resolution_rows)}</section><section class='panel'><h3>Codec</h3>{table_html(['Codec','Số video','Tỉ trọng'], codec_rows)}<p><strong>moov trước mdat:</strong> {fmt_int(summary['faststart']['moov_before_mdat'])}; <strong>moov sau mdat:</strong> {fmt_int(summary['faststart']['moov_after_mdat'])}.</p><p><strong>Audio:</strong> {html.escape(', '.join(f'{name}: {count}' for name, count in summary['audio'].items()))}.</p></section></div>
<h2>4. Các video cần xem xét</h2><p>{fmt_int(len(issues))} video có warning/error theo ngưỡng kỹ thuật bảo thủ. Xem danh sách đầy đủ tại <code>video_issues.csv</code>; inventory toàn bộ tại <code>video_inventory.csv</code>.</p>{table_html(['Video ID','Status','Issue','Duration (s)','Resolution','FPS','Bitrate (Mb/s)'], [[r.video_id,r.status,r.issue,fmt_float(r.duration_seconds),f'{r.width}×{r.height}' if r.width and r.height else '—',fmt_float(r.fps),fmt_float(r.bitrate_mbps)] for r in [VideoRecord(**row) for row in issues[:100]]])}
<footer><p>Đầu ra đi kèm: <code>quality_summary.json</code>, <code>video_inventory.csv</code>, <code>video_issues.csv</code>.</p></footer></main></body></html>"""
    (output_dir / "dashboard.html").write_text(html_doc, encoding="utf-8")
    readme = f"""# WLASL local video-quality audit

Dashboard: [dashboard.html](dashboard.html)

- Quét: {fmt_int(summary['files_scanned'])} MP4; lỗi container: {fmt_int(summary['errors'])}; cảnh báo: {fmt_int(summary['warnings'])}.
- Tổng thời lượng theo timeline video: {fmt_float(summary['total_duration_hours'])} giờ.
- CSV inventory có một dòng/video; CSV issues chỉ chứa video warning/error.

Giới hạn: báo cáo kiểm tra metadata MP4, không giải mã toàn bộ frame/audio.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit technical metadata of WLASL MP4 files.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    records, summary = analyze(args.video_dir)
    write_outputs(records, summary, args.out_dir)
    print(f"Video-quality report created at: {args.out_dir}")
    print(f"Scanned: {summary['files_scanned']}; errors: {summary['errors']}; warnings: {summary['warnings']}")


if __name__ == "__main__":
    main()
