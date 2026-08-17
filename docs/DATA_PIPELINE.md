# WLASL Data Pipeline

This guide describes the repository's path from immutable WLASL source files to
model-ready landmark sequences. Commands are intended to be run from the repository
root.

## Data contract

The pipeline supports `wlasl100`, `wlasl300`, `wlasl1000`, and `wlasl2000`. There is no
WLASL3000 option in this project.

Place source files under `data/raw`:

```text
data/raw/
|-- WLASL_v0.3.json
|-- wlasl_class_list.txt
|-- nslt_100.json
|-- nslt_300.json
|-- nslt_1000.json
|-- nslt_2000.json
`-- videos/
    `-- <video_id>.mp4
```

`WLASL_v0.3.json` and the four official `nslt_*.json` maps are required by the complete
audit. `wlasl_class_list.txt` should also be supplied so `classes.csv` and
`gloss_to_id.json` retain the original class IDs. If the class list is absent, the audit
creates a deterministic lexical mapping instead.

The pipeline never downloads WLASL and never edits files below `data/raw`. Video files
that are missing or unreadable are omitted from generated splits; they are not replaced
or fabricated.

## Directory lifecycle

```text
data/raw/                                      immutable user-provided source data
    |
    +--> data/interim/manifests/               audit inventory and valid/invalid lists
    |
    +--> data/splits/<subset>/                 usable official train/val/test CSV files
    |
    +--> data/interim/landmarks/mediapipe_v1/  raw per-video MediaPipe XYZ NPZ files
    |
    `--> data/processed/landmarks/
         shoulder_norm_v1/                     normalized/recovered per-video NPZ files
              |
              `--> data/processed/encoder_features/
                   <subset>/seq<length>/       fixed-length training cache
```

Train, validation, and test videos are not copied into separate directories. Split CSVs
reference video IDs and preserve the official assignment.

## End-to-end command

```powershell
python -m src.pipeline.run_data_pipeline --subset wlasl100
```

The orchestrator runs five stages:

1. Audit all annotated local videos, unless `valid_videos.csv` already exists.
2. Rebuild the requested subset's usable official split CSVs.
3. Reuse or download the three required MediaPipe task bundles.
4. Extract raw landmarks for every usable video in the selected subset.
5. Normalize landmarks and recover eligible short hand gaps.

Important options:

| Option | Behavior |
| --- | --- |
| `--subset` | Required; one of the four supported WLASL subsets. |
| `--refresh-audit` | Re-scan all raw videos even when `valid_videos.csv` exists. |
| `--model-dir PATH` | Override the MediaPipe task-bundle directory. |
| `--skip-model-download` | Require valid local task bundles and make no download attempt. |
| `--overwrite` | Regenerate existing raw and processed landmark NPZ files. |
| `--max-gap-frames N` | Set the maximum recoverable internal hand gap; default `2`. |
| `--report-every N` | Persist extraction results every N videos; default `25`. |
| `--input-is-mirrored` | Treat input as selfie-mirrored rather than normal WLASL footage. |
| `--show-mediapipe-logs` | Show native MediaPipe/TFLite diagnostic output. |

The presence of `valid_videos.csv` is the orchestrator's audit reuse condition; its
contents are not automatically checked against later changes in `data/raw`. Use
`--refresh-audit` after adding, removing, or replacing videos.

## Stage 1: audit the local dataset

```powershell
python -m src.data.audit_dataset
```

For every annotation instance, the audit checks file existence, opens the MP4 with
OpenCV, validates basic stream metadata, and decodes one frame. A failure is isolated to
that video so one bad file does not abort the full audit.

Outputs:

| Path | Contents |
| --- | --- |
| `data/interim/manifests/video_inventory.csv` | One row per annotated video with availability, readability, dimensions, FPS, duration, and error status. |
| `data/interim/manifests/valid_videos.csv` | Inventory rows that passed the readability check. |
| `data/interim/manifests/invalid_videos.csv` | Missing or unreadable video IDs and reasons. |
| `data/metadata/dataset_summary.json` | Overall, per-split, and per-subset availability coverage. |
| `data/metadata/classes.csv` | Class ID and gloss table. |
| `data/metadata/gloss_to_id.json` | Gloss-to-class-ID mapping. |

The audit decodes only the first frame. It is an availability/readability gate, not a
complete visual-quality inspection of every frame.

## Stage 2: build usable official splits

Build one subset:

```powershell
python -m src.data.build_splits --subset wlasl100
```

Build all supported subsets:

```powershell
python -m src.data.build_splits
```

The builder intersects the official NSLT map with `valid_videos.csv` and
`WLASL_v0.3.json`. It preserves:

- the official `train`, `val`, or `test` assignment;
- `action[0]` from the selected `nslt_*.json` as the numeric label; and
- the gloss from `WLASL_v0.3.json`.

Each output row has `video_id`, `gloss`, `label`, and `split`. Entries missing from the
local readable corpus or inconsistent with the source annotations are omitted.

## Stage 3: obtain MediaPipe model assets

```powershell
python -m src.preprocessing.download_mediapipe_models
```

The downloader retrieves the official full pose, hand, and face MediaPipe Tasks bundles:

```text
models/mediapipe/pose_landmarker.task
models/mediapipe/hand_landmarker.task
models/mediapipe/face_landmarker.task
```

Downloads use a temporary `.part` file and replace the destination only after a basic
size validation. Valid existing assets are reused. The model directory can also be set
with `--model-dir` or the `WLASL_MEDIAPIPE_MODEL_DIR` environment variable. These binary
assets are intentionally not committed.

## Stage 4: extract raw landmarks

Always validate extraction on a small sample first:

```powershell
python -m src.preprocessing.extract_landmarks --subset wlasl100 --limit 5
```

Extract selected videos instead of a complete subset:

```powershell
python -m src.preprocessing.extract_landmarks --video-id 00335 --video-id 00336
```

Extraction uses MediaPipe Tasks Vision in VIDEO mode with at most one pose, two hands,
and one face. Output is one compressed NPZ per video under
`data/interim/landmarks/mediapipe_v1`.

### Raw `mediapipe_v1` schema

All coordinate arrays are `float32` with shape `[T, points, 3]` and coordinate order
`[x, y, z]`:

| Field | Points | Selection |
| --- | ---: | --- |
| `pose_xyz` | 9 | Nose; left/right shoulders, elbows, wrists, and hips. |
| `left_hand_xyz` | 21 | Full left hand. |
| `right_hand_xyz` | 21 | Full right hand. |
| `mouth_xyz` | 40 | Explicit lip topology without duplicate indices. |
| `left_eye_xyz` | 16 | Left eye contour; iris excluded. |
| `right_eye_xyz` | 16 | Right eye contour; iris excluded. |
| `left_eyebrow_xyz` | 10 | Left eyebrow. |
| `right_eyebrow_xyz` | 10 | Right eyebrow. |

Missing detections are stored as `NaN`, never as an observed all-zero landmark. The
frame-level masks `pose_observed`, `left_hand_observed`, `right_hand_observed`, and
`face_observed` record whether the source detector returned that modality.

Each NPZ also stores `video_id`, `fps`, decoded `num_frames`, `width`, `height`,
`schema_version="mediapipe_v1"`, and `coordinate_order=["x", "y", "z"]`.

MediaPipe handedness labels assume mirrored selfie input. Normal WLASL footage is
externally recorded, so the extractor reverses the detector's handedness label by
default. Use `--input-is-mirrored` only when the input itself is selfie-mirrored. The
coordinates are never mirrored by this option; only the left/right label mapping changes.

Extraction records per-video detection rates and errors in
`data/metadata/landmark_extraction_report.csv`. It updates this report every 25 videos by
default and once more when the loop exits or is interrupted.

## Stage 5: normalize and recover hand gaps

```powershell
python -m src.preprocessing.preprocess_landmarks --subset wlasl100 --limit 5
```

Processed files are written to `data/processed/landmarks/shoulder_norm_v1` and retain the
raw schema fields.

### Shoulder normalization

For each frame `t`, let `C_t` be the midpoint between the detected shoulders and `d_t`
their XY distance. The video-level scale is the median valid shoulder distance:

```text
d_video = median(d_t for valid shoulder frames)
xy_processed(t) = (xy_raw(t) - C_t) / (d_video + epsilon)
```

Only XY is transformed. MediaPipe Z values are preserved because pose, hand, and face Z
do not form a shared metric 3D coordinate system.

If a frame lacks a finite shoulder pair, that frame's coordinates remain unchanged and
`normalization_valid[t]` is false. If no frame has a valid shoulder pair, the full video's
coordinates remain unchanged and `shoulder_scale` is `NaN`. Consumers that require a
single normalized coordinate space should inspect `normalization_valid`; the current
feature-cache builder does not filter these frames automatically.

Additional processed fields include:

- `normalization_valid[T]`
- `normalization_center_xy[T, 2]`
- `shoulder_scale`
- `xy_normalized=True` and `z_normalized=False`
- `normalization="shoulder_norm_v1"`
- `hand_recovery="wrist_anchored_linear"`

### Short-gap hand recovery

Only internal missing-hand runs with observed frames on both sides are candidates. A gap
is recovered when all of the following are true:

- its length is at most `--max-gap-frames` (default `2`);
- both endpoint hand arrays are finite;
- endpoint pose wrists are finite; and
- the pose wrist is finite in every missing target frame.

For XY, the hand is expressed relative to the pose wrist at both endpoints, linearly
interpolated, and attached to the target frame's wrist. Z is linearly interpolated
directly. Boundary gaps, long gaps, and gaps with missing anchors remain unresolved.

The masks `left_hand_imputed`, `right_hand_imputed`, `left_hand_unresolved`, and
`right_hand_unresolved` distinguish generated values from unresolved missing detections.
The original `*_hand_observed` masks are not changed.

## Resume and overwrite behavior

- Raw extraction skips an existing NPZ only when it passes the raw schema validator.
- Preprocessing skips an existing NPZ only when it passes the processed schema validator.
- `--overwrite` regenerates both stages in the end-to-end command.
- Split CSVs are rebuilt every time the orchestrator runs.
- Audit manifests are reused based on file presence; use `--refresh-audit` when raw data
  changes.
- A failure for one video is recorded and processing continues with later videos.

## Quality inspection and optional reports

Summarize landmark detection and recovery quality:

```powershell
python -m src.preprocessing.summarize_extraction
```

Render raw and processed overlays on the source RGB frame:

```powershell
python -m src.preprocessing.visualize_landmarks --video-id 00335 --frame 0
python -m src.preprocessing.visualize_landmarks --video-id 00335 --processed --frame 0
```

Default images are written under `reports/landmark_debug`. Processed XY coordinates are
mapped back with the saved shoulder center and scale before drawing; Z is not rendered.

Two additional read-only reporting tools are available:

```powershell
python -m src.data.analyze_dataset
python -m src.data.inspect_video_quality
```

They write self-contained HTML, JSON, and CSV reports under `reports/dataset_analysis`
and `reports/video_quality`. The metadata analysis does not decode video frames. The
video-quality tool inspects MP4 container and track metadata and likewise does not prove
that every visual frame is artifact-free.

## Troubleshooting

### A changed raw dataset is not reflected in the splits

Refresh the audit before rebuilding the splits:

```powershell
python -m src.pipeline.run_data_pipeline --subset wlasl100 --refresh-audit
```

### MediaPipe assets cannot be downloaded

Place the three `.task` files in `models/mediapipe` (or another `--model-dir`) and run:

```powershell
python -m src.pipeline.run_data_pipeline --subset wlasl100 --skip-model-download
```

### Native detector diagnostics are hidden

Python exceptions are still recorded per video. To expose MediaPipe and TensorFlow Lite
native logs as well, add `--show-mediapipe-logs`.

### A file is repeatedly reprocessed

The existing NPZ is probably absent, incomplete, or incompatible with the expected
schema. Inspect the extraction report and rerun the individual video with `--overwrite`
and `--show-mediapipe-logs`.
