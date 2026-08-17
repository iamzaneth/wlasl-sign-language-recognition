# WLASL Sign Language Recognition

An end-to-end research pipeline for isolated American Sign Language recognition on the
[WLASL](https://github.com/dxli94/WLASL) dataset. The project audits locally available
videos, preserves the official WLASL splits, extracts MediaPipe pose/hand/face
landmarks, normalizes the landmark sequences, and trains Transformer encoder-only
classifiers on configurable landmark subsets.

This repository currently focuses on data preparation, landmark-based experiments,
and evaluation. It does **not** include a webcam demo, an inference CLI, or a deployment
API.

## What is implemented

- Dataset inventory and video-readability auditing without modifying source files.
- Official WLASL100, WLASL300, WLASL1000, and WLASL2000 split generation, restricted
  to locally available readable videos.
- Resumable MediaPipe Tasks extraction for pose, both hands, mouth, eyes, and eyebrows.
- Shoulder-centered XY normalization and conservative short-gap hand recovery.
- A reusable, memory-mapped, fixed-length feature cache.
- A CLS-token Transformer encoder classifier with class balancing, augmentation,
  early stopping, AMP, and CPU/CUDA support.
- Sequential landmark ablations and resumable, multi-seed adaptive WLASL100 tuning.
- Portable JSON/CSV experiment metadata and validation-selected model summaries.

## Pipeline

```text
WLASL annotations + local MP4 files
                |
                v
       audit and usable splits
                |
                v
     raw MediaPipe XYZ landmarks
                |
                v
 shoulder normalization + hand recovery
                |
                v
       fixed-length feature cache
                |
                v
       Transformer encoder training
                |
                v
       metrics, predictions, checkpoints
```

## Quick start

Run commands from the repository root. Python 3.10 or newer is required; the recorded
experiments used Python 3.12.

### 1. Create an environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`requirements.txt` pins `torch==2.11.0+cu128` for the project's CUDA 12.8 training
environment. Install that exact environment with:

```powershell
python -m pip install -r requirements.txt
```

If the CUDA 12.8 wheel is not compatible with your platform or hardware, install an
appropriate PyTorch build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) and install the
remaining dependencies separately:

```powershell
python -m pip install "numpy>=1.26,<3" "opencv-python>=4.8,<5" "mediapipe>=0.10.14,<0.11" "pytest>=8,<9"
```

### 2. Provide the WLASL source data

The repository does not download or redistribute WLASL. Obtain the dataset from its
official source and arrange the raw files as follows:

```text
data/raw/
|-- WLASL_v0.3.json
|-- wlasl_class_list.txt
|-- nslt_100.json
|-- nslt_300.json
|-- nslt_1000.json
|-- nslt_2000.json
`-- videos/
    |-- 00335.mp4
    |-- 00336.mp4
    `-- ...
```

The end-to-end audit computes coverage for every supported subset, so all four
`nslt_*.json` files are expected even when processing only WLASL100. The class list is
recommended because it preserves the supplied class IDs; the code has a deterministic
lexical fallback when it is absent.

### 3. Run the data pipeline

Start with WLASL100:

```powershell
python -m src.pipeline.run_data_pipeline --subset wlasl100
```

This command audits the raw videos, creates usable official splits, downloads the three
MediaPipe Tasks model bundles when needed, extracts raw landmarks, and creates processed
landmarks. Existing schema-valid landmark files are reused, so interrupted runs can be
started again with the same command. Landmark extraction over a large video collection
is computationally expensive; use the standalone commands in the
[data pipeline guide](docs/DATA_PIPELINE.md) for small validation runs.

### 4. Build features and run a training smoke test

```powershell
python scripts/prepare_encoder_features.py --subset wlasl100 --seq-len 60
python scripts/train_encoder_only.py --subset wlasl100 --landmark-subsets hands --epochs 2 --device auto
```

For full experiments, remove `--epochs 2`, select one or more canonical landmark
subsets, or use `configs/encoder_only_example.json`. See the
[encoder training guide](docs/TRAINING_ENCODER_ONLY.md).

### 5. Run tests

```powershell
pytest
```

## Current result snapshot

The tracked WLASL100 cache contains 1,013 usable videos: 748 train, 165 validation,
and 100 test samples. This is only 49.7% of the 2,038 entries in the official WLASL100
manifest, and the local test split represents 72 of the 100 classes. Consequently, the
tracked metrics are a project snapshot on an incomplete local corpus, not a directly
comparable full-WLASL100 benchmark.

Among the currently published single-run winners, the hands-only model has the highest
test macro-F1 (`0.7019`) and test top-1 accuracy (`0.6900`). Model selection used
validation macro-F1; test metrics were reporting-only. Full per-subset values, dataset
coverage, and interpretation are documented in [Results and limitations](docs/RESULTS.md).

## Documentation

- [Data pipeline](docs/DATA_PIPELINE.md): raw-data contract, audit, extraction,
  preprocessing, schemas, recovery behavior, and troubleshooting.
- [Encoder-only training](docs/TRAINING_ENCODER_ONLY.md): feature cache, model,
  configuration, metrics, artifacts, and adaptive tuning.
- [Results and limitations](docs/RESULTS.md): current data coverage, tracked WLASL100
  results, and comparison caveats.

## Repository layout

```text
configs/       Landmark subset definitions and example training trials
data/          Raw inputs, generated manifests, splits, landmarks, and feature caches
docs/          Detailed project documentation
models/        Local MediaPipe task bundles (not committed)
output/        Training summaries, experiment metadata, and local run artifacts
reports/       Generated dataset, video-quality, and visualization reports
scripts/       Feature preparation, training, and tuning entrypoints
src/config/    Repository-relative path definitions
src/data/      Dataset readers, audits, analysis, and split construction
src/pipeline/  End-to-end data-pipeline orchestration
src/preprocessing/  Landmark extraction, validation, normalization, and recovery
src/training/  Feature cache, model, training loop, and adaptive tuning
tests/         Path, split, preprocessing, feature-cache, and tuning tests
```

## Generated and tracked artifacts

Raw videos, MediaPipe task bundles, per-video landmark NPZ files, training run
directories, logs, reports, and PyTorch checkpoints are excluded from Git. Small
manifests, split CSVs, the current WLASL100 feature cache, aggregate metrics, and best-run
metadata are tracked to make the experiment contract inspectable.

Because `*.pt` and nested `runs/` directories are ignored, a fresh clone contains the
recorded metrics and configurations but not trained weights or the complete run history.
Reproduce a checkpoint by preparing the corresponding data and rerunning its saved
configuration.

## Data, models, and licensing

WLASL data and MediaPipe model assets remain subject to their upstream licenses and
terms. Cite the original WLASL work when using the dataset or publishing results. This
repository does not currently include a project-level license file; do not assume rights
beyond those granted by the upstream resources and the repository owner.
