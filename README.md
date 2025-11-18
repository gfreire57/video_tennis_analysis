# Tennis Stroke Recognition System

A complete tennis stroke recognition and analysis system using MediaPipe pose estimation and LSTM neural networks. The system can classify tennis strokes (forehand, backhand, serve, slices) and analyze continuous video to generate stroke timelines and statistics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Annotate Videos](#1-annotate-videos)
  - [2. Verify Annotations](#2-verify-annotations)
  - [3. Train Model](#3-train-model)
  - [4. Analyze New Videos](#4-analyze-new-videos)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

## Overview

This project provides an end-to-end pipeline for:

1. **Video Annotation**: Using Label Studio to annotate tennis strokes in videos
2. **Model Training**: Training an LSTM neural network to classify stroke types from pose data
3. **Video Analysis**: Detecting and analyzing strokes in continuous tennis videos
4. **Experiment Tracking**: Using MLflow to track and compare training experiments

The system uses **MediaPipe Pose** to extract 33 body landmarks (132 features per frame) and an **LSTM network** to recognize temporal patterns in stroke sequences.

## Features

- **Pose-based recognition**: Uses body keypoints instead of raw pixels for robust detection
- **Temporal modeling**: LSTM networks capture the sequential nature of tennis strokes
- **Continuous video analysis**: Sliding window detection with configurable parameters
- **Timeline visualization**: Color-coded stroke timelines with frequency statistics
- **Experiment tracking**: MLflow integration for reproducible research
- **GPU support**: Automatic GPU detection and configuration (optional)
- **Flexible annotation**: Compatible with Label Studio export formats

## Installation

### Requirements

- Python 3.11 or higher
- Poetry (for dependency management)
- (Optional) CUDA-capable GPU for faster training

### Install Dependencies

```bash
# Clone the repository
cd video_tennis_analysis

# Install dependencies using Poetry
poetry install
```

### Dependencies Installed

- `mediapipe`: Pose estimation (body landmark detection)
- `opencv-python`: Video processing
- `tensorflow`: Deep learning framework for LSTM
- `scikit-learn`: Data preprocessing and evaluation
- `matplotlib`: Visualization
- `mlflow`: Experiment tracking

## Quick Start

### 1. Verify Your Annotations

```bash
poetry run python src/verify_annotation.py
```

This will check your Label Studio annotations and generate a verification report.

### 2. Train the Model

```bash
poetry run python src/train_model.py
```

Training will:
- Extract pose landmarks from all annotated videos
- Create training sequences with sliding windows
- Train an LSTM classifier
- Save the model and metrics
- Track the experiment in MLflow (if enabled)

### 3. Analyze a New Video

```bash
poetry run python src/detect_strokes.py path/to/your/video.mp4
```

This will:
- Detect all strokes in the video
- Generate a timeline visualization
- Create a detailed report with timestamps
- Save stroke data as JSON

## Project Structure

```
video_tennis_analysis/
├── src/
│   ├── train_model.py          # Main training pipeline
│   ├── detect_strokes.py       # Continuous video analysis
│   ├── verify_annotation.py    # Annotation verification
│   └── check_gpu.py            # GPU capability checker
├── data/
│   ├── annotations/            # Label Studio JSON exports
│   └── videos/                 # Training videos
├── output/                     # Training outputs
│   ├── tennis_stroke_model.keras    # Trained model
│   ├── label_classes.npy            # Class labels
│   ├── training_history.png         # Training curves
│   └── confusion_matrix.txt         # Evaluation metrics
├── analysis_output/            # Video analysis results
│   ├── *_timeline.png          # Stroke timelines
│   ├── *_report.txt            # Detailed reports
│   └── *_strokes.json          # Stroke data
├── mlruns/                     # MLflow experiment data
├── pyproject.toml              # Project dependencies
├── README.md                   # This file
├── MLFLOW_GUIDE.md            # MLflow documentation
├── DEVELOPMENT_NOTES.md        # Design decisions
└── USAGE_GUIDE.md             # Detailed usage examples
```

## Usage

### 1. Annotate Videos

Use [Label Studio](https://labelstud.io/) to annotate your tennis videos:

1. Create a new project in Label Studio
2. Upload your tennis videos to `data/videos/`
3. Use timeline labels to mark stroke segments
4. Export annotations as JSON to `data/annotations/`

**Supported stroke types:**
- `fronthand` (or `forehand`)
- `backhand`
- `saque` (or `serve`)
- `slice direita` (right slice)
- `slice esquerda` (left slice)

### 2. Verify Annotations

Before training, verify your annotations:

```bash
poetry run python src/verify_annotation.py
```

**Output**: `verification_report.md` with:
- Annotation statistics
- Video file checks
- Label distribution
- Potential issues

### 3. Train Model

#### Basic Training

```bash
poetry run python src/train_model.py
```

#### Configuration

Edit the `CONFIG` dictionary in [train_model.py](src/train_model.py) to adjust:

```python
CONFIG = {
    'window_size': 30,           # Frames per sequence
    'overlap': 15,               # Frame overlap between windows
    'batch_size': 32,            # Training batch size
    'epochs': 100,               # Maximum epochs
    'learning_rate': 0.001,      # Optimizer learning rate
    'use_mlflow': True,          # Enable MLflow tracking
    # ... more options
}
```

#### What Happens During Training

1. **Pose Extraction**: MediaPipe extracts landmarks from all frames
2. **Sequence Creation**: Sliding windows create 30-frame sequences
3. **Label Assignment**: Majority voting assigns labels to windows
4. **Model Training**: LSTM network trains with early stopping
5. **Evaluation**: Model tested on held-out test set
6. **Artifact Saving**: Model, plots, and metrics saved to `output/`

#### Expected Output

```
Total sequences: 1234
  backhand: 456 (37.0%)
  fronthand: 778 (63.0%)

Training split: 987 train / 247 test

Epoch 1/100
loss: 0.6234 - accuracy: 0.7100 - val_loss: 0.5321 - val_accuracy: 0.7850

...

Test accuracy: 85.43%

Classification Report:
              precision    recall  f1-score   support
    backhand       0.83      0.89      0.86       112
   fronthand       0.88      0.82      0.85       135
```

#### Important: Neutral Class Removed

The system **does not use a "neutral" class**. Only annotated stroke segments are used for training. This prevents the model from being overwhelmed by non-stroke frames.

See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for the reasoning behind this decision.

### 4. Analyze New Videos

Once trained, analyze continuous tennis videos:

```bash
poetry run python src/detect_strokes.py path/to/video.mp4
```

#### Advanced Options

```bash
# Custom output directory
poetry run python src/detect_strokes.py video.mp4 --output-dir ./my_results

# Adjust confidence threshold (0-1, lower = more detections)
poetry run python src/detect_strokes.py video.mp4 --confidence 0.6

# Adjust detection stride (lower = more precise but slower)
poetry run python src/detect_strokes.py video.mp4 --stride 3
```

#### Detection Configuration

Edit `CONFIG` in [detect_strokes.py](src/detect_strokes.py):

```python
CONFIG = {
    'confidence_threshold': 0.7,     # Minimum confidence to accept
    'stride': 5,                     # Frames between predictions
    'min_stroke_duration': 13,       # Minimum frames for valid stroke
    'merge_nearby_strokes': 15,      # Merge strokes within N frames
}
```

**Tuning tips:**
- Lower `confidence_threshold`: More detections (may include false positives)
- Lower `stride`: More precise timing (but slower)
- Higher `min_stroke_duration`: Filter out brief false positives
- Higher `merge_nearby_strokes`: Combine fragmented detections

## Configuration

### Training Configuration (train_model.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 30 | Frames per sequence (must be consistent for detection) |
| `overlap` | 15 | Frame overlap between training windows |
| `batch_size` | 32 | Training batch size |
| `epochs` | 100 | Maximum training epochs (early stopping may end sooner) |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `use_mlflow` | True | Enable MLflow experiment tracking |
| `use_gpu` | True | Use GPU if available |
| `mixed_precision` | True | Enable mixed precision training (faster on GPU) |

### Detection Configuration (detect_strokes.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 30 | Must match training window size |
| `stride` | 5 | Frames to skip between predictions |
| `confidence_threshold` | 0.7 | Minimum prediction confidence (0-1) |
| `min_stroke_duration` | 13 | Minimum frames for valid stroke |
| `merge_nearby_strokes` | 15 | Merge strokes within N frames |

## Outputs

### Training Outputs (`output/`)

- **tennis_stroke_model.keras**: Trained LSTM model
- **label_classes.npy**: Class label mapping
- **training_history.png**: Training/validation curves
- **confusion_matrix.txt**: Detailed classification report

### Analysis Outputs (`analysis_output/`)

- **{video}_timeline.png**: Visual timeline with color-coded strokes
- **{video}_report.txt**: Detailed text report with timestamps
- **{video}_strokes.json**: Machine-readable stroke data

### MLflow Outputs (`mlruns/`)

MLflow tracks:
- **Parameters**: window_size, overlap, batch_size, learning_rate, etc.
- **Metrics**: accuracy, loss, per-class counts
- **Artifacts**: model, plots, confusion matrix

## MLflow Experiment Tracking

### Start MLflow UI

```bash
poetry run mlflow ui
```

Then open http://localhost:5000 in your browser.

### Features

- Compare multiple training runs side-by-side
- Visualize metrics over time
- Filter runs by parameters
- Download artifacts from any run
- Load previous models for inference

See [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md) for complete documentation.

## Troubleshooting

### GPU Not Detected

**Check GPU availability:**
```bash
poetry run python src/check_gpu.py
```

**If you don't have a GPU:**
- Training will use CPU (slower but still works)
- Set `use_gpu: False` in CONFIG to suppress warnings

### No Strokes Detected

If `detect_strokes.py` finds no strokes:

1. **Lower confidence threshold**:
   ```bash
   poetry run python src/detect_strokes.py video.mp4 --confidence 0.5
   ```

2. **Check video contains trained stroke types**:
   - Model only recognizes strokes it was trained on
   - Check `label_classes.npy` for available classes

3. **Verify pose detection**:
   - Ensure player is clearly visible
   - MediaPipe works best with full-body shots

### Low Training Accuracy

If model accuracy is low (<70%):

1. **Check annotation quality**: Use `verify_annotation.py`
2. **Increase training data**: Annotate more videos
3. **Adjust window size**: Try 45 or 60 frames
4. **Check class balance**: Ensure enough samples per class

### Dependency Issues

**Protobuf conflict (MediaPipe vs TensorFlow):**
- Use TensorFlow 2.17-2.18 (compatible with MediaPipe's protobuf <5)
- Already configured in `pyproject.toml`

**If poetry install fails:**
```bash
poetry lock --no-update
poetry install
```

## Documentation

- **[README.md](README.md)**: This file - overview and usage
- **[MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)**: MLflow experiment tracking
- **[DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md)**: Design decisions and lessons learned
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Detailed step-by-step examples

## License

This project is for educational and research purposes.

## Acknowledgments

- MediaPipe for pose estimation
- TensorFlow for deep learning framework
- Label Studio for video annotation
- MLflow for experiment tracking
