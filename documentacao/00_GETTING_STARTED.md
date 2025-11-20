# Getting Started - Tennis Stroke Recognition System

**Quick start guide for new users**

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start Workflow](#quick-start-workflow)
- [First Training Run](#first-training-run)
- [First Video Analysis](#first-video-analysis)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## Overview

This system recognizes tennis strokes (forehand, backhand, serve, slices) using:
- **MediaPipe Pose**: Extracts body keypoints from video
- **LSTM Neural Network**: Learns temporal patterns in strokes
- **MLflow**: Tracks and compares experiments

**Complete pipeline:**
1. Annotate videos in Label Studio
2. Train LSTM model on pose data
3. Detect strokes in continuous videos
4. Track experiments with MLflow

---

## System Requirements

### Required
- **Python 3.11+**
- **Poetry** (dependency manager)
- **4GB+ RAM**
- **Label Studio** (for annotation)

### Optional
- **CUDA-capable GPU** (10x faster training)
- **10GB+ disk space** (for videos and models)

---

## Installation

### Step 1: Install Poetry

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Linux/Mac:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Step 2: Install Dependencies

```bash
cd video_tennis_analysis
poetry install
```

**This installs:**
- `mediapipe` - Pose estimation
- `tensorflow` - Deep learning
- `opencv-python` - Video processing
- `mlflow` - Experiment tracking
- `scikit-learn` - Data preprocessing

### Step 3: Verify Installation

```bash
poetry run python -c "import tensorflow as tf; import mediapipe as mp; import mlflow; print('✅ All imports successful!')"
```

### Step 4: Check GPU (Optional)

```bash
poetry run python src/check_gpu.py
```

**Expected output:**
```
✅ Found 1 GPU(s):
  GPU 0: /physical_device:GPU:0
✅ GPU memory growth enabled
✅ TensorFlow will use GPU for training
```

**No GPU?** No problem! Training works on CPU (just slower).

---

## Quick Start Workflow

### Create Directory Structure

**Windows:**
```powershell
mkdir data\annotations
mkdir data\videos
mkdir output
mkdir analysis_output
mkdir pose_data
```

**Linux/Mac:**
```bash
mkdir -p data/annotations data/videos output analysis_output pose_data
```

**Your structure:**
```
video_tennis_analysis/
├── data/
│   ├── annotations/    ← Label Studio JSON files
│   └── videos/         ← Training videos
├── output/             ← Training outputs (model, plots)
├── analysis_output/    ← Detection results
├── pose_data/          ← Cached pose data (for fast training)
└── mlruns/             ← MLflow experiment data
```

---

## First Training Run

### Step 0: Prepare Videos (Optional but Recommended)

**If you have dark videos or small player:**

```bash
# Convert to 720p and brighten
poetry run python src/preprocess_video.py input.MP4 output.MP4 --auto-brighten --static-zoom 1.5

# Batch process all videos
poetry run python src/batch_preprocess.py D:\videos\raw D:\videos\preprocessed --auto-brighten --static-zoom 1.5
```

**What this does:**
- Converts to 720p (faster pose extraction)
- Auto-brightens dark videos
- Applies 1.5x zoom to make player larger
- Corrects fisheye distortion (if needed)

**See:** [07_VIDEO_PREPROCESSING.md](07_VIDEO_PREPROCESSING.md)

### Step 1: Annotate Videos in Label Studio

1. **Install Label Studio:**
   ```bash
   pip install label-studio
   label-studio start
   ```

2. **Create project** at http://localhost:8080

3. **Upload videos** to `data/videos/`

4. **Annotate strokes** using timeline labels:
   - `fronthand` or `forehand`
   - `backhand`
   - `saque` or `serve`
   - `slice direita` (right slice)
   - `slice esquerda` (left slice)

5. **Export annotations** to `data/annotations/` as JSON

**Tip:** Annotate at least 10-20 instances per stroke type for good results.

### Step 2: Verify Annotations

```bash
poetry run python src/verify_annotation.py
```

**Output:** `annotation_analysis_report.txt`

**Check:**
- All videos found?
- Annotation count per class?
- Any missing files?

**Example output:**
```
Total annotations: 145
  fronthand: 67 (46.2%)
  backhand: 58 (40.0%)
  saque: 12 (8.3%)
  slice direita: 5 (3.4%)
  slice esquerda: 3 (2.1%)

⚠️ Warning: Classes with <10 annotations may not train well
```

### Step 3: Extract Poses (One-Time Setup)

**Extract and save poses to disk (MUCH FASTER for multiple training runs):**

```bash
poetry run python src/extract_poses.py
```

**What this does:**
- Extracts MediaPipe poses from all annotated videos
- Saves to `pose_data/*.npz` files
- Takes 30-60 minutes initially (but only once!)

**Expected output:**
```
Processing: video_001.MP4
  Extracting poses... 100%
  Saved to: pose_data\video_001_poses.npz

Total processed: 15 videos
Total time: 42 minutes
```

**See:** [03_DATA_CONFIGURATION.md](03_DATA_CONFIGURATION.md#pose-extraction-and-caching)

### Step 4: Train the Model

**Now training is FAST (loads from disk):**

```bash
poetry run python src/train_model.py
```

**What happens:**
1. **Load poses** from disk (~10 seconds instead of 30 minutes!)
2. **Create sequences** with sliding windows
3. **Train LSTM** network (5-15 minutes depending on GPU)
4. **Evaluate** on test set
5. **Save** model and metrics to `output/`

**Expected output:**
```
Loading saved poses from: .\pose_data
Loaded 15 videos with 12,543 frames

Total sequences: 1,234
  fronthand: 612 (49.6%)
  backhand: 622 (50.4%)

Training split: 987 train / 247 test

Epoch 1/150
loss: 0.6234 - accuracy: 0.7100 - val_loss: 0.5321 - val_accuracy: 0.7850
...
Epoch 47/150 (Early stopping)

Test accuracy: 85.43%

Classification Report:
              precision    recall  f1-score   support
  fronthand       0.88      0.82      0.85       135
   backhand       0.83      0.89      0.86       112

Model saved to: output/tennis_stroke_model.keras
```

**Training time:**
- **With GPU**: 5-10 minutes
- **Without GPU**: 20-40 minutes

---

## First Video Analysis

### Detect Strokes in New Video

```bash
poetry run python src/detect_strokes.py path/to/your/video.mp4
```

**Output files** (in `analysis_output/`):
- `video_timeline.png` - Visual timeline with color-coded strokes
- `video_report.txt` - Detailed report with timestamps
- `video_strokes.json` - Machine-readable stroke data

**Example report:**
```
Tennis Stroke Analysis Report
=============================

Video: match_001.mp4
Duration: 5:23 (323 seconds)
Detected strokes: 47

Stroke Timeline:
  00:12.3 - 00:13.8 | Fronthand (conf: 0.92)
  00:15.1 - 00:16.4 | Backhand (conf: 0.87)
  00:19.7 - 00:21.0 | Fronthand (conf: 0.91)
  ...

Statistics:
  Fronthand: 24 (51.1%)
  Backhand: 23 (48.9%)
```

### Adjust Detection Sensitivity

**Too few detections?** Lower confidence:
```bash
poetry run python src/detect_strokes.py video.mp4 --confidence 0.5
```

**Too many false positives?** Raise confidence:
```bash
poetry run python src/detect_strokes.py video.mp4 --confidence 0.8
```

**More options:**
```bash
# Custom output directory
poetry run python src/detect_strokes.py video.mp4 --output-dir ./my_results

# Adjust stride (lower = more precise but slower)
poetry run python src/detect_strokes.py video.mp4 --stride 3

# Disable FPS scaling
poetry run python src/detect_strokes.py video.mp4 --disable-fps-scaling
```

---

## Troubleshooting

### No Strokes Detected

**Problem:** Detection finds 0 strokes

**Solutions:**

1. **Lower confidence threshold:**
   ```bash
   poetry run python src/detect_strokes.py video.mp4 --confidence 0.3
   ```

2. **Check pose detection works:**
   ```bash
   poetry run python src/visualize_pose.py video.mp4 --max-frames 300
   ```
   - Detection rate should be >80%
   - If <50%, video quality may be poor

3. **Verify model trained on correct strokes:**
   - Check `output/label_classes.npy`
   - Model only detects strokes it was trained on

### Low Training Accuracy (<70%)

**Problem:** Model accuracy is too low

**Solutions:**

1. **Check annotation quality:**
   ```bash
   poetry run python src/verify_annotation.py
   ```

2. **Increase window size** (more context):
   Edit `src/train_model.py`:
   ```python
   CONFIG = {
       'window_size': 60,  # Changed from 45
       'overlap': 30,      # window_size / 2
   }
   ```

3. **Group rare classes:**
   ```python
   CONFIG = {
       'group_classes': True,  # Merge slices into main strokes
   }
   ```

4. **Collect more training data:**
   - Annotate more videos
   - Aim for 20+ instances per stroke type

### GPU Not Detected

**Problem:** GPU available but not used

**Check GPU:**
```bash
poetry run python src/check_gpu.py
```

**If GPU not found:**
- Verify CUDA and cuDNN installed
- Reinstall TensorFlow with GPU support:
  ```bash
  poetry add tensorflow[and-cuda]
  ```

**Or disable GPU warnings:**
Edit `src/train_model.py`:
```python
CONFIG = {
    'use_mixed_precision': False,
}
```

### Pose Extraction is Very Slow

**Problem:** Extract poses takes hours

**Solutions:**

1. **Use 720p videos** (2x faster):
   ```bash
   poetry run python src/batch_preprocess.py D:\videos\1080p D:\videos\720p
   ```

2. **Extract once, train many times:**
   - Already using `extract_poses.py`? ✅ You're set!
   - Training loads from disk (seconds instead of hours)

3. **Use saved poses:**
   Edit `src/train_model.py`:
   ```python
   CONFIG = {
       'use_saved_poses': True,  # Load from disk (FAST)
       'pose_data_dir': r'.\pose_data',
   }
   ```

### Dependency Issues

**Problem:** `poetry install` fails

**Solution:**
```bash
poetry lock --no-update
poetry install
```

**Protobuf conflict:**
- Already handled in `pyproject.toml`
- TensorFlow 2.17-2.18 compatible with MediaPipe

---

## Next Steps

### Compare Experiments with MLflow

```bash
poetry run mlflow ui
```

Open http://localhost:5000

**What you can do:**
- Compare accuracy across runs
- Filter by parameters (window_size, learning_rate, etc.)
- Download best model
- View confusion matrices

**See:** [06_MLFLOW_TRACKING.md](06_MLFLOW_TRACKING.md)

### Optimize Your Model

Try different configurations:

```bash
# Test multiple settings automatically
poetry run python src/grid_search.py --grid minimal
```

**What to tune:**
- `window_size` (30, 45, 60 frames)
- `learning_rate` (0.001, 0.0005, 0.0001)
- `batch_size` (16, 32, 64)
- `use_bidirectional` (False, True)

**See:** [04_MODEL_OPTIMIZATION.md](04_MODEL_OPTIMIZATION.md) and [05_GRID_SEARCH.md](05_GRID_SEARCH.md)

### Understand the Architecture

**How does it work?**
- Read [01_ARCHITECTURE_AND_THEORY.md](01_ARCHITECTURE_AND_THEORY.md)
- Explains pose features, LSTM architecture, design decisions

### Advanced Usage

**Complete workflows:**
- [02_USAGE_GUIDE.md](02_USAGE_GUIDE.md) - Detailed step-by-step examples

**Data configuration:**
- [03_DATA_CONFIGURATION.md](03_DATA_CONFIGURATION.md) - Sequences, FPS scaling, alignment

---

## Quick Reference Commands

```bash
# Extract poses (once)
poetry run python src/extract_poses.py

# Train model
poetry run python src/train_model.py

# Detect strokes
poetry run python src/detect_strokes.py video.mp4

# MLflow UI
poetry run mlflow ui

# Grid search
poetry run python src/grid_search.py --grid minimal

# Verify annotations
poetry run python src/verify_annotation.py

# Check GPU
poetry run python src/check_gpu.py

# Visualize poses
poetry run python src/visualize_pose.py video.mp4 --max-frames 300

# Preprocess videos
poetry run python src/preprocess_video.py input.mp4 output.mp4 --auto-brighten --static-zoom 1.5
```

---

## Expected Timeline

| Task | Time (First Run) | Time (Subsequent) |
|------|------------------|-------------------|
| Install dependencies | 5-10 min | - |
| Annotate 10 videos | 2-3 hours | - |
| Extract poses | 30-60 min | - (saved to disk) |
| Train model | 5-15 min | 5-15 min |
| Detect strokes | 2-5 min | 2-5 min |
| **Total to first results** | **3-4 hours** | **10-20 min** |

**Key insight:** Extract poses once, experiment many times!

---

## Success Checklist

- [ ] Dependencies installed (`poetry install`)
- [ ] GPU detected (optional: `src/check_gpu.py`)
- [ ] Videos annotated in Label Studio
- [ ] Annotations verified (`verify_annotation.py`)
- [ ] Poses extracted (`extract_poses.py`) ← Do this once!
- [ ] Model trained (`train_model.py`)
- [ ] Test accuracy >75%
- [ ] Strokes detected in new video (`detect_strokes.py`)
- [ ] MLflow UI explored (`mlflow ui`)

**All done?** You're ready to optimize! See [04_MODEL_OPTIMIZATION.md](04_MODEL_OPTIMIZATION.md)

---

## Getting Help

**Read the docs:**
- [README_DOCS.md](README_DOCS.md) - Documentation index
- [01_ARCHITECTURE_AND_THEORY.md](01_ARCHITECTURE_AND_THEORY.md) - How it works
- [02_USAGE_GUIDE.md](02_USAGE_GUIDE.md) - Detailed examples
- [08_TECHNICAL_REFERENCE.md](08_TECHNICAL_REFERENCE.md) - Quick reference

**Still stuck?**
- Check MLflow UI for experiment details
- Verify annotation quality
- Try preprocessing videos if pose detection is poor

---

**Ready to start?** Run `poetry install` and begin annotating! 🎾
