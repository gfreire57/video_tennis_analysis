# Usage Guide

This guide provides detailed, step-by-step examples for using the Tennis Stroke Recognition System.

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Workflow Overview](#workflow-overview)
- [Step 1: Annotate Videos](#step-1-annotate-videos)
- [Step 2: Verify Annotations](#step-2-verify-annotations)
- [Step 3: Train the Model](#step-3-train-the-model)
- [Step 4: Analyze New Videos](#step-4-analyze-new-videos)
- [Step 5: Compare Experiments](#step-5-compare-experiments)
- [Common Workflows](#common-workflows)
- [Tuning Guide](#tuning-guide)
- [FAQ](#faq)

## Installation & Setup

### First-Time Setup

1. **Install Poetry** (if not already installed):
   ```bash
   # Windows (PowerShell)
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

   # Linux/Mac
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone and navigate to project**:
   ```bash
   cd video_tennis_analysis
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Verify installation**:
   ```bash
   poetry run python -c "import tensorflow as tf; import mediapipe as mp; import mlflow; print('All imports successful!')"
   ```

5. **(Optional) Check GPU**:
   ```bash
   poetry run python src/check_gpu.py
   ```

### Project Structure Setup

Create the required directories:

```bash
# Windows
mkdir data\annotations
mkdir data\videos
mkdir output
mkdir analysis_output

# Linux/Mac
mkdir -p data/annotations data/videos output analysis_output
```

Your structure should look like:
```
video_tennis_analysis/
├── data/
│   ├── annotations/    ← Label Studio JSON files go here
│   └── videos/         ← Training videos go here
├── output/             ← Training outputs will be saved here
└── analysis_output/    ← Video analysis results will be saved here
```

## Workflow Overview

```
┌─────────────────────┐
│  1. Annotate Videos │ (Label Studio)
│     in Label Studio │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Verify          │ (verify_annotation.py)
│     Annotations     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Train Model     │ (train_model.py)
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Analyze New     │ (detect_strokes.py)
│     Videos          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Compare         │ (MLflow UI)
│     Experiments     │
└─────────────────────┘
```

## Step 1: Annotate Videos

### Setup Label Studio

1. **Install Label Studio**:
   ```bash
   pip install label-studio
   ```

2. **Start Label Studio**:
   ```bash
   label-studio start
   ```

3. **Open browser**: http://localhost:8080

4. **Create new project**:
   - Click "Create Project"
   - Name: "Tennis Stroke Recognition"
   - Choose "Video/Audio Classification"

### Configure Labeling Interface

Use this XML configuration:

```xml
<View>
  <Header value="Tennis Stroke Annotation"/>
  <Video name="video" value="$video"/>
  <TimelineLabels name="stroke" toName="video">
    <Label value="fronthand" background="#4ECDC4"/>
    <Label value="backhand" background="#FF6B6B"/>
    <Label value="saque" background="#FFE66D"/>
    <Label value="slice direita" background="#95E1D3"/>
    <Label value="slice esquerda" background="#A8E6CF"/>
  </TimelineLabels>
</View>
```

### Import Videos

1. **Copy videos to data folder**:
   ```bash
   # Example videos
   data/videos/match1.mp4
   data/videos/training_session.mp4
   data/videos/practice_rally.mp4
   ```

2. **In Label Studio**:
   - Settings → Cloud Storage → Add Source Storage
   - Storage Type: Local files
   - Absolute local path: `D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\data\videos`
   - File Filter Regex: `.*\.(mp4|avi|mov)$`
   - Click "Sync Storage"

### Annotate Strokes

1. **Open a video** from the task list

2. **Mark stroke segments**:
   - Play video and identify strokes
   - Select stroke label (e.g., "fronthand")
   - Drag on timeline to mark start/end of stroke
   - Repeat for all strokes in video

3. **Annotation Tips**:
   - Mark from backswing start to follow-through end
   - Include the full stroke motion
   - Don't worry about exact frame precision (±5 frames is fine)
   - Leave gaps between strokes unmarked (system will handle it)

4. **Submit** when done with video

### Export Annotations

1. **In Label Studio**:
   - Select all completed tasks
   - Export → JSON
   - Save as `task_export.json`

2. **Move to annotations folder**:
   ```bash
   move task_export.json data\annotations\
   ```

**Example annotation JSON structure**:
```json
{
  "task": {
    "data": {
      "video": "/data/local-files/?d=videos/match1.mp4"
    }
  },
  "result": [
    {
      "value": {
        "start": 2.5,
        "end": 3.8,
        "labels": ["fronthand"]
      },
      "type": "labels"
    },
    {
      "value": {
        "start": 5.2,
        "end": 6.5,
        "labels": ["backhand"]
      },
      "type": "labels"
    }
  ]
}
```

## Step 2: Verify Annotations

Before training, verify your annotations are correct.

### Basic Verification

```bash
poetry run python src/verify_annotation.py
```

**Output**: `verification_report.md` with:
- Number of annotated videos
- Label distribution
- Video file existence checks
- Potential issues

**Example output**:
```
======================================================================
ANNOTATION VERIFICATION REPORT
======================================================================

Checking annotations folder: data/annotations
Found 3 annotation files

Processing: task_export.json
  Video: match1.mp4
    Status: ✓ FOUND at data/videos/match1.mp4
    Labels found: 15 stroke(s)
      fronthand: 8
      backhand: 7

  Video: training_session.mp4
    Status: ✓ FOUND at data/videos/training_session.mp4
    Labels found: 23 stroke(s)
      fronthand: 12
      backhand: 9
      saque: 2

======================================================================
SUMMARY
======================================================================

Total videos: 2
Total labels: 38

Label distribution:
  fronthand: 20 (52.6%)
  backhand: 16 (42.1%)
  saque: 2 (5.3%)

⚠️ WARNING: 'saque' has very few samples (2). Consider annotating more.
```

### Reviewing the Report

**Check for**:
- ✅ All videos found
- ✅ Reasonable label distribution
- ⚠️ Classes with <20 samples (may need more data)
- ❌ Missing video files

**If videos not found**:
1. Check `video_base_path` in verify_annotation.py:
   ```python
   video_base_path = r'D:\path\to\your\data\videos'
   ```
2. Ensure videos are in the correct folder
3. Check file extensions match (mp4, avi, mov)

## Step 3: Train the Model

### Basic Training

```bash
poetry run python src/train_model.py
```

This runs with default configuration.

### Understanding the Output

**Phase 1: Setup & Data Loading**
```
======================================================================
TRAINING TENNIS STROKE CLASSIFIER
======================================================================

GPU Configuration:
  GPU Available: Yes
  Device: NVIDIA GeForce RTX 3080
  Mixed Precision: Enabled

Loading annotations from: data/annotations
Found 3 annotation files

Processing annotations...
  task_export.json: 2 videos, 38 stroke segments
```

**Phase 2: Pose Extraction**
```
Extracting pose landmarks from videos...

Video 1/2: match1.mp4
  Duration: 120s, FPS: 30.0, Frames: 3600
  Processed 3600/3600 frames (100.0%)

Video 2/2: training_session.mp4
  Duration: 180s, FPS: 30.0, Frames: 5400
  Processed 5400/5400 frames (100.0%)
```

**Phase 3: Sequence Creation**
```
Creating training sequences...
  Window size: 30 frames
  Overlap: 15 frames

Total sequences: 768
  fronthand: 420 (54.7%)
  backhand: 348 (45.3%)

Training split: 614 train / 154 test
```

**Phase 4: Model Training**
```
Building LSTM model...
  Input shape: (30, 132)
  LSTM layers: [128, 64, 32]
  Output classes: 2

Training model...

Epoch 1/100
loss: 0.6543 - accuracy: 0.6234 - val_loss: 0.5678 - val_accuracy: 0.7012
Epoch 2/100
loss: 0.5123 - accuracy: 0.7456 - val_loss: 0.4901 - val_accuracy: 0.7823
...
Epoch 23/100
loss: 0.2145 - accuracy: 0.9123 - val_loss: 0.3234 - val_accuracy: 0.8567

Early stopping triggered (no improvement for 10 epochs)
```

**Phase 5: Evaluation**
```
Evaluating on test set...
Test loss: 0.3456
Test accuracy: 84.42%

Classification Report:
              precision    recall  f1-score   support
   backhand       0.82      0.87      0.84        69
  fronthand       0.86      0.82      0.84        85

   accuracy                           0.84       154
  macro avg       0.84      0.84      0.84       154

Confusion Matrix:
              backhand  fronthand
   backhand        60          9
  fronthand        15         70
```

**Phase 6: Saving**
```
Saving outputs to: output/

Model saved: output/tennis_stroke_model.keras
Label classes saved: output/label_classes.npy
Training history plot saved: output/training_history.png

MLflow tracking:
  Experiment: tennis_stroke_recognition
  Run ID: 4a3b2c1d5e6f7g8h
  View at: http://localhost:5000
```

### Interpreting Results

**Good Results:**
- Test accuracy: >80%
- Precision/recall balanced for all classes (difference <10%)
- Confusion matrix: High diagonal values

**Warning Signs:**
- Test accuracy: <70% (may need more data or different parameters)
- One class much worse than others (need more samples of that class)
- Large gap between train and test accuracy (overfitting)

### Customizing Training

Edit `CONFIG` in [train_model.py](src/train_model.py):

```python
CONFIG = {
    # Data parameters
    'window_size': 30,              # Try 45 or 60 for longer context
    'overlap': 15,                  # Try 20 for more training samples

    # Model parameters
    'batch_size': 32,               # Try 16 if running out of memory
    'epochs': 100,                  # Maximum (early stopping will stop sooner)
    'learning_rate': 0.001,         # Try 0.0001 if not converging

    # Training options
    'use_gpu': True,                # Set False to force CPU
    'mixed_precision': True,        # Set False if GPU issues
    'use_mlflow': True,             # Set False to disable tracking

    # Paths
    'annotation_folder': r'D:\...\data\annotations',
    'video_base_path': r'D:\...\data\videos',
    'output_dir': './output',
}
```

**Example: Training with larger window**
```python
CONFIG = {
    'window_size': 45,  # Changed from 30
    'overlap': 22,      # Changed from 15 (keep ~50% overlap)
    # ... rest stays the same
}
```

Run training:
```bash
poetry run python src/train_model.py
```

MLflow will track this as a separate experiment for comparison.

## Step 4: Analyze New Videos

Once trained, analyze new videos to detect strokes.

### Basic Analysis

```bash
poetry run python src/detect_strokes.py path/to/video.mp4
```

**Example**:
```bash
poetry run python src/detect_strokes.py D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_15.MP4
```

### Output

**Console output**:
```
======================================================================
TENNIS VIDEO STROKE ANALYSIS
======================================================================

Loading trained model...
Model loaded. Classes: backhand, fronthand

Processing video: D:\Tennis\my_match.mp4
FPS: 30.00, Total frames: 9000

Extracting pose landmarks...
Processed 9000/9000 frames (100.0%)

Analyzing video with sliding window...
Window size: 30 frames, Stride: 5 frames
Analyzed 9000/9000 frames (100.0%)

Found 234 high-confidence predictions
Merged into 18 strokes

Detected 18 strokes in video:

 1. fronthand      | 0:00:05 - 0:00:06 | Duration: 1.2s | Confidence: 89%
 2. backhand       | 0:00:12 - 0:00:13 | Duration: 1.3s | Confidence: 92%
 3. fronthand      | 0:00:18 - 0:00:19 | Duration: 1.1s | Confidence: 87%
...
18. backhand       | 0:04:52 - 0:04:53 | Duration: 1.2s | Confidence: 91%

Stroke Frequency:
  fronthand: 11 (61.1%)
  backhand: 7 (38.9%)

Timeline saved to: analysis_output/my_match_timeline.png
Report saved to: analysis_output/my_match_report.txt
Stroke data saved to: analysis_output/my_match_strokes.json

======================================================================
ANALYSIS COMPLETE!
======================================================================
```

**Generated files**:

1. **`my_match_timeline.png`**: Visual timeline
   - Top panel: Color-coded stroke timeline
   - Bottom panel: Stroke frequency bar chart

2. **`my_match_report.txt`**: Detailed text report
   ```
   ======================================================================
   TENNIS STROKE ANALYSIS REPORT
   ======================================================================

   Video Duration: 0:05:00
   Total Strokes Detected: 18

   Stroke Frequency:
     fronthand: 11 (61.1%)
     backhand: 7 (38.9%)

   Detailed Stroke Timeline:
   ----------------------------------------------------------------------
     1. fronthand      | 0:00:05 - 0:00:06 | Duration: 1.20s | Confidence: 89.23%
     2. backhand       | 0:00:12 - 0:00:13 | Duration: 1.30s | Confidence: 92.45%
   ...
   ```

3. **`my_match_strokes.json`**: Machine-readable data
   ```json
   {
     "video_path": "D:\\Tennis\\my_match.mp4",
     "video_duration": 300.0,
     "fps": 30.0,
     "strokes": [
       {
         "class_name": "fronthand",
         "frame_start": 150,
         "frame_end": 186,
         "time_start": 5.0,
         "time_end": 6.2,
         "duration": 1.2,
         "avg_confidence": 0.8923
       },
       ...
     ]
   }
   ```

### Advanced Options

**Custom output directory**:
```bash
poetry run python src/detect_strokes.py video.mp4 --output-dir ./my_results
```

**Adjust confidence threshold**:
```bash
# Lower threshold = more detections (may include false positives)
poetry run python src/detect_strokes.py video.mp4 --confidence 0.6

# Higher threshold = fewer detections (only very confident ones)
poetry run python src/detect_strokes.py video.mp4 --confidence 0.85
```

**Adjust detection stride**:
```bash
# Smaller stride = more precise timing (but slower)
poetry run python src/detect_strokes.py video.mp4 --stride 3

# Larger stride = faster analysis (but may miss quick strokes)
poetry run python src/detect_strokes.py video.mp4 --stride 10
```

**Combine options**:
```bash
poetry run python src/detect_strokes.py video.mp4 \
  --output-dir ./results \
  --confidence 0.65 \
  --stride 3
```

### Tuning Detection Parameters

Edit `CONFIG` in [detect_strokes.py](src/detect_strokes.py):

```python
CONFIG = {
    'model_path': r'...\output\tennis_stroke_model.keras',
    'label_classes_path': r'...\output\label_classes.npy',

    'window_size': 30,               # Must match training!
    'stride': 5,                     # Frames between predictions
    'confidence_threshold': 0.7,     # Minimum confidence
    'min_stroke_duration': 13,       # Minimum frames for valid stroke
    'merge_nearby_strokes': 15,      # Merge strokes within N frames
}
```

**Parameter effects**:

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `stride` | More precise, slower | Faster, may miss strokes |
| `confidence_threshold` | More detections, more false positives | Fewer detections, fewer false positives |
| `min_stroke_duration` | Detect brief strokes | Filter out quick false positives |
| `merge_nearby_strokes` | Keep strokes separate | Merge fragmented detections |

## Step 5: Compare Experiments

Use MLflow to compare different training runs.

### Start MLflow UI

```bash
poetry run mlflow ui
```

Open browser: http://localhost:5000

### Compare Runs

1. **View all runs**:
   - Main page shows all training runs
   - Columns: parameters, metrics, start time, duration

2. **Select runs to compare**:
   - Check boxes for 2+ runs
   - Click "Compare" button

3. **Comparison view**:
   - **Parameters**: See what changed (window_size, learning_rate, etc.)
   - **Metrics**: Compare test_accuracy, test_loss
   - **Charts**: Visualize metric differences

4. **Download artifacts**:
   - Click a run → "Artifacts" tab
   - Download model, plots, confusion matrix

### Example Comparison Workflow

**Experiment 1: Baseline**
```python
# train_model.py
CONFIG = {
    'window_size': 30,
    'overlap': 15,
    'learning_rate': 0.001,
}
```
Run training → Note test accuracy: 82%

**Experiment 2: Larger window**
```python
CONFIG = {
    'window_size': 45,  # Increased
    'overlap': 22,
    'learning_rate': 0.001,
}
```
Run training → Note test accuracy: 85%

**Experiment 3: Lower learning rate**
```python
CONFIG = {
    'window_size': 45,
    'overlap': 22,
    'learning_rate': 0.0001,  # Decreased
}
```
Run training → Note test accuracy: 87%

**In MLflow**:
- Select all 3 runs
- Click "Compare"
- See that window_size=45 and learning_rate=0.0001 performed best
- Use this configuration going forward

## Common Workflows

### Workflow 1: Adding New Stroke Type

**Scenario**: You want to add "volley" to the existing model.

1. **Annotate new videos** with "volley" labels in Label Studio
2. **Export annotations** to `data/annotations/`
3. **Verify** annotations:
   ```bash
   poetry run python src/verify_annotation.py
   ```
4. **Retrain model**:
   ```bash
   poetry run python src/train_model.py
   ```
   Model will automatically include the new class.

5. **Test detection** on a video with volleys:
   ```bash
   poetry run python src/detect_strokes.py volley_video.mp4
   ```

### Workflow 2: Improving Accuracy for Specific Class

**Scenario**: Backhand detection is poor (precision: 65%).

1. **Check current data**:
   ```bash
   poetry run python src/verify_annotation.py
   ```
   Check backhand sample count.

2. **Annotate more backhand videos**:
   - Focus on different backhand types (topspin, flat, slice)
   - Add 20-30 more backhand annotations

3. **Verify** new annotations:
   ```bash
   poetry run python src/verify_annotation.py
   ```

4. **Retrain**:
   ```bash
   poetry run python src/train_model.py
   ```

5. **Compare in MLflow**:
   - Old run: backhand precision 65%
   - New run: backhand precision 78% (improved!)

### Workflow 3: Analyzing Multiple Videos in Batch

**Scenario**: Analyze 10 match videos.

Create a batch script:

**`analyze_batch.bat` (Windows)**:
```batch
@echo off
for %%f in (D:\Tennis\matches\*.mp4) do (
    echo Analyzing %%f
    poetry run python src/detect_strokes.py "%%f" --output-dir "./analysis_output"
)
echo All videos analyzed!
```

**`analyze_batch.sh` (Linux/Mac)**:
```bash
#!/bin/bash
for video in /path/to/matches/*.mp4; do
    echo "Analyzing $video"
    poetry run python src/detect_strokes.py "$video" --output-dir "./analysis_output"
done
echo "All videos analyzed!"
```

Run:
```bash
# Windows
analyze_batch.bat

# Linux/Mac
chmod +x analyze_batch.sh
./analyze_batch.sh
```

### Workflow 4: Exporting Results to Spreadsheet

**Scenario**: Create a CSV of all detected strokes.

Python script `export_to_csv.py`:
```python
import json
import csv
from pathlib import Path

# Load all JSON results
analysis_dir = Path('./analysis_output')
json_files = analysis_dir.glob('*_strokes.json')

# Prepare CSV
csv_file = 'all_strokes.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Video', 'Stroke Type', 'Start Time', 'End Time', 'Duration', 'Confidence'])

    for json_path in json_files:
        with open(json_path) as jf:
            data = json.load(jf)
            video_name = Path(data['video_path']).name

            for stroke in data['strokes']:
                writer.writerow([
                    video_name,
                    stroke['class_name'],
                    stroke['time_start'],
                    stroke['time_end'],
                    stroke['duration'],
                    stroke['avg_confidence']
                ])

print(f"Exported to {csv_file}")
```

Run:
```bash
poetry run python export_to_csv.py
```

Open `all_strokes.csv` in Excel/Google Sheets for further analysis.

## Tuning Guide

### If Detection Misses Strokes

**Problem**: Some obvious strokes not detected.

**Solutions**:

1. **Lower confidence threshold**:
   ```bash
   poetry run python src/detect_strokes.py video.mp4 --confidence 0.5
   ```

2. **Decrease stride** (more frequent checks):
   ```python
   # detect_strokes.py
   'stride': 3,  # Check every 3 frames instead of 5
   ```

3. **Check model was trained on that stroke type**:
   ```bash
   # View classes in model
   poetry run python -c "import numpy as np; print(np.load('output/label_classes.npy'))"
   ```

4. **Retrain with more examples** of that stroke.

### If Detection Has False Positives

**Problem**: Detecting strokes that aren't there.

**Solutions**:

1. **Raise confidence threshold**:
   ```bash
   poetry run python src/detect_strokes.py video.mp4 --confidence 0.8
   ```

2. **Increase minimum duration**:
   ```python
   # detect_strokes.py
   'min_stroke_duration': 20,  # Ignore brief detections
   ```

3. **Check if non-stroke movements are similar** to strokes:
   - May need to retrain with more diverse data
   - Consider annotating "negative examples" (future improvement)

### If Training Accuracy is Low

**Problem**: Test accuracy <70%.

**Solutions**:

1. **Check annotation quality**:
   ```bash
   poetry run python src/verify_annotation.py
   ```
   - Ensure enough samples per class (>50)
   - Check for mislabeled strokes

2. **Increase window size**:
   ```python
   # train_model.py
   'window_size': 45,  # More context
   ```

3. **Adjust learning rate**:
   ```python
   'learning_rate': 0.0001,  # Slower, more stable learning
   ```

4. **Add more training data**:
   - Annotate 10-20 more videos
   - Ensure variety (different players, angles, lighting)

5. **Check class balance**:
   - If one class has 200 samples and another has 20, add more of the minority class

### If Model Overfits

**Problem**: Train accuracy 95%, test accuracy 70%.

**Solutions**:

1. **Increase dropout** in model (edit train_model.py):
   ```python
   model.add(Dropout(0.5))  # Increase from 0.3
   ```

2. **Add more training data** (best solution)

3. **Reduce model complexity**:
   ```python
   LSTM(64, ...)  # Reduce from 128
   ```

4. **Increase overlap** (more augmentation):
   ```python
   'overlap': 20,  # More overlapping windows
   ```

## FAQ

### General Questions

**Q: How many videos do I need to annotate?**

A: Minimum 10-15 videos with 50+ strokes per class. More is better.

**Q: Can I use videos from different sources (phones, cameras, YouTube)?**

A: Yes! MediaPipe pose estimation works with any video. Variety helps generalization.

**Q: What video quality/FPS is required?**

A: Any modern video works. Recommendations:
- FPS: 24-60 fps (30 fps is ideal)
- Resolution: 720p or higher
- Format: MP4, AVI, MOV

**Q: Do I need a GPU?**

A: No, but it helps. Training on CPU takes 2-3x longer.

---

### Annotation Questions

**Q: How precise do annotations need to be?**

A: ±5 frames is fine. The system is robust to slight boundary inaccuracies.

**Q: Should I annotate non-stroke movements?**

A: No. Only annotate actual strokes. The system handles non-stroke frames automatically.

**Q: What if a stroke is partially off-screen?**

A: Annotate it anyway if the body is mostly visible. MediaPipe handles partial occlusions.

**Q: Can I have overlapping annotations?**

A: No. Each time segment should have only one label.

---

### Training Questions

**Q: How long does training take?**

A: Depends on data size and hardware:
- 500 sequences, CPU: 10-15 minutes
- 500 sequences, GPU: 3-5 minutes
- 2000 sequences, GPU: 10-15 minutes

**Q: Why did training stop early?**

A: Early stopping triggered. If validation loss doesn't improve for 10 epochs, training stops to prevent overfitting.

**Q: Can I resume training?**

A: Currently no. Each run trains from scratch. Use MLflow to track and compare runs.

**Q: How do I know if my model is good enough?**

A: Check:
- Test accuracy >80%: Good
- Precision/recall balanced across classes
- Real-world testing on new videos

---

### Detection Questions

**Q: Why are some strokes merged together?**

A: `merge_nearby_strokes` parameter. Reduce it to keep strokes separate:
```python
'merge_nearby_strokes': 5,  # Less aggressive merging
```

**Q: Can I detect strokes in real-time?**

A: Current implementation is offline (processes full video). Real-time requires optimization.

**Q: Why is detection slow?**

A: Pose extraction is the bottleneck. To speed up:
- Use GPU (if available)
- Increase stride (fewer predictions)
- Reduce video resolution before processing

**Q: Can I use a model trained with window_size=30 for detection with window_size=45?**

A: No. Detection window size MUST match training window size.

---

### MLflow Questions

**Q: Where is MLflow data stored?**

A: In `./mlruns` directory. Can grow large over time.

**Q: Can I delete old experiments?**

A: Yes. Delete specific run folders in `mlruns/0/` (backup first!).

**Q: Can I share MLflow results?**

A: Yes. Options:
1. Share entire `mlruns/` folder
2. Export run as ZIP from MLflow UI
3. Setup remote MLflow server (advanced)

**Q: MLflow UI won't start (port 5000 in use)?**

A: Use different port:
```bash
poetry run mlflow ui --port 5001
```

---

### Troubleshooting Questions

**Q: "ValueError: Cannot open video"?**

A: Check:
1. Video file exists at path
2. Path is absolute (not relative)
3. File is not corrupted
4. OpenCV supports the video codec

**Q: "No pose detected in frame"?**

A: MediaPipe couldn't find a person. Check:
- Person is clearly visible
- Not too far from camera
- Good lighting
- Try different frame (some frames may fail, that's okay)

**Q: "Out of memory" error during training?**

A: Reduce batch size:
```python
'batch_size': 16,  # Or even 8
```

**Q: Different results each training run?**

A: Normal. Neural network training has randomness. To reduce:
1. Set random seed (add to train_model.py)
2. Train multiple times and average
3. Use MLflow to track variance

---

## Next Steps

After mastering the basics:

1. **Read [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md)** to understand design decisions
2. **Experiment with parameters** using MLflow to track results
3. **Annotate more data** to improve accuracy
4. **Try advanced features** (ensemble, attention mechanisms)
5. **Contribute improvements** back to the project

Happy analyzing! 🎾
