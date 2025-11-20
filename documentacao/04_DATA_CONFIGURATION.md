# Data Configuration and Processing Pipeline

A comprehensive guide to configuring and processing video data for the tennis stroke detection system, covering pose extraction, sequence creation, FPS scaling, and prediction alignment.

---

## Table of Contents

1. [Overview](#overview)
2. [Pose Extraction and Caching](#pose-extraction-and-caching)
3. [Sequence Creation with Sliding Windows](#sequence-creation-with-sliding-windows)
4. [FPS Scaling for Temporal Consistency](#fps-scaling-for-temporal-consistency)
5. [Window Alignment and Prediction Timing](#window-alignment-and-prediction-timing)
6. [Troubleshooting](#troubleshooting)
7. [See Also](#see-also)

---

## Overview

This pipeline enables fast training and detection by separating data preprocessing into distinct stages:

1. **Pose Extraction** - Extract MediaPipe pose landmarks from videos once and save to disk
2. **Sequence Creation** - Create training sequences from landmarks using sliding windows
3. **FPS Scaling** - Maintain temporal consistency across videos with different frame rates
4. **Prediction Alignment** - Ensure stroke predictions appear at the correct time during detection

**Key Benefits:**
- ✅ **Fast Training** - 10-100x faster by reusing extracted poses
- ✅ **Flexible Configuration** - Adjust window sizes and parameters without re-extracting
- ✅ **Multi-FPS Support** - Automatically handles videos at 30, 48, 60, or other FPS
- ✅ **Accurate Timing** - Predictions appear at the correct moment in the video

---

## Pose Extraction and Caching

Extract pose landmarks **once** and reuse them for **fast training** and **grid search**.

### The Problem

Extracting MediaPipe poses from videos is **very slow**. Doing it every time you train wastes hours:

- **Before**: Extract poses → Train model (~hours per run)
- **After**: Extract poses once → Train model many times (~minutes per run)

### Solution: 3-Step Process

#### Step 1: Extract Poses Once (Slow - Do Once)

Extract pose landmarks from all videos and save to disk:

```bash
poetry run python src/extract_poses.py
```

**What it does:**
- Processes all videos in `video_base_path`
- Extracts MediaPipe pose landmarks
- Saves to `./pose_data/*.npz` files (compressed)
- Creates metadata file

**Output:**
```
pose_data/
├── video1_poses.npz
├── video2_poses.npz
├── ...
└── extraction_metadata.json
```

**Time**: Slow (same as before), but **only once**!

#### Step 2: Train Fast (Use Saved Poses)

Now training loads from disk instead of extracting:

```bash
poetry run python src/train_model.py
```

**CONFIG** (`use_saved_poses=True` by default):
```python
CONFIG = {
    'use_saved_poses': True,  # ← FAST MODE
    'pose_data_dir': r'.\pose_data',  # Where .npz files are
    # ...
}
```

**Speed**: **10-100x faster** data loading!

#### Step 3: Grid Search (Super Fast)

Grid search also uses saved poses:

```bash
poetry run python src/grid_search.py --grid small
```

**Speed**: Each run is **much faster** because no pose extraction!

### Commands

#### Extract Poses

```bash
# Extract all poses with default settings
poetry run python src/extract_poses.py

# Custom directories
poetry run python src/extract_poses.py \
    --video-dir "D:\Videos" \
    --annotations-dir "D:\Annotations" \
    --output-dir "./my_poses"

# Force re-extract (overwrite existing)
poetry run python src/extract_poses.py --force

# Verify existing pose data
poetry run python src/extract_poses.py --verify
```

#### Train with Saved Poses

```python
# In train_model.py CONFIG:
'use_saved_poses': True,   # Load from disk (FAST)
'pose_data_dir': r'.\pose_data',
```

```bash
poetry run python src/train_model.py
```

#### Train WITHOUT Saved Poses (Slow Mode)

```python
# In train_model.py CONFIG:
'use_saved_poses': False,  # Extract from videos (SLOW)
```

```bash
poetry run python src/train_model.py
```

### File Format

Pose data saved as `.npz` (compressed NumPy arrays):

```python
# Each *_poses.npz file contains:
{
    'landmarks': np.array,      # Shape: (num_frames, 132)
    'fps': float,                # Video FPS
    'video_filename': str,       # Original video filename
    'annotations': list,         # Label Studio annotations
    'num_frames': int,           # Number of frames
    'extracted_at': str,         # ISO timestamp
}
```

**Loading example:**
```python
data = np.load('pose_data/video1_poses.npz', allow_pickle=True)
landmarks = data['landmarks']  # (num_frames, 132)
fps = float(data['fps'])
```

### Workflow Examples

#### Initial Setup (Once)

```bash
# 1. Extract poses from all videos (slow, ~hours)
poetry run python src/extract_poses.py

# 2. Verify extraction worked
poetry run python src/extract_poses.py --verify

# 3. Train model (fast!)
poetry run python src/train_model.py
```

#### Daily Training Workflow

```bash
# Just train - poses already extracted!
poetry run python src/train_model.py

# Or run grid search
poetry run python src/grid_search.py --grid small
```

#### Adding New Videos

```bash
# 1. Add new videos to video_base_path
# 2. Add new annotations to label_studio_exports
# 3. Extract poses (only new ones)
poetry run python src/extract_poses.py  # Skips existing

# 4. Train with all data
poetry run python src/train_model.py
```

#### Re-extract Everything

```bash
# Force re-extraction (e.g., if you change MediaPipe settings)
poetry run python src/extract_poses.py --force
```

### Configuration

#### train_model.py

```python
CONFIG = {
    # Video paths (used when use_saved_poses=False)
    'video_base_path': r'D:\Videos',
    'label_studio_exports': r'D:\Annotations',

    # Pose extraction mode
    'use_saved_poses': True,  # True=fast, False=slow
    'pose_data_dir': r'.\pose_data',  # Where .npz files are saved

    # ... other settings
}
```

#### extract_poses.py CLI

```bash
# Default: uses train_model.CONFIG paths
poetry run python src/extract_poses.py

# Custom paths
poetry run python src/extract_poses.py \
    --video-dir "D:\MyVideos" \
    --annotations-dir "D:\MyAnnotations" \
    --output-dir "./custom_poses"
```

### Speed Comparison

| Task | Without Saved Poses | With Saved Poses | Speedup |
|------|---------------------|------------------|---------|
| **First extraction** | - | Slow (~hours) | - |
| **Each training run** | Slow (~hours) | **Fast (~minutes)** | **~10-100x** |
| **Grid search (20 runs)** | Days | Hours | **~20-50x** |

### Disk Space

- Each video: ~1-5 MB compressed (depends on length)
- 10 videos: ~10-50 MB total
- **Much smaller** than original videos!

### Benefits of Saved Poses

✅ **Consistency**: Same poses used for all experiments
✅ **Reproducibility**: Pose extraction settings saved in metadata
✅ **Flexibility**: Easy to try different model parameters without re-extracting
✅ **Grid Search**: Makes grid search actually practical

---

## Sequence Creation with Sliding Windows

This section explains **how training sequences are created** from your annotated videos and why you might get 0 sequences.

### How Sequence Creation Works

#### Step 1: Frame Labeling

**Code location:** src/train_model.py:248-269

First, the system creates a label for **every frame** in the video:

```
Video: 4596 frames total

Frame labels array:
  Frame 0-621:    'neutral' (no annotation)
  Frame 622-635:  'backhand' (your annotation)
  Frame 636-673:  'neutral' (gap between annotations)
  Frame 674-688:  'backhand' (your annotation)
  Frame 689-...:  'neutral'
  ...
```

#### Step 2: Sliding Window

**Code location:** src/train_model.py:271-296

Then, a sliding window moves through the video:

```
Window size: 40 frames
Overlap: 30 frames
Stride: 10 frames (stride = window_size - overlap = 40 - 30 = 10)

Window positions:
  Window 0: Frames 0-39
  Window 1: Frames 10-49   (overlaps 30 frames with Window 0)
  Window 2: Frames 20-59
  Window 3: Frames 30-69
  ...
```

**Visual representation:**
```
Frames:  [0────10────20────30────40────50────60────70────80]
Window 0: [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
Window 1:           [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
Window 2:                     [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
           ↑                   ↑
         Stride (10)       Overlap (30)
```

#### Step 3: Majority Voting

**Code location:** src/train_model.py:277-285

For each window, count which label appears most:

```python
# Example window covering frames 620-659 (40 frames)

window_labels = frame_labels[620:660]
# ['neutral', 'neutral', 'backhand', 'backhand', ..., 'neutral', 'neutral']

label_counts = {
    'neutral': 27,    # 27 frames labeled neutral
    'backhand': 13    # 13 frames labeled backhand
}

majority_label = 'neutral'  # max(label_counts) = neutral
majority_count = 27
```

#### Step 4: Filter Windows

**Code location:** src/train_model.py:287-292

Only keep windows where:
1. ✅ Majority label covers **>50% of frames** (>20 frames for window_size=40)
2. ✅ Majority label is **NOT 'neutral'** (we only want stroke sequences)

```python
if majority_count > window_size * 0.5 and majority_label != 'neutral':
    X.append(window)  # Keep this sequence
    y.append(majority_label)
else:
    # Reject this window
    pass
```

### Diagnosing Sequence Creation Issues

#### Why You Get 0 Sequences

**Your annotations vs Window Size:**

```
- backhand: frames 622-635 (13 frames)
- backhand: frames 674-688 (14 frames)
- backhand: frames 701-717 (16 frames)
...

Window size: 40 frames

Problem: Your annotations are too short!
```

**Example Calculation:**

Let's check if a window starting at frame 600 would be accepted:

```
Window: Frames 600-639 (40 frames)
Annotation: Frames 622-635 (13 frames of 'backhand')

Frame labels in window:
  600-621: 'neutral' (22 frames)
  622-635: 'backhand' (13 frames)
  636-639: 'neutral' (4 frames)

Label counts:
  'neutral': 26 frames (22 + 4)
  'backhand': 13 frames

Majority: 'neutral' (26 > 13)

Check 1: majority_count > 20?  → 26 > 20 ✅
Check 2: majority_label != 'neutral'?  → 'neutral' != 'neutral' ❌

Result: REJECTED
```

**Every window gets rejected** because:
- If window includes your 13-frame annotation → 'neutral' is still majority (27 vs 13)
- If window doesn't include annotation → 100% 'neutral' → rejected

#### How to Check if Window Size is Correct

**Run this check:**
```
Shortest annotation: 13 frames
Longest annotation: 16 frames
Window size: 40 frames

Is window_size <= shortest_annotation?
  NO (40 > 13)  ❌ PROBLEM!

Should be:
  window_size <= 13  ✅
  OR
  annotations extended to >= 40 frames  ✅
```

#### How to Verify Sequences Will Be Created

**Manual calculation:**
```
Annotation: frames 622-635 (13 frames labeled 'backhand')
Window size: 15 frames
Overlap: 10 frames

Window candidates:
  Window at 620: [620-634] → 12 backhand frames (80%) ✅
  Window at 625: [625-639] → 11 backhand frames (73%) ✅
  Window at 630: [630-644] → 6 backhand frames (40%) ❌

Expected sequences from this annotation: ~2 sequences
```

### Solutions

#### Option 1: Reduce Window Size (RECOMMENDED)

**Change window size to match your annotation length:**

```python
CONFIG = {
    'window_size': 15,  # Smaller than your shortest annotation (13 frames)
    'overlap': 10,      # 66% overlap for better coverage
}
```

**Why this works:**
```
Window: Frames 622-636 (15 frames)
Annotation: Frames 622-635 (13 frames of 'backhand')

Frame labels:
  622-635: 'backhand' (13 frames)
  636: 'neutral' (1 frame)

Label counts:
  'backhand': 13 frames
  'neutral': 2 frames

Majority: 'backhand' (13 > 2)

Check 1: 13 > 7.5 (50% of 15)?  ✅
Check 2: 'backhand' != 'neutral'?  ✅

Result: ACCEPTED ✅
```

**At 60 FPS:**
- 15 frames = 0.25 seconds
- Still captures key motion of stroke

#### Option 2: Extend Your Annotations

**Annotate more frames per stroke** to cover full motion:

```
Current:
  - backhand: frames 622-635 (13 frames = 0.22 seconds at 60 FPS)

Extended:
  - backhand: frames 610-660 (50 frames = 0.83 seconds at 60 FPS)
    ↑ Include preparation and follow-through
```

**Tennis stroke phases:**
```
[Preparation] [Backswing] [Forward swing] [Contact] [Follow-through]
    5 frames    8 frames     10 frames      5 frames    12 frames

    Total: ~40 frames = 0.67 seconds (matches window_size=40)
```

**How to extend annotations:**
1. Go back to Label Studio
2. For each stroke, drag the annotation earlier and later
3. Include:
   - Preparation phase (player getting ready)
   - Complete follow-through (arm fully extended)

#### Option 3: Lower Majority Threshold

**Modify the code to accept windows with <50% coverage:**

```python
# Current (line 290)
if majority_count > window_size * 0.5 and majority_label != 'neutral':

# Modified (accept >30% coverage)
if majority_count > window_size * 0.3 and majority_label != 'neutral':
```

**Risk:** You'll get noisier training data (windows with more neutral frames).

#### Option 4: Use Variable-Length Sequences

**More complex solution:** Modify the code to create sequences that match annotation length exactly:

```python
# Instead of fixed window_size, use annotation length
for anno in annotations:
    start = anno['start_frame']
    end = anno['end_frame']
    label = anno['label']

    sequence = landmarks[start:end]  # Variable length

    # Pad or truncate to fixed length for LSTM
    if len(sequence) < min_length:
        # Pad with zeros
        sequence = np.pad(sequence, ...)
    elif len(sequence) > max_length:
        # Truncate
        sequence = sequence[:max_length]
```

**Complexity:** Requires padding/truncating logic and may affect model performance.

### Recommended Action Plan

#### Quick Fix (5 minutes)

**Update CONFIG in train_model.py:**

```python
CONFIG = {
    'window_size': 15,  # Changed from 40
    'overlap': 10,      # Changed from 30
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 150,
    'group_classes': True,
    'use_mlflow': True,
}
```

**Expected result:**
```
Created 45 sequences  # Instead of 0!
Label distribution:
  backhand: 45
```

#### Better Solution (30 minutes)

1. **Re-annotate your videos in Label Studio**
2. **Extend each stroke annotation to ~40-50 frames:**
   - Start annotation when player begins preparation
   - End annotation after follow-through completes
3. **Keep window_size=40**

**Benefits:**
- Better temporal context for LSTM
- Model learns complete stroke motion (not just contact point)
- More robust to variations in stroke execution

### Understanding Window Size Selection

#### How to Choose Window Size

**At 60 FPS (your video):**
```
15 frames = 0.25 seconds (quick motion)
20 frames = 0.33 seconds (partial stroke)
30 frames = 0.50 seconds (most of stroke)
40 frames = 0.67 seconds (complete stroke)
60 frames = 1.00 seconds (stroke + recovery)
```

**At 30 FPS (standard video):**
```
15 frames = 0.50 seconds
20 frames = 0.67 seconds (typical choice)
30 frames = 1.00 seconds
```

**General guideline:**
- Window should capture **complete motion** (preparation → follow-through)
- Too small: Model only sees part of stroke (e.g., just contact)
- Too large: Model sees stroke + unrelated movements

**For tennis strokes:**
- **Groundstrokes (fronthand/backhand):** 30-40 frames at 60 FPS (0.5-0.67s)
- **Serves:** 50-60 frames at 60 FPS (0.83-1.0s, longer motion)
- **Volleys:** 20-30 frames at 60 FPS (0.33-0.5s, quicker motion)

---

## FPS Scaling for Temporal Consistency

Videos recorded at different frame rates (30 FPS, 48 FPS, 60 FPS) require different window sizes to capture the same temporal duration. This section explains automatic FPS scaling to maintain temporal consistency.

### The Problem

Without FPS scaling, a fixed window of 45 frames would represent:
- 1.5 seconds at 30 FPS ✓
- 0.75 seconds at 60 FPS ✗ (too short!)
- 0.94 seconds at 48 FPS ✗ (too short!)

### Solution: Automatic FPS Scaling

Both `train_model.py` and `detect_strokes.py` now automatically scale window parameters to maintain consistent **temporal duration** across different FPS videos.

#### Configuration

```python
CONFIG = {
    'reference_fps': 30,      # Reference FPS for calibration
    'window_size': 45,        # Frames at 30 FPS = 1.5 seconds
    'overlap': 15,            # Frames at 30 FPS = 0.5 seconds
    'MIN_ANNOTATION_LENGTH': 15,  # Frames at 30 FPS = 0.5 seconds
}
```

#### How It Works

When processing a video, the scripts:

1. **Detect video FPS** (e.g., 60 FPS)
2. **Calculate scale factor** = video_fps / reference_fps = 60 / 30 = 2.0x
3. **Scale all parameters:**
   - window_size: 45 × 2.0 = **90 frames** → 1.5 seconds at 60 FPS ✓
   - overlap: 15 × 2.0 = **30 frames** → 0.5 seconds at 60 FPS ✓
   - min_annotation_length: 15 × 2.0 = **30 frames** → 0.5 seconds at 60 FPS ✓

#### Examples

##### 30 FPS Video (Reference)
```
Reference: 30 FPS → Video: 30 FPS (scale factor: 1.00x)
Window: 45 → 45 frames (1.50s → 1.50s)
Overlap: 15 → 15 frames
```

##### 60 FPS Video
```
Reference: 30 FPS → Video: 60 FPS (scale factor: 2.00x)
Window: 45 → 90 frames (1.50s → 1.50s)
Overlap: 15 → 30 frames
```

##### 48 FPS Video
```
Reference: 30 FPS → Video: 48 FPS (scale factor: 1.60x)
Window: 45 → 72 frames (1.50s → 1.50s)
Overlap: 15 → 24 frames
```

### Benefits of FPS Scaling

✅ **Consistent temporal windows** across all videos regardless of FPS
✅ **No manual parameter adjustment** needed for different FPS
✅ **Same model works for all FPS** videos (maintains temporal patterns)
✅ **Automatic annotation expansion** scales proportionally

### Usage with FPS Scaling

#### Training with Mixed FPS Videos

Simply add your videos to the dataset - the training script automatically scales parameters:

```bash
python src/train_model.py
```

Output will show:
```
Processing: video_30fps.mp4
📐 Using reference FPS parameters (no scaling needed)
   Window: 45 frames, Overlap: 15 frames

Processing: video_60fps.mp4
📐 FPS Scaling:
   Reference: 30 FPS → Video: 60 FPS (scale factor: 2.00x)
   Window: 45 → 90 frames (1.50s → 1.50s)
   Overlap: 15 → 30 frames
```

#### Detection on Any FPS Video

The detection script automatically scales to match the video's FPS:

```bash
python src/detect_strokes.py video_48fps.mp4
```

Output:
```
📐 FPS Scaling:
   Reference: 30 FPS → Video: 48 FPS (scale factor: 1.60x)
   Window: 45 → 72 frames (1.50s → 1.50s)
   Overlap: 15 → 24 frames
```

### Important Notes

⚠️ **reference_fps must match between training and detection**
⚠️ **Model input shape is determined by the FIRST video processed during training**
⚠️ If your first training video is 60 FPS, all subsequent videos will be scaled to match that temporal window size

### Recommendations

1. **Use 30 FPS as reference** - most common, easier to reason about
2. **Calibrate window_size at reference FPS** - use `analyze_annotations.py` on 30 FPS videos
3. **Mix FPS videos freely** - the scaling handles it automatically
4. **Check console output** - verify scaling is working as expected

### Disabling FPS Scaling

By default, FPS scaling is **enabled**. You can disable it when it doesn't work well for your use case.

#### When to Disable FPS Scaling

**FPS Scaling May Not Work Well When:**

1. **Training data has mixed FPS without consistent temporal patterns**
   - Videos at 30fps, 48fps, and 60fps with different stroke speeds
   - Scaling assumptions break down

2. **Model was trained with FPS scaling disabled**
   - If you trained with fixed frame counts, detection must also use fixed counts
   - **Critical**: Training and detection settings must match!

3. **You want frame-based consistency instead of time-based**
   - Prefer same number of frames regardless of temporal duration
   - Useful for debugging or specific analysis

4. **Stroke durations vary significantly across different FPS videos**
   - Scaling may over/under-estimate window sizes
   - Fixed counts might work better empirically

#### How to Disable FPS Scaling

##### Method 1: Edit CONFIG (Persistent)

**In `train_model.py`:**

```python
CONFIG = {
    # ... other settings ...

    'enable_fps_scaling': False,  # ← Change from True to False
    'reference_fps': 30,
    'window_size': 45,  # Now treated as fixed frame count
    'overlap': 15,      # Now treated as fixed frame count
    'MIN_ANNOTATION_LENGTH': 15,  # Now treated as fixed frame count
}
```

**Effect**: All videos (30fps, 48fps, 60fps) will use exactly 45 frames for window size.

**In `detect_strokes.py`:**

```python
CONFIG = {
    # ... other settings ...

    'enable_fps_scaling': False,  # ← Change from True to False
    'reference_fps': 30,
    'window_size': 45,  # Now treated as fixed frame count
    'overlap': 15,      # Now treated as fixed frame count
}
```

**Effect**: Detection will use exactly 45 frames regardless of video FPS.

##### Method 2: CLI Argument (For Detection Only)

```bash
poetry run python src/detect_strokes.py path/to/video.mp4 --disable-fps-scaling
```

**Effect**: Temporarily disables FPS scaling for this detection run only.

#### Behavior Comparison

##### Example: Video @ 60fps with window_size=45, overlap=15

| FPS Scaling | Window Frames | Window Duration | Overlap Frames | Overlap Duration |
|-------------|---------------|-----------------|----------------|------------------|
| **Enabled** | 90 frames | 1.5 seconds | 30 frames | 0.5 seconds |
| **Disabled** | 45 frames | 0.75 seconds | 15 frames | 0.25 seconds |

##### Example: Video @ 30fps with window_size=45, overlap=15

| FPS Scaling | Window Frames | Window Duration | Overlap Frames | Overlap Duration |
|-------------|---------------|-----------------|----------------|------------------|
| **Enabled** | 45 frames | 1.5 seconds | 15 frames | 0.5 seconds |
| **Disabled** | 45 frames | 1.5 seconds | 15 frames | 0.5 seconds |

**Key Point**: At reference FPS (30), both modes behave identically.

#### Console Output Differences

##### With FPS Scaling Enabled (Default)

```
📐 FPS Scaling:
   Reference: 30 FPS → Video: 60 FPS (scale factor: 2.00x)
   Window: 45 → 90 frames (1.50s → 1.50s)
   Overlap: 15 → 30 frames (0.50s → 0.50s)
   Min annotation: 15 → 30 frames (0.50s → 0.50s)
```

##### With FPS Scaling Disabled

```
📐 FPS Scaling: DISABLED (using fixed frame counts)
   Video FPS: 60, Window: 45 frames, Overlap: 15 frames
```

#### Critical: Training and Detection MUST Match

If you disable FPS scaling in training, you **must** also disable it in detection:

```python
# train_model.py
CONFIG = {
    'enable_fps_scaling': False,  # Disabled during training
    'window_size': 45,
}
```

```python
# detect_strokes.py
CONFIG = {
    'enable_fps_scaling': False,  # MUST also be disabled for detection
    'window_size': 45,  # MUST match training
}
```

**Why?** The model learned patterns based on specific frame counts. Changing this during detection will produce incorrect predictions.

#### MLflow Logging

The `enable_fps_scaling` parameter is now logged to MLflow:

```
Parameters:
  enable_fps_scaling: True
  window_size: 45
  overlap: 15
  reference_fps: 30
```

**Use case**: When comparing runs, you can see which used FPS scaling and which didn't.

#### Decision Matrix

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| Training data all at same FPS | **Disable** | No need for scaling |
| Training data mixed FPS, consistent stroke durations | **Enable** | Maintains temporal consistency |
| Model performs worse with scaling | **Disable** | Trust empirical results |
| Already trained with scaling enabled | **Enable** for detection | Must match training |
| Already trained with scaling disabled | **Disable** for detection | Must match training |
| Debugging/testing | **Try both** | Compare in MLflow |

#### Testing Both Modes

##### Experiment Workflow

1. **Train with FPS scaling enabled:**
   ```python
   CONFIG['enable_fps_scaling'] = True
   ```
   Run training, note MLflow run ID

2. **Train with FPS scaling disabled:**
   ```python
   CONFIG['enable_fps_scaling'] = False
   ```
   Run training, note MLflow run ID

3. **Compare in MLflow UI:**
   ```bash
   mlflow ui
   ```
   - Select both runs
   - Compare `macro_avg_f1_score`
   - Check per-class recall balance
   - See which works better for your data

4. **Use best mode for detection:**
   - Set `enable_fps_scaling` in `detect_strokes.py` to match the better training run
   - Or use `--disable-fps-scaling` CLI flag if needed

#### Examples

##### Example 1: Disable for Training

```bash
# Edit train_model.py CONFIG
# Set enable_fps_scaling = False

poetry run python src/train_model.py
```

**Output:**
```
📐 FPS Scaling: DISABLED (using fixed frame counts)
   Video FPS: 60, Window: 45 frames, Overlap: 15 frames

Creating sequences from 2847 frames
Window size: 45, Overlap: 15
```

##### Example 2: Disable for Detection (CLI)

```bash
poetry run python src/detect_strokes.py video.mp4 --disable-fps-scaling
```

**Output:**
```
📐 FPS Scaling: DISABLED (using fixed frame counts)
   Video FPS: 60, Window: 45 frames, Overlap: 15 frames

Analyzing video with sliding window...
Window size: 45 frames, Overlap: 15 frames, Stride: 30 frames
```

##### Example 3: Keep Enabled (Default)

```bash
# No changes needed, default behavior
poetry run python src/train_model.py
```

**Output:**
```
📐 FPS Scaling:
   Reference: 30 FPS → Video: 60 FPS (scale factor: 2.00x)
   Window: 45 → 90 frames (1.50s → 1.50s)
   Overlap: 15 → 30 frames (0.50s → 0.50s)
```

---

## Window Alignment and Prediction Timing

When using sliding window predictions, you may notice that bounding boxes appear **too early** - sometimes 0.5-1.5 seconds before the actual stroke happens. This section explains why and provides solutions.

### The "Seeing the Future" Problem

Bounding boxes may appear **too early**, creating the illusion that the model is "predicting the future" rather than analyzing what's currently happening.

#### Why This Happens

##### 1. Window Position Bias

With a sliding window approach, each prediction is based on a window of frames (e.g., 45 frames = 1.5 seconds). The question is: **which frame should receive the prediction label?**

**Original behavior (before fix):**
```
Window: frames [0-45]  (0.0s - 1.5s)
Stroke happens: frames [35-45]  (1.17s - 1.5s)
Prediction assigned to: frame 0  (0.0s)
Result: Appears 1.17 seconds too early! ❌
```

**With center alignment (after fix):**
```
Window: frames [0-45]  (0.0s - 1.5s)
Stroke happens: frames [35-45]  (1.17s - 1.5s)
Prediction assigned to: frame 22  (0.73s)
Result: Much closer to actual timing! ✓
```

##### 2. Training Window Alignment

During training, majority voting assigns the stroke label to the **entire window** if the stroke covers >50% of frames. This means the model learns to recognize strokes even when seeing only the **beginning** of the movement.

##### 3. Preparatory Movement Recognition

The model may be learning to recognize **wind-up** and **stance changes** that happen before the actual racket-ball contact, causing early predictions.

### Solution: Configurable Prediction Alignment

The `detect_strokes.py` script supports three alignment modes to control when predictions are shown.

#### Configuration

```python
CONFIG = {
    'prediction_alignment': 'center',  # Options: 'start', 'center', 'end'
    # ... other settings
}
```

#### Alignment Modes

##### 1. `'start'` - Original Behavior (Earliest)

Assigns prediction to the **start** of the window.

**When to use:** Never recommended (causes "seeing the future" effect)

**Example:**
- Window: frames 0-45
- Prediction assigned: frames 0-45
- **Bias:** Up to 1.5 seconds too early

```
Window:     [════════════════════════════════════════════]
Prediction: [════════════════════════════════════════════]
Stroke:                                     [═════════]
                                            ↑
                                        Too early!
```

##### 2. `'center'` - Recommended (Most Accurate)

Centers prediction on the window, using a **quarter-window** (25%) on each side of center.

**When to use:** Default for most cases (balances early/late bias)

**Example:**
- Window: frames 0-45 (45 frames)
- Center: frame 22
- Quarter-window: 11 frames
- Prediction assigned: frames 11-33 (centered on 22)
- **Bias:** Minimal, typically within 0.25 seconds

```
Window:     [════════════════════════════════════════════]
Prediction:              [═════════════]
Stroke:                     [═════════]
                                ↑
                          Well aligned!
```

##### 3. `'end'` - Conservative (Latest)

Assigns prediction to the **second half** of the window.

**When to use:** If you want to ensure predictions never appear before the stroke starts

**Example:**
- Window: frames 0-45
- Prediction assigned: frames 22-45
- **Bias:** May appear slightly late, but never too early

```
Window:     [════════════════════════════════════════════]
Prediction:                      [══════════════════════]
Stroke:                                     [═════════]
                                            ↑
                                    Slightly late (safe)
```

### Usage

#### Basic Usage (Use Center Alignment)

The default configuration already uses center alignment:

```bash
python src/detect_strokes.py video.mp4
```

Output:
```
Analyzing video with sliding window...
Window size: 45 frames, Overlap: 15 frames, Stride: 30 frames
Prediction alignment: 'center' (to reduce 'seeing the future' effect)
```

#### Testing Different Alignments

Edit `CONFIG` in `detect_strokes.py`:

```python
# Test 1: Center alignment (recommended)
CONFIG['prediction_alignment'] = 'center'

# Test 2: End alignment (more conservative)
CONFIG['prediction_alignment'] = 'end'

# Test 3: Start alignment (original, not recommended)
CONFIG['prediction_alignment'] = 'start'
```

### Impact on Timing

Assuming window_size = 45 frames @ 30 FPS (1.5 seconds):

| Alignment | Offset from Window Start | Typical Timing Accuracy |
|-----------|-------------------------|------------------------|
| `'start'` | 0 frames (0.0s) | ❌ 0.5-1.5s too early |
| `'center'` | 22 frames (0.73s) | ✓ ±0.25s accurate |
| `'end'` | 22 frames (0.73s) | ✓ May be 0.25s late |

### Visual Comparison

#### Before Fix (Start Alignment)
```
Timeline: |----prep----|===STROKE===|----follow----|
Window:   [================================]
Box shown: ████████████████████████████████
Result:    Box appears during preparation phase ❌
```

#### After Fix (Center Alignment)
```
Timeline: |----prep----|===STROKE===|----follow----|
Window:   [================================]
Box shown:          █████████████
Result:    Box appears during stroke ✓
```

### Technical Details

#### How Center Alignment Works

```python
window_start = i  # e.g., frame 0
window_size = 45  # frames

# Calculate center point
window_center = i + window_size // 2  # frame 22

# Use quarter-window (25%) on each side
half_window = window_size // 4  # 11 frames

# Final prediction range
pred_start = window_center - half_window  # frame 11
pred_end = window_center + half_window    # frame 33

# Duration: 22 frames (0.73 seconds at 30fps)
```

#### Why Quarter-Window?

Using a smaller prediction range (25% of window on each side = 50% total) instead of the full window:
- ✓ Reduces prediction duration to more realistic stroke length
- ✓ Centers the prediction better on actual stroke
- ✓ Reduces overlap between consecutive predictions
- ✓ Makes merged predictions more accurate

### Recommendations

1. **Use `'center'` alignment** (default) - Best balance for most use cases
2. **Use `'end'` alignment** if false positives during preparation are problematic
3. **Never use `'start'` alignment** - only kept for backward compatibility

### Related Settings

Prediction alignment works together with other timing parameters:

```python
CONFIG = {
    'window_size': 45,           # Temporal window duration
    'overlap': 15,               # How much windows overlap
    'prediction_alignment': 'center',  # Where to place prediction
    'max_stroke_duration_seconds': 1.75,  # Max allowed stroke duration
    'merge_nearby_strokes': 15,  # Frames to merge consecutive predictions
}
```

All these work together to produce accurate stroke timing!

---

## Troubleshooting

### Pose Extraction Issues

#### "Pose data directory not found"
```
Run: poetry run python src/extract_poses.py
```

#### "No pose files found"
Check if extraction completed:
```bash
ls pose_data/
# Should see *_poses.npz files
```

#### Want to force slow mode (extract from videos)
```python
# In CONFIG:
'use_saved_poses': False,
```

#### Extracted poses but changed window_size
No problem! Poses are raw data. Just train again:
```bash
# Change window_size in CONFIG
poetry run python src/train_model.py
# Creates new sequences from saved poses
```

#### Changed MediaPipe settings
Re-extract with new settings:
```bash
# Edit PoseExtractor in train_model.py
poetry run python src/extract_poses.py --force
```

### Sequence Creation Issues

#### Created 0 sequences
See "Why You Get 0 Sequences" section in [Sequence Creation](#sequence-creation-with-sliding-windows).

**Quick fixes:**
1. Reduce window_size to match your annotation length
2. Extend your annotations to cover the full stroke
3. Lower the majority voting threshold

#### Too many sequences from one stroke
Your window size is too small. Increase it to capture more context.

#### Uneven label distribution
Check if certain strokes have shorter annotations. Consider:
- Extending short annotations
- Adjusting window size based on stroke duration

### FPS Scaling Issues

#### Model performance dropped after changing FPS scaling
**Solution:**
- Training and detection settings must match
- If you changed it in one script, change in the other too
- Re-train if you want to switch modes

#### Unsure which mode was used for training
**Solution:**
- Check MLflow for the training run
- Look at `enable_fps_scaling` parameter
- Use the same setting for detection

#### Different videos need different settings
**Solution:**
- Use CLI flag `--disable-fps-scaling` for specific videos
- Keep CONFIG with your most common setting
- Override as needed per video

### Prediction Timing Issues

#### Predictions Still Too Early
Try `'end'` alignment:
```python
CONFIG['prediction_alignment'] = 'end'
```

#### Predictions Too Late
You're probably already using `'end'`. Try `'center'`:
```python
CONFIG['prediction_alignment'] = 'center'
```

#### Predictions Very Accurate
Great! Stick with `'center'` (default).

---

## See Also

Related documentation files:
- **BIDIRECTIONAL_LSTM_GUIDE.md** - LSTM model architecture and training
- **MLFLOW_ENHANCED_LOGGING.md** - Experiment tracking and hyperparameter logging
- **GRID_SEARCH_GUIDE.md** - Automated hyperparameter tuning
- **USAGE_GUIDE.md** - General usage and command examples (if available)

Key code locations:
- **Frame labeling:** src/train_model.py:248-269
- **Sliding window:** src/train_model.py:271-296
- **Majority voting:** src/train_model.py:277-285
- **Filtering logic:** src/train_model.py:287-292
- **CONFIG:** src/train_model.py:72-86
- **Pose extraction:** src/extract_poses.py
- **Detection:** src/detect_strokes.py

---

## Quick Reference Summary

### Setup Workflow (Once)
```bash
# 1. Extract poses (slow, do once)
poetry run python src/extract_poses.py

# 2. Verify extraction
poetry run python src/extract_poses.py --verify

# 3. Train model (fast!)
poetry run python src/train_model.py
```

### Daily Workflow (Fast)
```bash
# Train with existing poses
poetry run python src/train_model.py

# Or run grid search
poetry run python src/grid_search.py --grid small

# Or detect strokes in video
poetry run python src/detect_strokes.py video.mp4
```

### Key Configuration Points

| Parameter | Purpose | Values |
|-----------|---------|--------|
| `use_saved_poses` | Fast vs slow training | True (fast) / False (slow) |
| `window_size` | Temporal window size | 15-60 frames |
| `overlap` | Window overlap | 10-30 frames |
| `enable_fps_scaling` | FPS consistency | True (default) / False |
| `reference_fps` | Reference frame rate | 30 (recommended) |
| `prediction_alignment` | Timing accuracy | 'center' (default) / 'end' / 'start' |

---

**Last Updated:** 2025-11-19
**Version:** 1.0 (Consolidated from 5 source documents)
