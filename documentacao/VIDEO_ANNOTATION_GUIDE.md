# Video Annotation Feature

## Overview

The stroke detection script now generates **annotated videos** showing:
- ✅ Pose estimation skeleton overlay
- ✅ Bounding box around detected player
- ✅ Stroke classification labels (FRONTHAND, BACKHAND, etc.)
- ✅ Confidence scores for each detection
- ✅ Frame counter and timestamp

This helps you **visualize what the model is seeing** and verify detection accuracy.

---

## How It Works

When you run stroke detection, the script will:

1. **Detect strokes** in the video (as before)
2. **Generate timeline and reports** (as before)
3. **Create annotated video** showing:
   - Pose skeleton drawn on every frame
   - Bounding box around player when stroke is detected
   - Label above box: `FRONTHAND (85%)` or `BACKHAND (72%)`
   - Frame number and timestamp in top-left corner

---

## Usage

### Basic Usage (Annotation Enabled by Default)

```bash
poetry run python src/detect_strokes.py your_video.mp4
```

**Output files:**
- `analysis_output/your_video_timeline.png` - Timeline visualization
- `analysis_output/your_video_report.txt` - Text report
- `analysis_output/your_video_strokes.json` - JSON data
- **`analysis_output/your_video_annotated.mp4`** ← Annotated video!

### Disable Annotation (Faster)

```bash
poetry run python src/detect_strokes.py your_video.mp4 --no-visualize
```

Skips video generation if you only need the analysis data.

### Custom Output Directory

```bash
poetry run python src/detect_strokes.py video.mp4 --output-dir ./my_results
```

Saves all outputs (including annotated video) to `./my_results/`

---

## Annotated Video Features

### 1. Pose Skeleton Overlay

- **Always shown** on every frame (if pose detected)
- Green lines connecting body landmarks
- Shows what MediaPipe sees

### 2. Bounding Box + Label

- **Only shown during detected strokes**
- Box color indicates stroke type:
  - 🔴 **Red** = Backhand
  - 🔵 **Cyan** = Fronthand
  - 🟡 **Yellow** = Serve/Saque
- Label shows: `STROKE_NAME (CONFIDENCE%)`

### 3. Frame Info

- **Top-left corner shows:**
  - Frame number: `Frame: 150/4500`
  - Timestamp: `Time: 5.00s`

---

## Visual Example

```
┌─────────────────────────────────────────┐
│ Frame: 450/4500   Time: 15.00s         │
│                                         │
│                                         │
│      ┌────────────────────┐             │
│      │ FRONTHAND (87%)    │             │
│      └────────────────────┘             │
│      ┌─────────────────────────┐        │
│      │         o               │ ← Box  │
│      │        /|\  ← Skeleton  │        │
│      │       / | \             │        │
│      │         |               │        │
│      │        / \              │        │
│      │       /   \             │        │
│      └─────────────────────────┘        │
│                                         │
└─────────────────────────────────────────┘
```

---

## Configuration

### In Code (CONFIG)

Edit `src/detect_strokes.py`:

```python
CONFIG = {
    ...
    'visualize_video': True,  # Enable/disable annotation
}
```

### Via Command Line

```bash
# Default (annotation enabled)
poetry run python src/detect_strokes.py video.mp4

# Explicitly enable
poetry run python src/detect_strokes.py video.mp4 --visualize

# Disable
poetry run python src/detect_strokes.py video.mp4 --no-visualize
```

---

## Performance Impact

**Annotation adds processing time:**

| Video Length | Without Annotation | With Annotation | Difference |
|--------------|-------------------|-----------------|------------|
| 1 min        | ~30 sec           | ~45 sec         | +50%       |
| 5 min        | ~2.5 min          | ~4 min          | +60%       |
| 10 min       | ~5 min            | ~8 min          | +60%       |

**Why slower?**
- Must process every frame twice (once for detection, once for visualization)
- Drawing pose landmarks takes time
- Video encoding takes time

**When to disable:**
- Processing many videos
- Only need detection data (JSON/report)
- Don't need visual verification

---

## Viewing the Annotated Video

### Windows

Double-click the `.mp4` file or:
```powershell
start analysis_output/video_annotated.mp4
```

### Command Line

```bash
# Using VLC
vlc analysis_output/video_annotated.mp4

# Using default player
python -m webbrowser analysis_output/video_annotated.mp4
```

---

## What to Look For

### Good Detection

✅ Box appears during actual strokes
✅ Label matches stroke type (fronthand when you hit fronthand)
✅ Confidence is high (>70%)
✅ Box tracks player smoothly

### Poor Detection

❌ Box appears when no stroke happening → False positive
❌ Label wrong (says backhand but you hit fronthand) → Misclassification
❌ No box during obvious stroke → False negative (missed detection)
❌ Confidence very low (<50%) → Model uncertain

### If Results Are Poor

1. **Check pose detection:**
   - Is skeleton drawn correctly?
   - If skeleton missing/jittery → Video quality issue

2. **Retrain model** (see [MODEL_IMPROVEMENTS.md](MODEL_IMPROVEMENTS.md))
   - Class imbalance may need fixing
   - Model may need better architecture

3. **Adjust thresholds:**
   ```bash
   # Lower threshold to detect more (but more false positives)
   poetry run python src/detect_strokes.py video.mp4 --confidence 0.3

   # Raise threshold for fewer false positives (but may miss strokes)
   poetry run python src/detect_strokes.py video.mp4 --confidence 0.7
   ```

---

## Troubleshooting

### Video file is huge

**Cause:** Uncompressed MP4

**Solution:** Compress using ffmpeg (if available):
```bash
ffmpeg -i analysis_output/video_annotated.mp4 -vcodec h264 -acodec aac analysis_output/video_compressed.mp4
```

### Can't open video

**Cause:** Codec issue

**Solutions:**
1. Install VLC player (plays almost anything)
2. Convert to different format
3. Check file isn't corrupted (re-run detection)

### Processing is very slow

**Solutions:**
1. Disable visualization: `--no-visualize`
2. Process shorter clip: Use video editing tool to extract segment
3. Lower video resolution before processing

### Pose skeleton not showing

**Causes:**
- Player not visible in frame
- Poor lighting
- Player too far away

**Solutions:**
- Use video preprocessing (see [VIDEO_PREPROCESSING_GUIDE.md](VIDEO_PREPROCESSING_GUIDE.md))
- Record with better camera position
- Improve lighting

---

## Example Workflow

### 1. Test on short clip first

```bash
# Extract 30 seconds for quick test
# (Use video editor or ffmpeg)

poetry run python src/detect_strokes.py test_clip.mp4
```

### 2. Check annotated video

```bash
start analysis_output/test_clip_annotated.mp4
```

**Verify:**
- Detections look accurate
- Confidence scores are reasonable
- No obvious false positives/negatives

### 3. Process full video

```bash
poetry run python src/detect_strokes.py full_match.mp4
```

### 4. Review and analyze

- Watch annotated video
- Check timeline PNG
- Read text report
- Use JSON data for further analysis

---

## Customizing Visualization

Want different colors or styles? Edit `src/detect_strokes.py`:

```python
# Line ~346: Change stroke colors
stroke_colors = {
    'backhand': (107, 107, 255),    # Red (BGR)
    'fronthand': (196, 205, 78),     # Cyan (BGR)
    # Add your custom colors here
}

# Line ~407: Change box thickness
cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 3)  # ← Change 3

# Line ~411: Change label font size
label = f"{class_name.upper()} ({confidence:.0%})"
...
cv2.FONT_HERSHEY_SIMPLEX,
0.8,  # ← Change font size
```

---

## Summary

**Quick Commands:**

```bash
# Default (with annotation)
poetry run python src/detect_strokes.py video.mp4

# Without annotation (faster)
poetry run python src/detect_strokes.py video.mp4 --no-visualize

# Custom confidence threshold
poetry run python src/detect_strokes.py video.mp4 --confidence 0.6

# Custom output directory
poetry run python src/detect_strokes.py video.mp4 --output-dir ./results
```

**Output:**
- Annotated video shows pose + predictions
- Helps verify model accuracy
- Enabled by default, can be disabled for speed

**Use annotation to:**
- ✅ Debug poor detection results
- ✅ Verify model improvements after retraining
- ✅ Demonstrate model capabilities
- ✅ Create training material/presentations

Enjoy visualizing your model! 🎾
