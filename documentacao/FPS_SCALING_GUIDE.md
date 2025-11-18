# FPS Scaling Guide

## Problem

Videos recorded at different frame rates (30 FPS, 48 FPS, 60 FPS) require different window sizes to capture the same temporal duration. For example:
- A 1-second window at 30 FPS = 30 frames
- A 1-second window at 60 FPS = 60 frames
- A 1-second window at 48 FPS = 48 frames

Without FPS scaling, a fixed window of 45 frames would represent:
- 1.5 seconds at 30 FPS ✓
- 0.75 seconds at 60 FPS ✗ (too short!)
- 0.94 seconds at 48 FPS ✗ (too short!)

## Solution: Automatic FPS Scaling

Both `train_model.py` and `detect_strokes.py` now automatically scale window parameters to maintain consistent **temporal duration** across different FPS videos.

### Configuration

```python
CONFIG = {
    'reference_fps': 30,      # Reference FPS for calibration
    'window_size': 45,        # Frames at 30 FPS = 1.5 seconds
    'overlap': 15,            # Frames at 30 FPS = 0.5 seconds
    'MIN_ANNOTATION_LENGTH': 15,  # Frames at 30 FPS = 0.5 seconds
}
```

### How It Works

When processing a video, the scripts:

1. **Detect video FPS** (e.g., 60 FPS)
2. **Calculate scale factor** = video_fps / reference_fps = 60 / 30 = 2.0x
3. **Scale all parameters**:
   - window_size: 45 × 2.0 = **90 frames** → 1.5 seconds at 60 FPS ✓
   - overlap: 15 × 2.0 = **30 frames** → 0.5 seconds at 60 FPS ✓
   - min_annotation_length: 15 × 2.0 = **30 frames** → 0.5 seconds at 60 FPS ✓

### Examples

#### 30 FPS Video (Reference)
```
Reference: 30 FPS → Video: 30 FPS (scale factor: 1.00x)
Window: 45 → 45 frames (1.50s → 1.50s)
Overlap: 15 → 15 frames
```

#### 60 FPS Video
```
Reference: 30 FPS → Video: 60 FPS (scale factor: 2.00x)
Window: 45 → 90 frames (1.50s → 1.50s)
Overlap: 15 → 30 frames
```

#### 48 FPS Video
```
Reference: 30 FPS → Video: 48 FPS (scale factor: 1.60x)
Window: 45 → 72 frames (1.50s → 1.50s)
Overlap: 15 → 24 frames
```

## Benefits

✅ **Consistent temporal windows** across all videos regardless of FPS
✅ **No manual parameter adjustment** needed for different FPS
✅ **Same model works for all FPS** videos (maintains temporal patterns)
✅ **Automatic annotation expansion** scales proportionally

## Usage

### Training with Mixed FPS Videos

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

### Detection on Any FPS Video

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

## Important Notes

⚠️ **reference_fps must match between training and detection**
⚠️ **Model input shape is determined by the FIRST video processed during training**
⚠️ If your first training video is 60 FPS, all subsequent videos will be scaled to match that temporal window size

## Recommendations

1. **Use 30 FPS as reference** - most common, easier to reason about
2. **Calibrate window_size at reference FPS** - use `analyze_annotations.py` on 30 FPS videos
3. **Mix FPS videos freely** - the scaling handles it automatically
4. **Check console output** - verify scaling is working as expected
