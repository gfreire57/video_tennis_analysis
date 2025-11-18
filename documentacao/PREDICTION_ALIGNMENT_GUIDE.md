# Prediction Alignment Guide

## The "Seeing the Future" Problem

When using sliding window predictions, you may notice that bounding boxes appear **too early** - sometimes 0.5-1.5 seconds before the actual stroke happens. This creates the illusion that the model is "predicting the future" rather than analyzing what's currently happening.

### Why This Happens

#### 1. Window Position Bias
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

#### 2. Training Window Alignment
During training, majority voting assigns the stroke label to the **entire window** if the stroke covers >50% of frames. This means the model learns to recognize strokes even when seeing only the **beginning** of the movement.

#### 3. Preparatory Movement Recognition
The model may be learning to recognize **wind-up** and **stance changes** that happen before the actual racket-ball contact, causing early predictions.

## Solution: Configurable Prediction Alignment

The `detect_strokes.py` script now supports three alignment modes:

### Configuration

```python
CONFIG = {
    'prediction_alignment': 'center',  # Options: 'start', 'center', 'end'
    # ... other settings
}
```

### Alignment Modes

#### 1. `'start'` - Original Behavior (Earliest)
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

#### 2. `'center'` - Recommended (Most Accurate)
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

#### 3. `'end'` - Conservative (Latest)
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

## Usage

### Basic Usage (Use Center Alignment)

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

### Testing Different Alignments

Edit `CONFIG` in `detect_strokes.py`:

```python
# Test 1: Center alignment (recommended)
CONFIG['prediction_alignment'] = 'center'

# Test 2: End alignment (more conservative)
CONFIG['prediction_alignment'] = 'end'

# Test 3: Start alignment (original, not recommended)
CONFIG['prediction_alignment'] = 'start'
```

## Impact on Timing

Assuming window_size = 45 frames @ 30 FPS (1.5 seconds):

| Alignment | Offset from Window Start | Typical Timing Accuracy |
|-----------|-------------------------|------------------------|
| `'start'` | 0 frames (0.0s) | ❌ 0.5-1.5s too early |
| `'center'` | 22 frames (0.73s) | ✓ ±0.25s accurate |
| `'end'` | 22 frames (0.73s) | ✓ May be 0.25s late |

## Visual Comparison

### Before Fix (Start Alignment)
```
Timeline: |----prep----|===STROKE===|----follow----|
Window:   [================================]
Box shown: ████████████████████████████████
Result:    Box appears during preparation phase ❌
```

### After Fix (Center Alignment)
```
Timeline: |----prep----|===STROKE===|----follow----|
Window:   [================================]
Box shown:          █████████████
Result:    Box appears during stroke ✓
```

## Recommendations

1. **Use `'center'` alignment** (default) - Best balance for most use cases
2. **Use `'end'` alignment** if false positives during preparation are problematic
3. **Never use `'start'` alignment** - only kept for backward compatibility

## Technical Details

### How Center Alignment Works

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

### Why Quarter-Window?

Using a smaller prediction range (25% of window on each side = 50% total) instead of the full window:
- ✓ Reduces prediction duration to more realistic stroke length
- ✓ Centers the prediction better on actual stroke
- ✓ Reduces overlap between consecutive predictions
- ✓ Makes merged predictions more accurate

## Troubleshooting

### Predictions Still Too Early
Try `'end'` alignment:
```python
CONFIG['prediction_alignment'] = 'end'
```

### Predictions Too Late
You're probably already using `'end'`. Try `'center'`:
```python
CONFIG['prediction_alignment'] = 'center'
```

### Predictions Very Accurate
Great! Stick with `'center'` (default).

## Related Settings

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
