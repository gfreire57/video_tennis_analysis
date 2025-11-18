# Static Zoom Feature - Quick Guide

The static zoom feature provides a simple center crop without player tracking. This is faster and avoids the "weird" tracking behavior.

## Quick Comparison

### Tracking Zoom (--zoom)
- **Pros:** Follows player around
- **Cons:** Can be jittery, slower (2 passes), may look strange
- **Use when:** Player moves a lot in frame

### Static Zoom (--static-zoom)
- **Pros:** Simple center crop, fast (1 pass), smooth, predictable
- **Cons:** Doesn't follow player (they may move out of frame)
- **Use when:** Player stays relatively centered in frame

## Usage Examples

### Basic Static Zoom (1.5x)

```bash
poetry run python src/preprocess_video.py \
  input.MP4 \
  output.MP4 \
  --static-zoom 1.5
```

**Result:**
- Crops to center of frame
- Makes everything 1.5x larger
- Original 1920x1080 → 1280x720 crop → scaled back to 1280x720

### Stronger Zoom (2x)

```bash
poetry run python src/preprocess_video.py \
  input.MP4 \
  output.MP4 \
  --static-zoom 2.0
```

**Result:**
- 2x zoom
- Original 1920x1080 → 960x540 crop → scaled back to 960x540
- Player appears twice as large

### Static Zoom + Brightness + Fisheye

```bash
poetry run python src/preprocess_video.py \
  night_video.MP4 \
  output.MP4 \
  --auto-brighten \
  --static-zoom 1.8 \
  --fisheye
```

**Perfect for:** GoPro night recordings where player stays relatively centered

## How to Choose Zoom Factor

| Zoom Factor | Effect | Use Case |
|------------|--------|----------|
| 1.2 | Subtle zoom | Player already fairly large |
| 1.5 | Moderate zoom | GoPro medium FOV (recommended) |
| 1.8 | Strong zoom | GoPro wide FOV or player very small |
| 2.0 | Very strong | Player is tiny in original |
| 2.5+ | Extreme | Only if absolutely necessary (may lose quality) |

## Visual Example

```
BEFORE (Original GoPro Medium FOV):
┌────────────────────────────────────┐
│                                    │
│                                    │
│           •  ← Player (small)      │
│          /|\                       │
│           |                        │
│          / \                       │
│                                    │
│                                    │
└────────────────────────────────────┘

AFTER (--static-zoom 1.8):
┌────────────────────────────────────┐
│          o  ← Player (large)       │
│         /|\                        │
│        / | \                       │
│          |                         │
│         / \                        │
│        /   \                       │
│                                    │
│   (Cropped to center, resized)    │
└────────────────────────────────────┘
```

## Testing Different Zoom Factors

Test on first 300 frames to find best zoom:

```bash
# Try 1.5x
poetry run python src/preprocess_video.py \
  input.MP4 test_1.5x.MP4 \
  --static-zoom 1.5 --max-frames 300

# Try 1.8x
poetry run python src/preprocess_video.py \
  input.MP4 test_1.8x.MP4 \
  --static-zoom 1.8 --max-frames 300

# Try 2.0x
poetry run python src/preprocess_video.py \
  input.MP4 test_2.0x.MP4 \
  --static-zoom 2.0 --max-frames 300
```

Then open all three and pick the best one!

## When Player Moves Off-Center

**Problem:** If player moves around a lot, static zoom may cut them off.

**Solutions:**

1. **Use tracking zoom instead:**
   ```bash
   poetry run python src/preprocess_video.py \
     input.MP4 output.MP4 --zoom
   ```

2. **Use lower static zoom factor:**
   ```bash
   # Less aggressive crop = more room for movement
   poetry run python src/preprocess_video.py \
     input.MP4 output.MP4 --static-zoom 1.3
   ```

3. **Record new videos** with player more centered

## Batch Processing with Static Zoom

You can use static zoom in batch processing too:

```bash
poetry run python src/batch_preprocess.py \
  D:\videos\raw \
  D:\videos\preprocessed \
  --auto-brighten \
  --static-zoom 1.5 \
  --fisheye
```

This will apply static zoom to all videos in the raw folder.

## Performance

| Mode | Speed | Passes | Smoothness |
|------|-------|--------|-----------|
| No zoom | Fast | 1 | N/A |
| **Static zoom** | **Fast** | **1** | **Very smooth** |
| Tracking zoom | Slow | 2 | Can be jittery |

**Static zoom is 2-3x faster than tracking zoom!**

## Recommended Settings for Your Videos

Based on your description (GoPro medium FOV):

### Night Videos
```bash
poetry run python src/preprocess_video.py \
  night_video.MP4 \
  output.MP4 \
  --auto-brighten \
  --static-zoom 1.6 \
  --fisheye --fisheye-strength 0.5
```

### Day Videos (Narrow FOV - already good)
```bash
poetry run python src/preprocess_video.py \
  day_video.MP4 \
  output.MP4 \
  --no-brighten \
  --static-zoom 1.2  # Just slight zoom
```

Or don't preprocess day videos at all if they're already good!

## Summary

**Quick Commands:**

```bash
# Simple static zoom (recommended)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --static-zoom 1.5

# With brightness
poetry run python src/preprocess_video.py input.MP4 output.MP4 --auto-brighten --static-zoom 1.5

# Full preprocessing (night video, GoPro)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --auto-brighten --static-zoom 1.6 --fisheye

# Test first
poetry run python src/preprocess_video.py input.MP4 test.MP4 --static-zoom 1.5 --max-frames 300
```

**Static zoom is perfect for your use case - simple, fast, and avoids the tracking weirdness!** 🎾
