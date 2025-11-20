# Video Preprocessing Guide

Improve your training videos for better pose detection and model performance.

## Why Preprocess Videos?

**Problems with raw footage:**
- 🌙 **Dark videos** (night recordings) - Pose detection fails in low light
- 🔍 **Player too small** (GoPro wide FOV) - Model can't see details
- 🐟 **Fisheye distortion** (GoPro lens) - Warped body proportions
- 📹 **Inconsistent quality** - Some videos great, others poor

**Solution:** Preprocess videos to:
- ✅ Brighten dark scenes automatically
- ✅ Auto-crop/zoom to make player larger (fills 60-80% of frame)
- ✅ Correct fisheye distortion
- ✅ Standardize all videos for consistent training

---

## Quick Start

**IMPORTANT:** All preprocessing features are OFF by default. You must explicitly enable the ones you need.

### Single Video (Night Recording, Need Brightness + Zoom)

```bash
poetry run python src/preprocess_video.py input.MP4 output.MP4 --auto-brighten --static-zoom 1.5
```

**This will:**
1. Automatically detect how dark the video is
2. Brighten it appropriately
3. Apply 1.5x static center crop zoom

### Batch Process All Videos

```bash
poetry run python src/batch_preprocess.py D:\videos\raw D:\videos\preprocessed --auto-brighten --static-zoom 1.5
```

Processes all `*.MP4` files in the raw folder with auto-brightness and 1.5x static zoom.

---

## Features Explained

### 1. Auto-Brightness 🌙

**What it does:** Analyzes each frame and brightens dark videos automatically.

**When to use:**
- Night recordings
- Indoor videos with poor lighting
- Underexposed footage

**How it works:**
- Calculates average brightness of each frame
- If mean brightness <80 (very dark): +60 brightness, 1.3× contrast
- If mean brightness 80-120 (somewhat dark): +30 brightness, 1.15× contrast
- If mean brightness >120 (OK): No adjustment

**Usage:**
```bash
# Auto-detect darkness and brighten (recommended)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --auto-brighten

# Manual brightness (0-100)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --brighten 50

# No brightness adjustment
poetry run python src/preprocess_video.py input.MP4 output.MP4 --no-brighten
```

### 2. Auto-Zoom/Crop 🔍

**What it does:** Automatically detects player and crops video to make them larger.

**When to use:**
- GoPro medium/wide FOV recordings
- Player appears small in frame
- Want to fill more of the frame with the player

**How it works:**
1. Uses MediaPipe to detect player in each frame
2. Calculates bounding box around player
3. Adds padding (default 30% extra space)
4. Smooths transitions to avoid jitter
5. Crops each frame to player's position
6. Resizes to consistent output size

**Before vs After:**
```
BEFORE (GoPro Medium FOV):          AFTER (Auto-Zoomed):
┌──────────────────────────┐        ┌──────────────────────────┐
│                          │        │        o                 │
│                          │        │       /|\                │
│         •                │        │      / | \               │
│        /|\               │        │        |                 │
│       / | \              │   →    │       / \                │
│         |                │        │      /   \               │
│        / \               │        │   (Player fills 70%)     │
│   (Player ~20% of frame) │        │                          │
└──────────────────────────┘        └──────────────────────────┘
```

**Usage:**
```bash
# Auto-zoom with default padding (30%)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --zoom

# More padding (50% extra space)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --zoom --zoom-padding 0.5

# Less padding (10% extra space)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --zoom --zoom-padding 0.1
```

**Zoom padding explained:**
- `0.1` = Player fills ~80% of frame (tight crop)
- `0.3` = Player fills ~60% of frame (balanced) ← **Default**
- `0.5` = Player fills ~40% of frame (lots of context)

### 3. Fisheye Correction 🐟

**What it does:** Corrects GoPro's fisheye lens distortion.

**When to use:**
- GoPro videos with wide/medium FOV
- Curved/warped appearance
- Edges of frame appear stretched

**Strength guide:**
- `0.3` = Mild correction (medium FOV GoPro)
- `0.5` = Moderate correction (default)
- `0.8` = Strong correction (wide FOV GoPro)

**Usage:**
```bash
# Correct fisheye with default strength
poetry run python src/preprocess_video.py input.MP4 output.MP4 --fisheye

# Strong correction for wide FOV
poetry run python src/preprocess_video.py input.MP4 output.MP4 --fisheye --fisheye-strength 0.8
```

**Note:** Fisheye correction happens BEFORE zoom, so corrected dimensions are used for cropping.

---

## Complete Examples

### Example 1: Night Recording with GoPro Medium FOV

**Problem:**
- Recorded at night (dark)
- GoPro medium FOV (player small, ~25% of frame)
- Slight fisheye distortion

**Solution:**
```bash
poetry run python src/preprocess_video.py \
  night_match.MP4 \
  night_match_preprocessed.MP4 \
  --auto-brighten \
  --zoom \
  --fisheye --fisheye-strength 0.3
```

**Result:**
- Video brightened automatically
- Player now fills ~60% of frame
- Fisheye distortion corrected
- Ready for training!

### Example 2: Day Recording with GoPro Narrow FOV

**Problem:**
- Good lighting (daytime)
- Narrow FOV (player already large)
- Minimal distortion

**Solution:**
```bash
poetry run python src/preprocess_video.py \
  day_match.MP4 \
  day_match_preprocessed.MP4 \
  --no-brighten
```

**Result:**
- No brightness adjustment (already good)
- No zoom (player already large)
- Minimal processing (keeps quality)

### Example 3: Batch Process Mixed Videos

**Problem:**
- 10 videos: 4 night recordings, 6 day recordings
- All GoPro medium FOV
- Need consistent preprocessing

**Solution:**
```bash
poetry run python src/batch_preprocess.py \
  D:\videos\raw \
  D:\videos\preprocessed \
  --auto-brighten \
  --zoom \
  --fisheye
```

**Result:**
- All videos processed with same settings
- Auto-brightness adjusts each video individually
- All players zoomed to consistent size
- All fisheye corrected
- Ready for batch training!

---

## Testing Before Full Processing

### Test on First 300 Frames

```bash
poetry run python src/preprocess_video.py \
  input.MP4 \
  test_output.MP4 \
  --auto-brighten --zoom --fisheye \
  --max-frames 300
```

**Why:**
- Quick test (processes ~10 seconds instead of full video)
- Check if settings are correct
- Verify brightness, zoom level, etc.

**Then:**
1. Open `test_output.MP4`
2. Check if it looks good
3. Adjust parameters if needed
4. Run full preprocessing

---

## Workflow

### Recommended Preprocessing Workflow

```
1. Identify problem videos
   ├─ Dark videos (night) → Need brightening
   ├─ Small player (wide FOV) → Need zoom
   └─ Fisheye distortion → Need correction

2. Test preprocessing on one video
   poetry run python src/preprocess_video.py \
     video.MP4 test.MP4 \
     --auto-brighten --zoom --fisheye \
     --max-frames 300

3. Check results
   ├─ Open test.MP4
   ├─ Is player larger?
   ├─ Is brightness better?
   └─ Does it look natural?

4. Verify pose detection improvement
   poetry run python src/visualize_pose.py test.MP4 --max-frames 300
   └─ Detection rate should increase (e.g., 60% → 90%)

5. If satisfied, batch process all videos
   poetry run python src/batch_preprocess.py \
     D:\videos\raw \
     D:\videos\preprocessed \
     --auto-brighten --zoom --fisheye

6. Replace original videos with preprocessed
   └─ Update Label Studio paths if needed
```

---

## Parameter Tuning

### Brightness

**Too dark still?**
```bash
# Increase brightness manually
poetry run python src/preprocess_video.py input.MP4 output.MP4 --brighten 80
```

**Too bright (washed out)?**
```bash
# Reduce brightness
poetry run python src/preprocess_video.py input.MP4 output.MP4 --brighten 30
```

**Brightness is fine?**
```bash
# Skip brightness adjustment
poetry run python src/preprocess_video.py input.MP4 output.MP4 --no-brighten
```

### Zoom

**Player still too small?**
```bash
# Tighter crop (less padding)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --zoom --zoom-padding 0.1
```

**Player too zoomed (parts cut off)?**
```bash
# More padding
poetry run python src/preprocess_video.py input.MP4 output.MP4 --zoom --zoom-padding 0.5
```

**Zoom is jittery?**
- This is normal - smoothing window is 15 frames
- Can't be adjusted (hardcoded for stability)

### Fisheye

**Still looks distorted?**
```bash
# Increase correction strength
poetry run python src/preprocess_video.py input.MP4 output.MP4 --fisheye --fisheye-strength 0.8
```

**Looks over-corrected (pinched)?**
```bash
# Reduce correction strength
poetry run python src/preprocess_video.py input.MP4 output.MP4 --fisheye --fisheye-strength 0.3
```

---

## Batch Processing

### Basic Batch

```bash
poetry run python src/batch_preprocess.py input_folder output_folder
```

### Custom Pattern

```bash
# Process only .mp4 files (lowercase)
poetry run python src/batch_preprocess.py input_folder output_folder --pattern "*.mp4"

# Process .avi files
poetry run python src/batch_preprocess.py input_folder output_folder --pattern "*.avi"
```

### Disable Features

```bash
# Brighten but don't zoom
poetry run python src/batch_preprocess.py input_folder output_folder --auto-brighten --no-zoom

# Zoom but don't brighten
poetry run python src/batch_preprocess.py input_folder output_folder --no-brighten --zoom

# Just fisheye correction
poetry run python src/batch_preprocess.py input_folder output_folder --no-brighten --no-zoom --fisheye
```

---

## Comparing Results

### Before/After Comparison

1. **Open both videos side-by-side**
   - Original: `raw_video.MP4`
   - Preprocessed: `raw_video_preprocessed.MP4`

2. **Check improvements:**
   - [ ] Is player larger in frame?
   - [ ] Is brightness better?
   - [ ] Is fisheye corrected?
   - [ ] Does motion look smooth?

3. **Check pose detection:**
   ```bash
   # Original
   poetry run python src/visualize_pose.py raw_video.MP4 --max-frames 300

   # Preprocessed
   poetry run python src/visualize_pose.py raw_video_preprocessed.MP4 --max-frames 300
   ```

4. **Compare detection rates:**
   - Original: e.g., 65% detection
   - Preprocessed: e.g., 92% detection ← Goal!

---

## Common Issues

### Issue 1: Player Gets Cut Off During Zoom

**Cause:** Player moves quickly or camera doesn't follow

**Solutions:**
1. Increase zoom padding:
   ```bash
   --zoom-padding 0.5  # More room
   ```

2. Check if camera is stable (should follow player)

3. If player leaves frame often, zoom may not work well

### Issue 2: Video Becomes Too Small

**Cause:** Player detection varies too much frame-to-frame

**Solution:**
- Check pose detection on original:
  ```bash
  poetry run python src/visualize_pose.py original.MP4 --max-frames 300
  ```
- If detection <70%, preprocessing won't help much
- Consider different camera angle/distance

### Issue 3: Brightness Looks Unnatural

**Cause:** Auto-brightness too aggressive or video has mixed lighting

**Solutions:**
1. Use manual brightness:
   ```bash
   --brighten 40  # Moderate increase
   ```

2. Or disable:
   ```bash
   --no-brighten
   ```

### Issue 4: Fisheye Correction Makes Things Worse

**Cause:** Wrong strength for your FOV setting

**Solutions:**
1. Try different strengths:
   ```bash
   --fisheye-strength 0.3  # GoPro Medium FOV
   --fisheye-strength 0.5  # GoPro Wide FOV (default)
   --fisheye-strength 0.8  # GoPro SuperView
   ```

2. Or disable if not needed:
   - Don't use `--fisheye` flag

### Issue 5: Processing is Slow

**Causes:**
- Video is long
- High resolution (4K)
- Zoom enabled (requires 2 passes)

**Solutions:**
1. Test on subset first:
   ```bash
   --max-frames 300
   ```

2. Process at lower resolution (TODO: not implemented yet)

3. Be patient - preprocessing is one-time cost!

---

## Performance Impact

### Processing Time

| Video Length | Resolution | Features | Time (approx) |
|-------------|-----------|----------|---------------|
| 1 min | 1080p | Brighten only | ~30 sec |
| 1 min | 1080p | Brighten + Zoom | ~90 sec |
| 5 min | 1080p | Brighten + Zoom | ~7 min |
| 10 min | 4K | All features | ~30 min |

**Note:** Zoom requires 2 passes (analyze + process), so it's slower.

### File Size

Preprocessed videos are usually:
- **Similar size** if same resolution
- **Smaller** if zoomed (smaller output dimensions)
- **Slightly larger** if fisheye corrected (depends on codec)

---

## Summary

### When to Preprocess

| Problem | Solution |
|---------|----------|
| Dark video (night) | `--auto-brighten` |
| Player too small (wide FOV) | `--zoom` |
| Fisheye distortion | `--fisheye` |
| All of the above | `--auto-brighten --zoom --fisheye` |

### Quick Commands

```bash
# Night recording, GoPro medium FOV
poetry run python src/preprocess_video.py input.MP4 output.MP4 --auto-brighten --zoom --fisheye

# Day recording, player already visible
poetry run python src/preprocess_video.py input.MP4 output.MP4 --no-brighten

# Batch process all videos
poetry run python src/batch_preprocess.py raw_folder preprocessed_folder --auto-brighten --zoom

# Test first (300 frames)
poetry run python src/preprocess_video.py input.MP4 test.MP4 --auto-brighten --zoom --max-frames 300
```

### Workflow Checklist

- [ ] Identify which videos need preprocessing
- [ ] Test preprocessing on one video (300 frames)
- [ ] Check output looks good
- [ ] Verify pose detection improved
- [ ] Batch process all problem videos
- [ ] Replace originals or update annotation paths
- [ ] Retrain model with improved videos

**Better videos = Better pose detection = Better model!** 🎾
