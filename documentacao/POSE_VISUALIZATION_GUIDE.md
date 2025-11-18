# Pose Visualization Guide

This guide shows you how to verify that MediaPipe can detect your body in tennis videos.

## Why Visualize Pose?

**Before training a model, you should check:**
- Can MediaPipe see your body in the video?
- What's the pose detection rate?
- Are certain frames failing (player occluded, too far, etc.)?

**If pose detection is poor (<50%), the model won't work** - it needs to see your body to recognize strokes!

---

## Basic Usage

### Quick Test (First 300 Frames)

```bash
poetry run python src/visualize_pose.py your_video.mp4 --max-frames 300
```

This will:
- Show a window with skeleton overlay on your body
- Display detection rate in real-time
- Process only the first 300 frames (quick test)

**Press `q` to quit**

### Full Video Analysis

```bash
poetry run python src/visualize_pose.py your_video.mp4
```

Processes entire video (may take a while for long videos).

### Save Output Video

```bash
poetry run python src/visualize_pose.py your_video.mp4 --output pose_output.mp4
```

Saves video with skeleton overlay so you can review later.

### Headless Mode (No Display)

```bash
poetry run python src/visualize_pose.py your_video.mp4 --no-display --output result.mp4
```

Useful if you're on a server or want to process multiple videos.

---

## Interactive Controls

When the visualization window is open:

| Key | Action |
|-----|--------|
| `q` | Quit visualization |
| `p` | Pause/Resume playback |
| `s` | Save current frame as PNG |

---

## What You See

### Good Detection

```
┌─────────────────────────────────────────┐
│  Frame 150/300 - POSE DETECTED         │ ← Green text
│  Detection rate: 95.3%                 │
│                                         │
│        o  ← Head                       │
│       /|\                              │
│      / | \                             │
│        |                               │
│       / \                              │
│      /   \                             │
│     ↑     ↑                            │
│   Feet  Feet                           │
│                                         │
│  (Skeleton overlay on player)          │
└─────────────────────────────────────────┘
```

### Poor Detection

```
┌─────────────────────────────────────────┐
│  Frame 150/300 - NO POSE DETECTED      │ ← Red text
│  Detection rate: 42.7%                 │
│                                         │
│  (No skeleton - player not detected)   │
│                                         │
│  Possible issues:                      │
│  - Player off-screen                   │
│  - Too far from camera                 │
│  - Poor lighting                       │
└─────────────────────────────────────────┘
```

---

## Interpreting Results

### Detection Rate Thresholds

| Detection Rate | Status | What It Means |
|---------------|--------|---------------|
| **>80%** | ✅ Excellent | Video is perfect for training |
| **50-80%** | ⚠️ OK | Some frames will be lost, but usable |
| **<50%** | ❌ Poor | Video may not work well for training |

### Terminal Output Example

```
======================================================================
MEDIAPIPE POSE VISUALIZATION
======================================================================

Input video: D:\videos\match.mp4
Video properties:
  Resolution: 1920x1080
  FPS: 30.00
  Total frames: 9000
  Duration: 300.00 seconds

======================================================================
PROCESSING VIDEO
======================================================================

Controls:
  - Press 'q' to quit
  - Press 'p' to pause/resume
  - Press 's' to save current frame as image

Processed 100 frames... Detection rate: 92.0%
Processed 200 frames... Detection rate: 89.5%
Processed 300 frames... Detection rate: 91.3%

======================================================================
SUMMARY
======================================================================

Total frames processed: 300
Frames with pose detected: 274 (91.3%)
Frames without pose: 26 (8.7%)

✅ Good detection rate (>80%)!
   Video is suitable for training
```

---

## Common Issues & Solutions

### Issue 1: Low Detection Rate (<50%)

**Symptoms:**
- Red "NO POSE DETECTED" appears frequently
- No skeleton visible on player

**Possible Causes:**
1. **Player not visible**
   - Solution: Use videos where player is fully visible
   - Check: Player should be in frame during strokes

2. **Camera too far**
   - Solution: Use closer camera angles
   - MediaPipe works best when player fills 20-80% of frame

3. **Poor lighting**
   - Solution: Use well-lit videos
   - Avoid backlighting (player silhouette)

4. **Low video quality**
   - Solution: Use higher resolution videos (720p+)
   - Minimum: 480p

5. **Player occluded**
   - Solution: Avoid videos with net/fence blocking player
   - Use court-side angles instead of behind-net angles

### Issue 2: Detection Works Sometimes

**Symptoms:**
- Detection rate 50-80%
- Works in some frames, fails in others

**Likely Cause:**
- Player moves in/out of good camera view
- Some strokes are visible, others aren't

**Solution:**
- Review saved output video to see which frames fail
- Annotate only frames where pose is detected
- Or use multiple camera angles

### Issue 3: Detection Rate Drops During Strokes

**Symptoms:**
- Detection good when standing still
- Drops during fast movements

**Likely Cause:**
- Motion blur from fast strokes
- Player moving out of frame during swing

**Solution:**
- Use higher FPS videos (60fps better than 30fps)
- Better camera that handles motion
- Ensure full stroke is in frame

---

## Example Workflows

### Workflow 1: Quick Check Before Training

```bash
# Test first 300 frames of each training video
poetry run python src/visualize_pose.py video1.mp4 --max-frames 300
poetry run python src/visualize_pose.py video2.mp4 --max-frames 300
poetry run python src/visualize_pose.py video3.mp4 --max-frames 300
```

**Decision:**
- All >80%? → Good to train!
- Any <50%? → Don't use that video

### Workflow 2: Save Annotated Videos for Review

```bash
# Process and save all videos
poetry run python src/visualize_pose.py video1.mp4 -o output1.mp4
poetry run python src/visualize_pose.py video2.mp4 -o output2.mp4
poetry run python src/visualize_pose.py video3.mp4 -o output3.mp4
```

Review outputs later to decide which videos to use.

### Workflow 3: Diagnose Poor Model Performance

If your model isn't working:

```bash
# Check if pose detection is the issue
poetry run python src/visualize_pose.py training_video.mp4
```

**If detection <80%:**
- Problem is likely the video quality
- Model can't learn from frames where it can't see the body

**If detection >80%:**
- Problem is likely model parameters or annotations
- Video quality is fine

---

## Advanced Options

### Process Specific Section of Video

```bash
# Process frames 100-400 only (quick, targeted check)
# Note: Currently processes from start, use --max-frames to limit

# Example: Check middle section
poetry run python src/visualize_pose.py video.mp4 --max-frames 500
# Then skip first 200 frames in your analysis
```

### Batch Processing Multiple Videos

Create a script `check_all_videos.bat`:

```batch
@echo off
for %%f in (D:\videos\*.mp4) do (
    echo Checking %%f
    poetry run python src/visualize_pose.py "%%f" --max-frames 300 --no-display
)
```

Run:
```bash
check_all_videos.bat
```

---

## Understanding MediaPipe Pose

### What MediaPipe Detects

33 body landmarks:
- **Face:** Nose, eyes, ears, mouth
- **Upper body:** Shoulders, elbows, wrists, hands
- **Core:** Hips
- **Lower body:** Knees, ankles, feet, heels, toes

Each landmark has:
- `x, y` coordinates (position in frame)
- `z` coordinate (depth/distance)
- `visibility` score (0-1, how visible it is)

### Why Tennis is Good for Pose Detection

✅ **Pros:**
- Player usually fully visible
- Clear body movements
- Well-lit outdoor courts
- Single person in focus

⚠️ **Challenges:**
- Fast movements (motion blur)
- Wide camera angles (player small)
- Net/fence occlusions
- Bright sun (overexposure)

---

## Tips for Better Pose Detection

### Recording New Videos

1. **Camera position:**
   - Side-on angle (not behind net)
   - Player should fill 30-60% of frame
   - Keep player centered

2. **Lighting:**
   - Avoid shooting into sun
   - Overcast days are ideal (even lighting)
   - Avoid heavy shadows

3. **Camera settings:**
   - 720p minimum, 1080p preferred
   - 30fps minimum, 60fps better
   - Enable image stabilization

4. **During play:**
   - Keep camera still (tripod best)
   - Follow player if they move
   - Ensure full body visible during strokes

### Using Existing Videos

1. **Evaluate first:**
   ```bash
   poetry run python src/visualize_pose.py video.mp4 --max-frames 300
   ```

2. **Accept if:**
   - Detection rate >80%
   - Player clearly visible
   - Most strokes have good detection

3. **Reject if:**
   - Detection rate <50%
   - Player frequently off-screen
   - Severe motion blur

---

## Troubleshooting

### Video Window Doesn't Appear

**Windows:**
- Make sure you're not using `--no-display`
- Try updating OpenCV: `poetry add opencv-python --latest`

**Linux/Remote:**
- Use `--no-display` and save output: `--output result.mp4`
- Or setup X11 forwarding for remote display

### "Cannot open video" Error

**Check:**
1. File path is correct (use absolute paths on Windows)
2. Video format is supported (MP4, AVI, MOV)
3. Video file isn't corrupted

### Video Plays Too Fast/Slow

This is normal - processing speed depends on your computer.
- The output video will be correct speed
- Press `p` to pause if needed

### Saved Screenshots Location

Screenshots are saved in current directory:
- `pose_frame_0.png`
- `pose_frame_150.png`
- etc.

---

## Summary

### Quick Commands

```bash
# Quick test (first 300 frames)
poetry run python src/visualize_pose.py video.mp4 --max-frames 300

# Full video with output
poetry run python src/visualize_pose.py video.mp4 --output pose_video.mp4

# Batch check (no display)
poetry run python src/visualize_pose.py video.mp4 --no-display
```

### Decision Tree

```
Run visualization
    ↓
Detection rate >80%?
    ├─ YES → ✅ Use video for training
    └─ NO  → Detection rate >50%?
              ├─ YES → ⚠️ May use, but expect some loss
              └─ NO  → ❌ Don't use, find better video
```

### Before Training Checklist

- [ ] Visualized pose on all training videos
- [ ] All videos have >80% detection rate (or >50% minimum)
- [ ] Reviewed which frames fail (if any)
- [ ] Verified strokes are visible and detected
- [ ] Videos show full body during stroke execution

**Now you're ready to train with confidence!** 🎾
