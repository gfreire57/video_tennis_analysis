# Quick Start - Fixing Your Model

## TL;DR - What to Do Right Now

Your model has **54% accuracy** and **detects almost nothing**. Here's the fix:

### Step 0: Verify Pose Detection Works (2 min)

**IMPORTANT:** First check if MediaPipe can actually detect your body in the video!

```bash
poetry run python src/visualize_pose.py D:\path\to\your\video.mp4 --max-frames 300
```

**What to check:**
- Do you see the skeleton overlay on your body?
- Detection rate should be >80% (shown in terminal)
- If <50%, your videos may not be suitable (see warnings)

Press `q` to quit the visualization.

**If pose detection is poor (<50%):**
- Your videos have issues (player not visible, too far, poor lighting)
- **Try preprocessing first** (see Step 0.5 below)
- Or consider using different videos

**If pose detection is good (>80%):** Continue to Step 1!

### Step 0.5: Preprocess Videos (Optional - if pose detection was poor)

**If you have dark videos or small player in frame:**

```bash
# Single video (night recording, GoPro medium FOV)
poetry run python src/preprocess_video.py input.MP4 output.MP4 --auto-brighten --static-zoom 1.5 --fisheye

# Batch process all videos
poetry run python src/batch_preprocess.py D:\videos\raw D:\videos\preprocessed --auto-brighten --static-zoom 1.5
```

**What this does:**
- Auto-brightens dark videos (night recordings)
- Applies 1.5x static zoom to make player larger
- Corrects fisheye distortion

**Then recheck pose detection on preprocessed video:**
```bash
poetry run python src/visualize_pose.py output.MP4 --max-frames 300
```

Detection rate should improve (e.g., 55% → 90%!)

**See:** [VIDEO_PREPROCESSING_GUIDE.md](VIDEO_PREPROCESSING_GUIDE.md) for details

### Step 1: Retrain with Grouped Classes (5 min)

```bash
cd video_tennis_analysis
poetry run python src/train_model.py
```

**What this does:**
- Groups `slice direita` + `slice esquerda` into main classes
- Ignores `saque` (not enough data)
- Reduces from 5 classes → 2 classes (fronthand, backhand)
- **Expected improvement:** 54% → 75-85% accuracy

### Step 2: Test on Your Video (2 min)

```bash
poetry run python src/detect_strokes.py D:\path\to\your\video.mp4
```

**What to check:**
- How many strokes detected?
- Are they mostly correct?

### Step 3: Compare in MLflow (2 min)

```bash
poetry run mlflow ui
```

Open http://localhost:5000

- Find your new run
- Compare `test_accuracy` with old run
- Check confusion matrix in Artifacts tab

---

## If Results Are Still Bad

### Problem: Accuracy still low (<70%)

**Try larger window:**

Edit `src/train_model.py`:
```python
CONFIG = {
    'window_size': 45,  # Changed from 30
    'overlap': 22,      # Changed from 15
    'learning_rate': 0.001,
    'batch_size': 32,
    'group_classes': True,
}
```

Retrain:
```bash
poetry run python src/train_model.py
```

### Problem: Still detecting nothing

**Lower detection threshold:**

Edit `src/detect_strokes.py`:
```python
CONFIG = {
    'confidence_threshold': 0.3,  # Lower from 0.5
    'min_stroke_duration': 8,     # Lower from 10
}
```

Test again:
```bash
poetry run python src/detect_strokes.py your_video.mp4
```

### Problem: Too many false positives

**Raise detection threshold:**

Edit `src/detect_strokes.py`:
```python
CONFIG = {
    'confidence_threshold': 0.7,  # Raise from 0.5
    'min_stroke_duration': 15,    # Raise from 10
}
```

---

## Parameter Cheat Sheet

### Window Size (frames of context)

```python
'window_size': 30   # Current - 1 second
'window_size': 45   # Try this - 1.5 seconds (more context)
'window_size': 60   # Or this - 2 seconds (even more)
```

When to increase: Strokes are slow and deliberate
**Don't forget to update overlap:** `overlap = window_size / 2`

### Learning Rate (how fast model learns)

```python
'learning_rate': 0.001   # Current - standard
'learning_rate': 0.0005  # Try this - slower, more stable
'learning_rate': 0.0001  # Or this - very slow, very stable
```

When to decrease: Training is erratic, not converging

### Confidence Threshold (detection sensitivity)

```python
# In detect_strokes.py
'confidence_threshold': 0.3   # Very sensitive (many detections)
'confidence_threshold': 0.5   # Current - balanced
'confidence_threshold': 0.7   # Conservative (few detections)
```

Adjust until detection rate feels right

---

## Automated Tuning (Optional)

Want to try many combinations automatically?

```bash
poetry run python src/hyperparameter_tuning.py
```

**Warning:** Takes 8× training time!

Then compare all in MLflow UI.

---

## What Each File Does

| File | Purpose | When to edit |
|------|---------|-------------|
| `src/train_model.py` | Training script | Change window_size, learning_rate, batch_size |
| `src/detect_strokes.py` | Detection script | Change confidence_threshold, min_stroke_duration |
| `src/hyperparameter_tuning.py` | Auto-tuning | Run to test many combinations |

---

## Expected Timeline

1. **Retrain with grouped classes:** 5-10 minutes
2. **Test on video:** 2-5 minutes (depends on video length)
3. **Check MLflow:** 2 minutes
4. **If needed, tune and repeat:** 5-10 minutes per iteration

**Total to good results:** 15-30 minutes

---

## Expected Results

### With Grouped Classes (2 classes)

| Metric | Before | After (Expected) |
|--------|--------|-----------------|
| Accuracy | 54% | 75-85% |
| Classes | 5 (2 with 0%) | 2 (both good) |
| Real video | 1 wrong detection | 10-20+ detections |

### After Tuning (if needed)

| Metric | Value |
|--------|-------|
| Accuracy | 80-90% |
| Precision (both classes) | >80% |
| Real video | Good detection rate |

---

## Quick Checklist

- [ ] Retrained with `group_classes: True`
- [ ] Test accuracy improved (check MLflow)
- [ ] Tested on real video
- [ ] Strokes being detected?
  - [ ] Yes, mostly correct → Done! ✅
  - [ ] Yes, but many wrong → Increase confidence_threshold
  - [ ] No → Lower confidence_threshold or try larger window_size
- [ ] If accuracy <70%, tuned window_size or learning_rate

---

## Getting Help

1. **Read detailed guides:**
   - [TUNING_GUIDE.md](TUNING_GUIDE.md) - Complete tuning reference
   - [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - What changed and why
   - [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - How the model works

2. **Check MLflow UI:**
   - See all experiments
   - Compare parameters
   - Download best model

3. **Verify data:**
   ```bash
   poetry run python src/verify_annotation.py
   ```

---

## One-Liner Commands

```bash
# Visualize pose (check if MediaPipe sees you)
poetry run python src/visualize_pose.py video.mp4 --max-frames 300

# Retrain
poetry run python src/train_model.py

# Detect
poetry run python src/detect_strokes.py video.mp4

# MLflow
poetry run mlflow ui

# Auto-tune
poetry run python src/hyperparameter_tuning.py
```

**Start with retraining!** The grouped classes should fix most issues.

Good luck! 🎾
