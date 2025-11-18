# Hyperparameter Tuning Guide

This guide explains how to tune the model parameters to improve performance.

## Current Issues

Based on your training results:

1. **Low accuracy (54%)** - Model is barely better than random guessing
2. **Poor detection** - Only detected 1 incorrect stroke in 2.5 minute video
3. **Class imbalance** - slice classes have only 4-5 test samples
4. **Grouped classes needed** - Too many classes for limited data

## Solutions Implemented

### 1. Class Grouping ✅

**Change made:**
- `slice direita` → absorbed into `fronthand`
- `slice esquerda` → absorbed into `backhand`
- `saque` → ignored (not enough data)

**Result:** Only 2 classes (fronthand, backhand) instead of 5

**In train_model.py:**
```python
CONFIG = {
    ...
    'group_classes': True,  # ← NEW: Groups classes
}
```

The grouping happens automatically in `create_sequences_from_frames()`.

### 2. Lowered Detection Thresholds ✅

**Change made in detect_strokes.py:**
```python
CONFIG = {
    'confidence_threshold': 0.5,  # Lowered from 0.7
    'min_stroke_duration': 10,    # Lowered from 13
}
```

**Why:** Your current model might have low confidence, so lower thresholds allow more detections.

---

## How to Tune Parameters

### Option 1: Manual Tuning (Recommended)

Edit parameters in `train_model.py` CONFIG and retrain:

```python
CONFIG = {
    'window_size': 30,       # ← Change this
    'overlap': 15,           # ← Change this
    'learning_rate': 0.001,  # ← Change this
    'batch_size': 32,        # ← Change this
}
```

Then run:
```bash
poetry run python src/train_model.py
```

MLflow will track each run automatically.

### Option 2: Automated Grid Search

Run multiple experiments automatically:

```bash
poetry run python src/hyperparameter_tuning.py
```

This will:
1. Run 8 different parameter combinations
2. Track all results in MLflow
3. Let you compare which works best

**Warning:** This takes a long time (8× training time)!

---

## Parameters to Tune

### 1. Window Size

**What it does:** Number of frames in each sequence (how much history the model sees)

**Current:** 30 frames (1 second @ 30fps)

**Try:**
- `window_size: 45` (1.5 seconds) - Better for slower, more deliberate strokes
- `window_size: 60` (2 seconds) - Even more context

**When to increase:**
- Strokes are long and deliberate
- Model confuses similar strokes (needs more context)

**When to decrease:**
- Strokes are quick
- Running out of memory

**How to adjust overlap:**
```python
'window_size': 45,
'overlap': 22,  # Keep overlap = window_size / 2 (50%)
```

### 2. Learning Rate

**What it does:** How big the weight updates are during training

**Current:** 0.001

**Try:**
- `learning_rate: 0.0005` - Slower, more careful learning
- `learning_rate: 0.0001` - Very slow, very stable

**When to decrease:**
- Training loss is erratic (jumps around)
- Model not converging
- Validation accuracy plateaus early

**When to increase:**
- Training is very slow
- Loss decreases too slowly

**Symptom → Solution:**
```
Loss jumps around → Lower LR to 0.0005
Loss barely moves → Increase LR to 0.002
```

### 3. Batch Size

**What it does:** Number of samples processed before updating weights

**Current:** 32

**Try:**
- `batch_size: 16` - Smaller batches, more updates
- `batch_size: 64` - Larger batches, stabler training

**When to decrease:**
- Running out of memory
- Want more frequent weight updates
- Small dataset

**When to increase:**
- Training is unstable
- Have lots of data
- Want faster training (if GPU has memory)

### 4. Overlap

**What it does:** How much frames overlap between windows

**Current:** 15 frames (50% of window_size=30)

**Try:**
- `overlap: 20` - More training samples, more augmentation
- `overlap: 25` - Even more samples (83% overlap)

**When to increase:**
- Need more training data
- Model underfitting (low train accuracy)
- Class balance issues

**When to decrease:**
- Too much data (slow training)
- Model overfitting
- Very long videos

**Rule:** Keep `overlap = window_size / 2` for balanced approach

---

## Recommended Tuning Sequence

### Step 1: Test Grouped Classes (NOW)

**Run training with grouped classes:**

```bash
poetry run python src/train_model.py
```

**Expected improvement:**
- Before: 5 classes, 54% accuracy, some classes 0%
- After: 2 classes, should be >70% accuracy

**If accuracy is still low (<70%)**, proceed to Step 2.

### Step 2: Try Larger Window

**Edit train_model.py:**
```python
CONFIG = {
    'window_size': 45,  # Changed from 30
    'overlap': 22,      # Changed from 15
    'learning_rate': 0.001,
    'batch_size': 32,
}
```

**Run:**
```bash
poetry run python src/train_model.py
```

**Check MLflow** (open http://localhost:5000):
- Compare test_accuracy between window_size=30 and window_size=45
- If 45 is better, keep it

### Step 3: Try Lower Learning Rate

**Edit train_model.py:**
```python
CONFIG = {
    'window_size': 45,  # Use best from Step 2
    'overlap': 22,
    'learning_rate': 0.0005,  # Changed from 0.001
    'batch_size': 32,
}
```

**Run:**
```bash
poetry run python src/train_model.py
```

**Check MLflow:**
- Compare test_accuracy
- Look at training curves (training_history.png artifact)
- Smoother curve = better learning rate

### Step 4: Test on Real Video

After each training run:

```bash
poetry run python src/detect_strokes.py path/to/your/video.mp4
```

**Evaluate results:**
- How many strokes detected?
- Are they correct?
- Too many false positives? → Increase confidence_threshold in detect_strokes.py
- Too few detections? → Decrease confidence_threshold

---

## Comparing Results in MLflow

### Start MLflow UI

```bash
poetry run mlflow ui
```

Open: http://localhost:5000

### Compare Experiments

1. **Select multiple runs** (checkbox on left)
2. **Click "Compare"** button
3. **View comparison:**
   - Parameters tab: See which settings differed
   - Metrics tab: See test_accuracy side-by-side
   - Charts tab: Visualize metric differences

### Sorting Runs

Click column headers to sort:
- **test_accuracy** (descending) - Find best model
- **Start Time** (descending) - Find recent runs

### Download Best Model

1. Click on best run (highest test_accuracy)
2. **Artifacts tab**
3. **Download:**
   - Model (if you want to use it directly)
   - Confusion matrix (see which classes confuse the model)
   - Training history plot (check for overfitting)

---

## Quick Tuning Experiments

### Experiment A: Grouped Classes (2 classes)

```python
# train_model.py
CONFIG = {
    'window_size': 30,
    'overlap': 15,
    'learning_rate': 0.001,
    'batch_size': 32,
    'group_classes': True,  # ← This is key
}
```

**Expected:** ~70-85% accuracy (much better than 54%)

### Experiment B: Larger Context

```python
CONFIG = {
    'window_size': 45,
    'overlap': 22,
    'learning_rate': 0.001,
    'batch_size': 32,
    'group_classes': True,
}
```

**Expected:** Slightly better if strokes are slow/deliberate

### Experiment C: Slower Learning

```python
CONFIG = {
    'window_size': 45,
    'overlap': 22,
    'learning_rate': 0.0005,  # ← Lower
    'batch_size': 32,
    'group_classes': True,
}
```

**Expected:** Smoother training, possibly better generalization

### Experiment D: More Training Data

```python
CONFIG = {
    'window_size': 45,
    'overlap': 30,  # ← More overlap = more samples
    'learning_rate': 0.0005,
    'batch_size': 32,
    'group_classes': True,
}
```

**Expected:** Even more training samples, less overfitting

---

## Troubleshooting

### Problem: Accuracy still low after grouping classes

**Possible causes:**
1. Not enough training data
2. Annotations are poor quality
3. Videos too different from each other
4. Model too simple for task

**Solutions:**
1. Verify annotations:
   ```bash
   poetry run python src/verify_annotation.py
   ```
   Check that labels make sense

2. Check confusion matrix:
   - Look at MLflow artifacts
   - If model confuses fronthand/backhand equally → need more data
   - If one class much worse → need more samples of that class

3. Annotate more videos:
   - Aim for 100+ samples per class
   - Currently have ~400 fronthand, ~350 backhand after grouping

### Problem: Model detects nothing in real videos

**Possible causes:**
1. Model confidence too low
2. Detection thresholds too strict
3. Model not trained on similar videos

**Solutions:**
1. Lower confidence in detect_strokes.py:
   ```python
   'confidence_threshold': 0.3,  # Even lower
   ```

2. Lower minimum duration:
   ```python
   'min_stroke_duration': 5,  # Very permissive
   ```

3. Check what model predicts:
   Add debug output to detect_strokes.py to see all predictions (even low confidence)

### Problem: Training loss not decreasing

**Possible causes:**
1. Learning rate too high
2. Model architecture wrong
3. Data quality issues

**Solutions:**
1. Lower learning rate:
   ```python
   'learning_rate': 0.0001,
   ```

2. Check training curves in MLflow artifacts:
   - If loss jumps around → LR too high
   - If loss flat → LR too low or data issue

### Problem: Overfitting (train=95%, test=60%)

**Possible causes:**
1. Not enough data
2. Model too complex
3. Need regularization

**Solutions:**
1. Increase overlap (more training samples):
   ```python
   'overlap': 25,  # More augmentation
   ```

2. Already have dropout in model (good!)

3. Collect more diverse training videos

---

## Expected Results

### With Current Data (~884 samples)

**5 classes (current):**
- Best achievable: ~60% accuracy
- Problem: slice classes have <20 samples

**2 classes (grouped):**
- Best achievable: ~75-85% accuracy
- fronthand: ~440 samples (421 + 18 slice direita)
- backhand: ~370 samples (347 + 25 slice esquerda)

### With More Data (target)

**2 classes, 100+ samples each:**
- Target: >85% accuracy
- Need: ~1000 total sequences

---

## Summary

### Immediate Actions

1. ✅ **Class grouping enabled** - Should improve accuracy significantly
2. ✅ **Detection thresholds lowered** - Should detect more strokes

### Next Steps

1. **Train with grouped classes:**
   ```bash
   poetry run python src/train_model.py
   ```

2. **Check improvement:**
   - Compare old accuracy (54%) vs new accuracy (should be >70%)
   - Look at classification report (both classes should have good precision)

3. **Test on real video:**
   ```bash
   poetry run python src/detect_strokes.py your_video.mp4
   ```

4. **If still poor, tune window_size:**
   - Try 45 frames
   - Compare in MLflow

5. **If accuracy good but detection poor:**
   - Lower confidence_threshold in detect_strokes.py
   - Check if test videos are similar to training videos

### Long-term

- **Annotate more videos** (target: 1000 total sequences)
- **Use automated tuning** for systematic optimization
- **Focus on 2 classes only** (fronthand, backhand)

Good luck! 🎾
