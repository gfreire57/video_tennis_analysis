# Sequence Creation Explained

## Overview

This document explains **how training sequences are created** from your annotated videos and why you might get 0 sequences.

---

## The Problem You're Experiencing

**Your output:**
```
Created 0 sequences
```

**Why this happens:**
Your stroke annotations (13-16 frames) are **too short** for the window size (40 frames).

---

## How Sequence Creation Works

### Step 1: Frame Labeling

**Code location:** [src/train_model.py:248-269](src/train_model.py#L248-L269)

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

### Step 2: Sliding Window

**Code location:** [src/train_model.py:271-296](src/train_model.py#L271-L296)

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

### Step 3: Majority Voting

**Code location:** [src/train_model.py:277-285](src/train_model.py#L277-L285)

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

### Step 4: Filter Windows

**Code location:** [src/train_model.py:287-292](src/train_model.py#L287-L292)

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

---

## Why You Get 0 Sequences

### Your Annotations vs Window Size

**Your annotations:**
```
- backhand: frames 622-635 (13 frames)
- backhand: frames 674-688 (14 frames)
- backhand: frames 701-717 (16 frames)
...
```

**Window size:** 40 frames

**Problem:** Your annotations are **too short**!

### Example Calculation

Let's check if Window starting at frame 600 would be accepted:

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

---

## Solutions

### Option 1: Reduce Window Size (RECOMMENDED)

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

### Option 2: Extend Your Annotations

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

### Option 3: Lower Majority Threshold

**Modify the code to accept windows with <50% coverage:**

```python
# Current (line 290)
if majority_count > window_size * 0.5 and majority_label != 'neutral':

# Modified (accept >30% coverage)
if majority_count > window_size * 0.3 and majority_label != 'neutral':
```

**Risk:** You'll get noisier training data (windows with more neutral frames).

### Option 4: Use Variable-Length Sequences

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

---

## Recommended Action Plan

### Quick Fix (5 minutes)

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

### Better Solution (30 minutes)

1. **Re-annotate your videos in Label Studio**
2. **Extend each stroke annotation to ~40-50 frames:**
   - Start annotation when player begins preparation
   - End annotation after follow-through completes
3. **Keep window_size=40**

**Benefits:**
- Better temporal context for LSTM
- Model learns complete stroke motion (not just contact point)
- More robust to variations in stroke execution

---

## Understanding Window Size Selection

### How to Choose Window Size

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

## Diagnostic Questions

### How to Check if Window Size is Correct

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

### How to Verify Sequences Will Be Created

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

---

## Code References

- **Frame labeling:** [src/train_model.py:248-269](src/train_model.py#L248-L269)
- **Sliding window:** [src/train_model.py:271-296](src/train_model.py#L271-L296)
- **Majority voting:** [src/train_model.py:277-285](src/train_model.py#L277-L285)
- **Filtering logic:** [src/train_model.py:287-292](src/train_model.py#L287-L292)
- **CONFIG:** [src/train_model.py:72-86](src/train_model.py#L72-L86)

---

## Summary

**Your issue:**
```
window_size (40) > annotation_length (13-16)
  → All windows have 'neutral' as majority
  → All windows rejected
  → 0 sequences created
```

**Solution:**
```
Option 1: window_size (15) < annotation_length (13-16)  ← QUICK FIX
Option 2: annotation_length (40-50) > window_size (40)  ← BETTER LONG-TERM
```

**Next steps:**
1. Change `window_size: 15` in CONFIG
2. Re-run training
3. You should see sequences created!

Alternatively:
1. Extend annotations to 40-50 frames in Label Studio
2. Keep `window_size: 40`
3. Better temporal context for model
