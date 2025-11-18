# Feature Selection Guide: Using Only Relevant Pose Landmarks

## Overview

This guide explains how we select only the **most relevant pose landmarks** for tennis stroke classification, reducing from 132 features to **60 features** (85% reduction!).

---

## Why Reduce Features?

### Problems with Using All 132 Features:

1. **Irrelevant features add noise:**
   - Face details (eyes, ears, mouth) don't help classify strokes
   - Foot details (heel, foot index) are less important than ankles
   - More features = more parameters = easier to overfit with small datasets

2. **Computational cost:**
   - 132 features × 30 frames = 3,960 inputs per sequence
   - 60 features × 30 frames = 1,800 inputs (54% reduction!)
   - Faster training, less memory

3. **Better generalization:**
   - Focus on biomechanically relevant landmarks
   - Model learns patterns that actually matter for strokes

---

## MediaPipe Pose Landmarks (All 33)

```
HEAD/FACE (11 landmarks): 0-10
  0  - Nose
  1  - Left eye (inner)
  2  - Left eye
  3  - Left eye (outer)
  4  - Right eye (inner)
  5  - Right eye
  6  - Right eye (outer)
  7  - Left ear
  8  - Right ear
  9  - Mouth (left)
  10 - Mouth (right)

UPPER BODY (6 landmarks): 11-16
  11 - Left shoulder    ← CRITICAL for stroke rotation
  12 - Right shoulder   ← CRITICAL for stroke rotation
  13 - Left elbow       ← CRITICAL for backhand mechanics
  14 - Right elbow      ← CRITICAL for fronthand mechanics
  15 - Left wrist       ← CRITICAL for backhand trajectory
  16 - Right wrist      ← CRITICAL for fronthand trajectory

HANDS (6 landmarks): 17-22
  17 - Left pinky
  18 - Right pinky
  19 - Left index
  20 - Right index
  21 - Left thumb
  22 - Right thumb

LOWER BODY (6 landmarks): 23-28
  23 - Left hip         ← CRITICAL for body rotation
  24 - Right hip        ← CRITICAL for body rotation
  25 - Left knee        ← Important for weight transfer
  26 - Right knee       ← Important for weight transfer
  27 - Left ankle       ← Important for stance
  28 - Right ankle      ← Important for stance

FEET (4 landmarks): 29-32
  29 - Left heel
  30 - Right heel
  31 - Left foot index
  32 - Right foot index
```

---

## Selected Landmarks (15 total)

### **What We're Keeping:**

```python
SELECTED_LANDMARKS = [
    # Head (1 landmark) - Track head orientation
    0,   # Nose (represents head position/direction)

    # Upper body (6 landmarks) - Critical for stroke mechanics
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist

    # Lower body (8 landmarks) - Body rotation and stance
    23,  # Left hip
    24,  # Right hip
    25,  # Left knee
    26,  # Right knee
    27,  # Left ankle
    28,  # Right ankle
    29,  # Left heel (for complete stance)
    30,  # Right heel (for complete stance)
]
```

**Total features:** 15 landmarks × 4 values = **60 features** (down from 132)

### **What We're Removing and Why:**

```
❌ REMOVED: Eyes (1, 2, 3, 4, 5, 6)
   Reason: Eye position doesn't affect stroke classification
   Alternative: Nose (0) is sufficient for head orientation

❌ REMOVED: Ears (7, 8)
   Reason: Similar to eyes, not biomechanically relevant
   Alternative: Nose captures head direction

❌ REMOVED: Mouth (9, 10)
   Reason: Facial details irrelevant for stroke classification

❌ REMOVED: Hand details (17, 18, 19, 20, 21, 22)
   Reason: Wrist position (15, 16) is sufficient
   Note: If grip matters, we could add index finger (19, 20)

❌ REMOVED: Foot indices (31, 32)
   Reason: Ankle (27, 28) + heel (29, 30) capture foot position
   Note: Foot detail not critical for upper-body strokes
```

---

## Biomechanical Justification

### What Each Landmark Group Tells Us:

**Head (Nose - landmark 0):**
```
Tells us:
  - Is player facing camera or sideways?
  - Head rotation during stroke
  - Body orientation

Example:
  Fronthand: Nose moves from right → left (follows body rotation)
  Backhand:  Nose moves from left → right (opposite rotation)
```

**Shoulders (11, 12):**
```
Tells us:
  - Torso rotation angle
  - Shoulder line orientation
  - Power generation from core

Example:
  Fronthand: Right shoulder (12) rotates forward (x increases)
  Backhand:  Left shoulder (11) rotates forward (x increases)
```

**Elbows (13, 14):**
```
Tells us:
  - Arm bend/extension
  - Stroke mechanics (bent vs extended)
  - Power transfer through kinetic chain

Example:
  Fronthand: Right elbow (14) extends during forward swing (z increases)
  Backhand:  Left elbow (13) extends during forward swing
```

**Wrists (15, 16):**
```
Tells us:
  - Racket trajectory (wrist follows racket)
  - Stroke path (left→right vs right→left)
  - Contact point location

Example:
  Fronthand: Right wrist (16) moves left → right (x: 0.3 → 0.7)
  Backhand:  Left wrist (15) moves right → left (x: 0.7 → 0.3)
```

**Hips (23, 24):**
```
Tells us:
  - Hip rotation (power source)
  - Weight transfer direction
  - Lower body engagement

Example:
  Fronthand: Right hip forward (hip line rotates clockwise)
  Backhand:  Left hip forward (hip line rotates counter-clockwise)
```

**Knees (25, 26):**
```
Tells us:
  - Leg bend (loading/exploding)
  - Weight distribution
  - Balance and stability

Example:
  During stroke: Knee bend decreases (player extends upward)
```

**Ankles/Heels (27, 28, 29, 30):**
```
Tells us:
  - Foot placement
  - Weight transfer (left foot → right foot)
  - Stance stability

Example:
  Fronthand: Weight shifts from left ankle → right ankle
  Backhand:  Weight shifts from right ankle → left ankle
```

---

## Feature Vector Comparison

### Before (All Landmarks):
```
Frame features: [132 values]

[nose_x, nose_y, nose_z, nose_vis,           # 0
 left_eye_inner_x, ..., left_eye_inner_vis,  # 1
 left_eye_x, ..., left_eye_vis,              # 2
 ... (all 33 landmarks)
 right_foot_index_x, ..., right_foot_index_vis]  # 32

Total: 33 landmarks × 4 values = 132 features
```

### After (Selected Landmarks):
```
Frame features: [60 values]

[nose_x, nose_y, nose_z, nose_vis,           # 0 - Head
 left_shoulder_x, ..., left_shoulder_vis,    # 11
 right_shoulder_x, ..., right_shoulder_vis,  # 12
 left_elbow_x, ..., left_elbow_vis,          # 13
 right_elbow_x, ..., right_elbow_vis,        # 14
 left_wrist_x, ..., left_wrist_vis,          # 15
 right_wrist_x, ..., right_wrist_vis,        # 16
 left_hip_x, ..., left_hip_vis,              # 23
 right_hip_x, ..., right_hip_vis,            # 24
 left_knee_x, ..., left_knee_vis,            # 25
 right_knee_x, ..., right_knee_vis,          # 26
 left_ankle_x, ..., left_ankle_vis,          # 27
 right_ankle_x, ..., right_ankle_vis,        # 28
 left_heel_x, ..., left_heel_vis,            # 29
 right_heel_x, ..., right_heel_vis]          # 30

Total: 15 landmarks × 4 values = 60 features
```

---

## Expected Benefits

### 1. Reduced Overfitting
```
Before: 132 features → Model can memorize noise
After:  60 features → Model focuses on relevant patterns

With small datasets (<500 sequences), this is CRITICAL
```

### 2. Faster Training
```
Before: 3,960 inputs per sequence (132 × 30 frames)
After:  1,800 inputs per sequence (60 × 30 frames)

Expected: ~2x faster training
```

### 3. Better Generalization
```
Model learns from biomechanically meaningful features:
- Shoulder rotation (power generation)
- Hip rotation (core engagement)
- Arm extension (stroke mechanics)
- Weight transfer (footwork)

Not distracted by:
- Eye positions
- Mouth movements
- Exact finger positions
```

### 4. Easier Interpretation
```
When debugging, you can focus on:
"Is the model looking at shoulder rotation?"
"Does it detect wrist trajectory?"

Instead of:
"Why does left eye inner position affect classification??"
```

---

## Implementation Details

### Code Changes Required:

**1. Update `PoseExtractor.extract_landmarks()` to filter landmarks:**

```python
SELECTED_LANDMARKS = [
    0,   # Nose (head)
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists
    23, 24,  # Hips
    25, 26,  # Knees
    27, 28,  # Ankles
    29, 30,  # Heels
]

def extract_landmarks(self, image):
    """Extract only selected landmarks (60 features instead of 132)"""
    results = self.pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = []
        for idx in SELECTED_LANDMARKS:  # Only selected landmarks
            lm = results.pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks)  # Length: 60
    else:
        return np.zeros(len(SELECTED_LANDMARKS) * 4)  # 60 zeros
```

**2. Update `detect_strokes.py` with same filtering:**

The detection script must use the SAME landmark selection to match training.

**3. Update CONFIG:**

```python
CONFIG = {
    'num_landmarks': 15,  # Down from 33
    'input_features': 60,  # Down from 132
}
```

**4. Model architecture automatically adapts:**

```python
# Input shape: (window_size, 60) instead of (window_size, 132)
model = build_model(
    input_shape=(CONFIG['window_size'], 60),  # Automatically uses 60 features
    num_classes=len(label_encoder.classes_)
)
```

---

## Alternative Configurations

### Minimal (Upper body only) - 32 features:
```python
MINIMAL_LANDMARKS = [
    0,   # Nose
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists
]
# 8 landmarks × 4 = 32 features
# Use if: Dataset is VERY small (<200 sequences)
```

### Moderate (Current recommendation) - 60 features:
```python
MODERATE_LANDMARKS = [
    0,   # Nose
    11, 12, 13, 14, 15, 16,  # Upper body
    23, 24, 25, 26, 27, 28, 29, 30,  # Lower body
]
# 15 landmarks × 4 = 60 features
# Use if: Dataset is small-medium (200-1000 sequences)
```

### Extended (Add hand detail) - 68 features:
```python
EXTENDED_LANDMARKS = [
    0,   # Nose
    11, 12, 13, 14, 15, 16,  # Upper body
    19, 20,  # Index fingers (grip orientation)
    23, 24, 25, 26, 27, 28, 29, 30,  # Lower body
]
# 17 landmarks × 4 = 68 features
# Use if: Grip/hand position matters & dataset is large (>1000 sequences)
```

---

## Validation Strategy

### How to Test If Feature Selection Helps:

1. **Train baseline (all 132 features):**
   ```bash
   poetry run python src/train_model.py
   ```
   Note the test accuracy and per-class recall.

2. **Train with selected features (60 features):**
   ```bash
   # After implementing changes
   poetry run python src/train_model.py
   ```

3. **Compare in MLflow:**
   ```bash
   mlflow ui
   ```
   Look for:
   - ✅ Higher test accuracy
   - ✅ More balanced recall (backhand vs fronthand)
   - ✅ Faster training time
   - ✅ Lower overfitting (smaller gap between train/val accuracy)

### Expected Results:

```
Before (132 features):
  Test accuracy: 61%
  Backhand recall: 31%
  Fronthand recall: 83%
  Training time: ~5min

After (60 features):
  Test accuracy: 65-70% (expected improvement)
  Backhand recall: 50-60% (better balance)
  Fronthand recall: 75-80% (slight decrease OK)
  Training time: ~2-3min (faster)
```

---

## Summary

**Selected landmarks (15 total):**
- Head: Nose (1)
- Upper body: Shoulders, elbows, wrists (6)
- Lower body: Hips, knees, ankles, heels (8)

**Benefits:**
- 60 features instead of 132 (54% reduction)
- Focus on biomechanically relevant features
- Reduce overfitting with small datasets
- Faster training (2x speedup expected)
- Better generalization

**Trade-offs:**
- Lose facial detail (not needed for strokes)
- Lose hand detail (wrist position sufficient)
- Lose foot detail (ankle + heel sufficient)

**Next step:** Implement the feature selection in `train_model.py` and `detect_strokes.py`.
