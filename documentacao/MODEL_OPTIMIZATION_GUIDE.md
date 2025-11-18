# Model Optimization Guide

This guide covers two key strategies for improving model performance: **balancing class weights** and **selecting relevant features**.

---

## Table of Contents

1. [Class Balancing with Weights](#class-balancing-with-weights)
2. [Feature Selection](#feature-selection)
3. [Combined Workflow](#combined-workflow)
4. [Validation Strategies](#validation-strategies)

---

## Class Balancing with Weights

### The Problem: Class Imbalance

**Symptoms:**
```
Confusion Matrix:
                Predicted
                Backhand  Fronthand
Actual Backhand    38        86       ← Only 31% of backhands detected!
       Fronthand   26       127       ← 83% of fronthands detected

Overall accuracy: 61%
```

**Problem:** Model is biased toward predicting FRONTHAND because:
- Training data likely has more fronthand examples
- Model learns: "Safest strategy = predict fronthand more often"
- Result: Good at fronthand (83%), terrible at backhand (31%)

### Why Does This Happen?

**Example imbalanced dataset:**
```
Training data:
  Fronthand: 1000 sequences (70%)
  Backhand:   430 sequences (30%)

Model's "lazy" strategy:
  Always predict "fronthand"
  → Accuracy = 1000/1430 = 70%!
  → Without learning anything meaningful!
```

The model minimizes overall loss, and predicting the majority class achieves low loss easily.

### The Solution: Class Weights

**Class weights make mistakes on minority class MORE EXPENSIVE:**

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate balanced weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Formula: n_samples / (n_classes * n_samples_for_class)
```

### Example Calculation

```python
# Your data (hypothetical):
y_train = [
    0, 0, 0, ...,  # 430 backhand samples
    1, 1, 1, ...,  # 1000 fronthand samples
]

# Total: 1430 samples
# Classes: 2 (backhand=0, fronthand=1)

# Backhand weight:
# 1430 / (2 * 430) = 1430 / 860 = 1.66

# Fronthand weight:
# 1430 / (2 * 1000) = 1430 / 2000 = 0.72

class_weights = {
    0: 1.66,  # Backhand (minority) - Higher weight!
    1: 0.72   # Fronthand (majority) - Lower weight
}
```

### How It Works During Training

**Without class weights:**
```
Loss calculation per sample:
  Misclassify backhand  → Loss = 1.0
  Misclassify fronthand → Loss = 1.0

Batch loss:
  86 backhand errors  → Total loss = 86 × 1.0 = 86
  26 fronthand errors → Total loss = 26 × 1.0 = 26
  Combined: 112

Gradient update: Focuses more on reducing fronthand errors
```

**With class weights:**
```
Loss calculation per sample:
  Misclassify backhand  → Loss = 1.0 × 1.66 = 1.66 (OUCH!)
  Misclassify fronthand → Loss = 1.0 × 0.72 = 0.72 (not as bad)

Batch loss:
  86 backhand errors  → Total loss = 86 × 1.66 = 143
  26 fronthand errors → Total loss = 26 × 0.72 = 19
  Combined: 162

Gradient update: Focuses heavily on reducing backhand errors!
```

**Result:** Model learns to pay equal attention to both classes.

### Implementation

**Step 1: Calculate Class Weights**

Location: [src/train_model.py:642-657](src/train_model.py#L642-L657)

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate balanced weights based on training data distribution
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convert to dictionary format for Keras
class_weights = dict(enumerate(class_weights_array))

# Display weights
print(f"\n{'='*70}")
print("CLASS WEIGHTS (to balance training)")
print(f"{'='*70}")
for idx, weight in class_weights.items():
    class_name = label_encoder.classes_[idx]
    class_count = np.sum(y_train == idx)
    print(f"  {class_name}: {weight:.3f} (n={class_count})")
```

**Example output:**
```
======================================================================
CLASS WEIGHTS (to balance training)
======================================================================
  backhand: 1.663 (n=430)
  fronthand: 0.715 (n=1000)
```

**Step 2: Apply During Training**

Location: [src/train_model.py:680](src/train_model.py#L680)

```python
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    class_weight=class_weights,  # ← CRITICAL: Apply weights here!
    callbacks=callbacks,
    verbose=1
)
```

### Expected Results

**Before class weights:**
```
Confusion Matrix:
                Predicted
                Backhand  Fronthand
Actual Backhand    38        86       ← 31% recall
       Fronthand   26       127       ← 83% recall
```

**After class weights:**
```
Confusion Matrix (expected):
                Predicted
                Backhand  Fronthand
Actual Backhand    70        54       ← 56% recall (much better!)
       Fronthand   40       113       ← 74% recall (slight decrease OK)
```

**Key improvements:**
- Backhand recall: 31% → 50-60% ✅
- Fronthand recall: 83% → 70-75% (acceptable trade-off)
- More balanced predictions overall

---

## Feature Selection

### Why Reduce Features?

**Problems with using all 132 features:**

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

### MediaPipe Pose Landmarks (All 33)

```
HEAD/FACE (11 landmarks): 0-10
  0  - Nose
  1-6  - Eyes (inner, outer, left, right)
  7-8  - Ears
  9-10 - Mouth

UPPER BODY (6 landmarks): 11-16
  11 - Left shoulder    ← CRITICAL for stroke rotation
  12 - Right shoulder   ← CRITICAL for stroke rotation
  13 - Left elbow       ← CRITICAL for backhand mechanics
  14 - Right elbow      ← CRITICAL for fronthand mechanics
  15 - Left wrist       ← CRITICAL for backhand trajectory
  16 - Right wrist      ← CRITICAL for fronthand trajectory

HANDS (6 landmarks): 17-22
  17-22 - Fingers (pinky, index, thumb)

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

### Selected Landmarks (15 total)

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

### What We're Removing and Why

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

❌ REMOVED: Foot indices (31, 32)
   Reason: Ankle (27, 28) + heel (29, 30) capture foot position
```

### Biomechanical Justification

**Head (Nose - landmark 0):**
```
Tells us:
  - Is player facing camera or sideways?
  - Head rotation during stroke
  - Body orientation
```

**Shoulders (11, 12):**
```
Tells us:
  - Torso rotation angle
  - Shoulder line orientation
  - Power generation from core
```

**Wrists (15, 16):**
```
Tells us:
  - Racket trajectory (wrist follows racket)
  - Stroke path (left→right vs right→left)
  - Contact point location
```

**Hips (23, 24):**
```
Tells us:
  - Hip rotation (power source)
  - Weight transfer direction
  - Lower body engagement
```

### Expected Benefits

1. **Reduced Overfitting:**
   - Model focuses on relevant patterns
   - Less memorization of noise

2. **Faster Training:**
   - 3,960 → 1,800 inputs per sequence
   - ~2x faster training

3. **Better Generalization:**
   - Learns biomechanically meaningful features
   - Not distracted by irrelevant details

4. **Easier Interpretation:**
   - Can focus on: "Is the model looking at shoulder rotation?"
   - Instead of: "Why does left eye position matter??"

### Implementation

**Update `PoseExtractor.extract_landmarks()` to filter landmarks:**

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

---

## Combined Workflow

### Step 1: Balance Your Data

```python
# In train_model.py - Already implemented!
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
```

**This fixes:** Class imbalance (one class dominates predictions)

### Step 2: Select Your Features

```python
# In train_model.py and detect_strokes.py
SELECTED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
```

**This fixes:** Overfitting, slow training, noise from irrelevant features

### Step 3: Train Better Model

```bash
poetry run python src/train_model.py
```

**Expected improvements:**
- More balanced class predictions
- Faster training time
- Better generalization
- Higher F1-scores

### Step 4: Validate Results

```bash
# View in MLflow
mlflow ui
```

**Check:**
- Both classes have similar recall (balanced)
- F1-scores improved
- Training time decreased

---

## Validation Strategies

### Compare Before & After

#### Without Optimization (Baseline):
```
Test accuracy: 61%
Backhand recall: 31%
Fronthand recall: 83%
Training time: ~5min
Features: 132
```

#### With Class Weights Only:
```
Test accuracy: 68%
Backhand recall: 56%
Fronthand recall: 74%
Training time: ~5min
Features: 132
```

#### With Feature Selection Only:
```
Test accuracy: 65%
Backhand recall: 40%
Fronthand recall: 80%
Training time: ~2-3min
Features: 60
```

#### With Both (Optimal):
```
Test accuracy: 70-75%
Backhand recall: 60-70%
Fronthand recall: 75-80%
Training time: ~2-3min
Features: 60
```

### Use MLflow to Compare

1. Train baseline (all features, no weights)
2. Train with class weights only
3. Train with feature selection only
4. Train with both optimizations

Then in MLflow UI:
- Select all 4 runs
- Click "Compare"
- Sort by `macro_avg_f1_score`
- Check confusion matrices in Artifacts

---

## Troubleshooting

### Class Weights Not Helping

**Possible causes:**
1. **Extreme imbalance (>90/10):**
   - Solution: Collect more minority class data

2. **Weights too conservative:**
   ```python
   # Manually increase minority weight
   class_weights[0] = class_weights[0] * 1.5
   ```

3. **Other issues:**
   - Check annotation quality
   - Verify features capture class differences

### Feature Selection Not Helping

**Possible causes:**
1. **Need more features:**
   - Try extended set (add index fingers for grip)

2. **Need fewer features:**
   - Try minimal set (upper body only)

3. **Wrong landmarks selected:**
   - Review which landmarks are most important for your strokes

### Training Still Unstable

**Try:**
1. **Reduce learning rate:**
   ```python
   'learning_rate': 0.0001
   ```

2. **Cap maximum weight:**
   ```python
   max_weight = 2.0
   class_weights = {k: min(v, max_weight) for k, v in class_weights.items()}
   ```

---

## Summary

### Key Optimizations

1. **Class Balancing:**
   - ✅ Fixes: Imbalanced predictions
   - ✅ How: Weight minority class higher in loss calculation
   - ✅ Result: Balanced recall across all classes

2. **Feature Selection:**
   - ✅ Fixes: Overfitting, slow training, noise
   - ✅ How: Use only biomechanically relevant landmarks
   - ✅ Result: Faster training, better generalization

### Combined Benefits

- More balanced class predictions
- Faster training (2x speedup)
- Better generalization
- Easier to interpret
- Higher F1-scores

### Next Steps

1. Implement both optimizations
2. Train and compare in MLflow
3. Validate on real videos
4. Iterate if needed

Happy optimizing! 🎾
