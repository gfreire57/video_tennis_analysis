# Model Optimization - Comprehensive Guide

## Overview of Optimization Strategies

This guide consolidates strategies for improving model performance across multiple dimensions:

1. **Class Balancing** - Addressing imbalanced training data where one class dominates predictions
2. **Feature/Landmark Selection** - Reducing input dimensionality by selecting biomechanically relevant features
3. **Architecture Improvements** - Enhancing model capacity and learning stability through BatchNormalization, Dropout, and layer design
4. **Bidirectional LSTM** - Implementing bidirectional sequence processing for improved temporal context
5. **Hyperparameter Tuning** - Optimizing learning rate, batch size, window size, and overlap for better convergence
6. **Integration Strategy** - When and how to apply each optimization for maximum impact

These optimizations work synergistically - combining multiple strategies typically yields better results than any single approach.

---

## 1. Class Balancing with Weights

### The Problem: Class Imbalance

**Symptoms of class imbalance:**
```
Confusion Matrix:
                Predicted
                Backhand  Fronthand
Actual Backhand    38        86       ← Only 31% of backhands detected!
       Fronthand   26       127       ← 83% of fronthands detected

Overall accuracy: 61%
```

**Why this happens:**
- Training data likely has more fronthand examples than backhand
- Model learns: "Safest strategy = predict fronthand more often"
- Result: Good at fronthand (83%), terrible at backhand (31%)

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

Location: `src/train_model.py:642-657`

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

Location: `src/train_model.py:680`

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

## 2. Feature Selection / Landmark Selection

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

## 3. Architecture Improvements

### Why Improve Architecture?

Class imbalance and feature selection are critical, but architecture also plays a role in:
- Stabilizing learning (BatchNormalization)
- Preventing overfitting (Dropout)
- Improving feature extraction (additional capacity)
- Better gradient flow (depth)

### Old vs New Architecture

**Old architecture:**
```python
LSTM(128) → Dropout → LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(32) → Dense(num_classes)
```

**New architecture (with improvements):**
```python
LSTM(128) → BatchNorm → Dropout(0.4) →
LSTM(96)  → BatchNorm → Dropout(0.4) →
LSTM(64)  → BatchNorm → Dropout(0.3) →
Dense(64) → BatchNorm → Dropout(0.3) →
Dense(32) → BatchNorm → Dropout(0.2) →
Dense(num_classes)
```

### Key Improvements

#### BatchNormalization

**What it does:**
- Normalizes activations between layers
- Prevents "covariate shift" (changing distributions during training)
- Acts as mild regularization
- Allows higher learning rates
- Improves gradient flow through deep networks

**Benefits:**
- More stable training
- Faster convergence
- Better generalization

#### Increased Capacity

**Changes:**
- LSTM layers: 128 → 96 → 64 (better feature extraction than 128 → 64 → 32)
- Dense layers: Now 64 → 32 instead of just 32
- More intermediate representation space

**Why:**
- Better feature discrimination between classes
- More capacity to learn complex patterns
- Prevents early bottlenecking

#### Improved Dropout Strategy

**Changes:**
- LSTM layers: 0.4 dropout (was variable)
- Dense layers: 0.3, 0.2 dropout (graduated dropout)
- Higher dropout in LSTM layers (prevents overfitting)

**Why:**
- Higher dropout (0.4) in LSTM layers prevents model from relying on minority class patterns
- Graduated dropout (decreasing towards output) preserves information for final classification
- More regularization overall prevents overfitting to training data

### Expected Improvement

Better feature discrimination between classes, enabling class weights to work more effectively.

---

## 4. Bidirectional LSTM

### What is Bidirectional LSTM?

A Bidirectional LSTM processes the input sequence twice:
1. **Forward direction**: frame 1 → frame 2 → ... → frame N
2. **Backward direction**: frame N → frame N-1 → ... → frame 1

The outputs from both directions are concatenated, **doubling the output units**.

**Example**:
- Regular LSTM with 64 units → Output: 64 units
- Bidirectional LSTM with 64 units → Output: 128 units (64×2)

### When to Use Bidirectional LSTM?

#### Potential Benefits for Tennis Stroke Recognition:

✅ **Better temporal context**: The model can see what happens both before and after each frame
✅ **Improved stroke detection**: Beginning and end of strokes may be better recognized
✅ **Higher accuracy**: Studies show 2-3% F1-score improvement on average

#### Trade-offs:

⚠️ **Slower training**: ~2x slower because sequences are processed twice
⚠️ **More parameters**: Double the units means more memory usage
⚠️ **Not always better**: May overfit on small datasets

### How to Enable/Disable

#### Option 1: Single Training Run

Edit `CONFIG` in `src/train_model.py`:

```python
CONFIG = {
    # ... other settings ...
    'use_bidirectional': True,  # Enable Bidirectional on first layer
    # ... other settings ...
}
```

Then train:
```bash
poetry run python src/train_model.py
```

#### Option 2: Grid Search Comparison

Use the `MINIMAL_TEST` grid in `src/grid_search_configs.py` which tests both modes:

```python
MINIMAL_TEST = {
    'lstm_layers': [[128, 64], [64, 32]],
    'dense_units': [64],
    'dropout_rates': [[0.3, 0.3, 0.2]],
    'use_batch_norm': [False],
    'use_bidirectional': [False, True],  # ← Tests both!
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32],
    'window_size': [45],
    'overlap': [15],
}
```

Run grid search:
```bash
poetry run python src/grid_search.py --grid minimal
```

This will run **8 experiments** (2 lstm_layers × 2 use_bidirectional × 2 learning_rate = 8 runs).

### Implementation Details

#### Architecture (V1 - Current Active)

**Without Bidirectional** (`use_bidirectional=False`):
```
Input (45, 60)
├─ LSTM(64, return_sequences=True)  → Output: (45, 64)
├─ Dropout(0.4)
├─ LSTM(128, return_sequences=True) → Output: (45, 128)
├─ Dropout(0.4)
├─ LSTM(64, return_sequences=False) → Output: (64)
├─ Dropout(0.3)
├─ Dense(64, relu)
├─ Dropout(0.3)
├─ Dense(64, relu)
├─ Dropout(0.2)
└─ Dense(num_classes, softmax)
```

**With Bidirectional** (`use_bidirectional=True`):
```
Input (45, 60)
├─ Bidirectional(LSTM(64, return_sequences=True))  → Output: (45, 128) ← DOUBLED!
├─ Dropout(0.4)
├─ LSTM(128, return_sequences=True)                → Output: (45, 128)
├─ Dropout(0.4)
├─ LSTM(64, return_sequences=False)                → Output: (64)
├─ Dropout(0.3)
├─ Dense(64, relu)
├─ Dropout(0.3)
├─ Dense(64, relu)
├─ Dropout(0.2)
└─ Dense(num_classes, softmax)
```

**Key difference**: First layer outputs 128 units instead of 64.

### MLflow Tracking

Both `train_model.py` and `grid_search.py` automatically log the Bidirectional setting:

**Logged parameters**:
- `uses_bidirectional`: True/False
- `lstm_layer_1_units`: 64 (regular) or 128 (bidirectional)
- `num_lstm_layers`: 3
- `architecture_summary`: Shows the full architecture

**Compare in MLflow UI**:
```bash
mlflow ui
```

Then filter experiments by `uses_bidirectional` to compare performance.

### Testing Workflow

#### Quick Test (Recommended First)

1. **Enable Bidirectional in train_model.py**:
   ```python
   'use_bidirectional': True,
   ```

2. **Train once**:
   ```bash
   poetry run python src/train_model.py
   ```

3. **Check results in MLflow**:
   - Look at F1-scores, precision, recall
   - Compare training time
   - Check if accuracy improved

#### Systematic Comparison (Grid Search)

1. **Run minimal grid search**:
   ```bash
   poetry run python src/grid_search.py --grid minimal
   ```

2. **Open MLflow UI**:
   ```bash
   mlflow ui
   ```

3. **Compare results**:
   - Filter by `uses_bidirectional = True` vs `False`
   - Check F1-scores, accuracy, training time
   - Look at confusion matrices

### Expected Results

Based on similar sequence classification tasks:

| Metric | Without Bidirectional | With Bidirectional | Change |
|--------|----------------------|-------------------|--------|
| **F1-Score** | ~0.85 | ~0.87-0.88 | +2-3% |
| **Training Time** | 100% | ~150-200% | +50-100% |
| **Parameters** | Baseline | +30-50% | More |

**Note**: Actual results may vary based on your dataset!

### Troubleshooting

#### Out of Memory Error

If you get GPU/memory errors with Bidirectional:

**Solution 1**: Reduce batch size
```python
'batch_size': 16,  # Instead of 32
```

**Solution 2**: Disable Bidirectional
```python
'use_bidirectional': False,
```

#### No Improvement in Accuracy

Possible reasons:
- Dataset too small (Bidirectional may overfit)
- Current architecture already sufficient
- Need more epochs for Bidirectional to converge

**Try**:
- Increase `epochs` to 200
- Add more regularization (higher dropout)
- Collect more training data

---

## 5. Hyperparameter Tuning

### Parameters to Tune

#### 1. Window Size

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

#### 2. Learning Rate

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

#### 3. Batch Size

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

#### 4. Overlap

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

### Recommended Tuning Sequence

#### Step 1: Test Grouped Classes (NOW)

**Run training with grouped classes:**

```bash
poetry run python src/train_model.py
```

**Expected improvement:**
- Before: 5 classes, 54% accuracy, some classes 0%
- After: 2 classes, should be >70% accuracy

**If accuracy is still low (<70%)**, proceed to Step 2.

#### Step 2: Try Larger Window

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

#### Step 3: Try Lower Learning Rate

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

#### Step 4: Test on Real Video

After each training run:

```bash
poetry run python src/detect_strokes.py path/to/your/video.mp4
```

**Evaluate results:**
- How many strokes detected?
- Are they correct?
- Too many false positives? → Increase confidence_threshold in detect_strokes.py
- Too few detections? → Decrease confidence_threshold

### Quick Tuning Experiments

#### Experiment A: Grouped Classes (2 classes)

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

#### Experiment B: Larger Context

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

#### Experiment C: Slower Learning

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

#### Experiment D: More Training Data

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

### Comparing Results in MLflow

#### Start MLflow UI

```bash
poetry run mlflow ui
```

Open: http://localhost:5000

#### Compare Experiments

1. **Select multiple runs** (checkbox on left)
2. **Click "Compare"** button
3. **View comparison:**
   - Parameters tab: See which settings differed
   - Metrics tab: See test_accuracy side-by-side
   - Charts tab: Visualize metric differences

#### Sorting Runs

Click column headers to sort:
- **test_accuracy** (descending) - Find best model
- **Start Time** (descending) - Find recent runs

#### Download Best Model

1. Click on best run (highest test_accuracy)
2. **Artifacts tab**
3. **Download:**
   - Model (if you want to use it directly)
   - Confusion matrix (see which classes confuse the model)
   - Training history plot (check for overfitting)

---

## 6. When and How to Optimize Each Component

### Decision Tree for Optimization

```
START: Train baseline model
  ↓
Q1: Is prediction balanced between classes?
  ├─ NO → Apply CLASS WEIGHTS (Section 1)
  │       Re-train and re-evaluate
  │       ↓
  │    Better? → Continue
  │    Worse? → Revert and try feature selection
  │
  └─ YES → Continue

Q2: Is accuracy still low (<75%)?
  ├─ YES → Apply FEATURE SELECTION (Section 2)
  │        Re-train and re-evaluate
  │        ↓
  │     Better? → Continue
  │     Worse? → Revert and try architecture
  │
  └─ NO → Continue

Q3: Does model seem to overfit (train=95%, test=70%)?
  ├─ YES → Apply ARCHITECTURE IMPROVEMENTS (Section 3)
  │        - Add BatchNormalization
  │        - Adjust dropout rates
  │        - Re-train
  │
  └─ NO → Continue

Q4: Have you reached peak performance for current architecture?
  ├─ YES → Apply BIDIRECTIONAL LSTM (Section 4)
  │        Grid search: with/without
  │        ↓
  │     Better? → Keep it
  │     Worse? → Disable (add to regularization instead)
  │
  └─ NO → Apply HYPERPARAMETER TUNING (Section 5)
         - Try different window sizes
         - Try different learning rates
         - Grid search systematically

DONE: Compare all runs in MLflow, select best
```

### Component Interdependencies

**Class Weights depend on:**
- Feature selection (works better with relevant features)
- Architecture (works better with adequate model capacity)

**Feature Selection depends on:**
- Class weights (both work synergistically)
- Architecture (simpler features need better architecture)

**Architecture Improvements depend on:**
- Feature selection (good features are needed to be learned)
- Class weights (stable training from balanced data)

**Bidirectional LSTM depends on:**
- All of the above (only add after other optimizations)

**Hyperparameter Tuning depends on:**
- All of the above (tune after structural improvements)

### Recommended Optimization Sequence

**Phase 1: Fix Fundamental Issues (Required)**
1. Apply class weights → Ensures balanced learning
2. Apply feature selection → Reduces noise, speeds up training
3. Apply architecture improvements → Stabilizes training

**Phase 2: Enhance Architecture (Optional)**
4. Apply bidirectional LSTM → 2-3% improvement if conditions are right

**Phase 3: Fine-tune Parameters (Always)**
5. Hyperparameter tuning → Optimize for your specific data

### Performance Impact Summary

| Optimization | Impact | Training Time | Implementation Difficulty |
|--------------|--------|----------------|---------------------------|
| **Class Weights** | +10-20% accuracy | No change | Very Easy |
| **Feature Selection** | +5-10% accuracy | -50% faster | Easy |
| **Architecture Improvements** | +5-15% accuracy | +10-20% slower | Medium |
| **Bidirectional LSTM** | +2-3% accuracy | +100% slower | Medium |
| **Hyperparameter Tuning** | +5-10% accuracy | Variable | Hard |

**Combined effect:** Typically 25-45% accuracy improvement when all are applied together.

---

## 7. Complete Optimization Workflow

### Step-by-Step Implementation

#### Phase 1: Class Balancing

**1. Calculate class weights:**

Location: `src/train_model.py:642-657`

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))
```

**2. Apply during training:**

Location: `src/train_model.py:680`

```python
history = model.fit(
    X_train, y_train,
    class_weight=class_weights,  # ← Critical
    ...
)
```

**3. Verify implementation:**

```bash
poetry run python src/train_model.py
# Check for "CLASS WEIGHTS" output
```

#### Phase 2: Feature Selection

**1. Update landmark selection:**

In `src/train_model.py` and `src/extract_poses.py`:

```python
SELECTED_LANDMARKS = [
    0,   # Nose
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists
    23, 24,  # Hips
    25, 26,  # Knees
    27, 28,  # Ankles
    29, 30,  # Heels
]
```

**2. Filter in extraction:**

```python
def extract_landmarks(self, image):
    results = self.pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for idx in SELECTED_LANDMARKS:
            lm = results.pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks)  # 60 features
```

**3. Verify implementation:**

```bash
poetry run python src/train_model.py
# Check for 60 features in model input shape
```

#### Phase 3: Architecture Improvements

**1. Enable BatchNormalization in model:**

```python
model = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(window_size, 60), return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    # ... rest of layers with BatchNorm ...
])
```

**2. Adjust dropout rates:**

```python
# LSTM layers: 0.4 dropout
# Dense layers: 0.3, 0.2 dropout (graduated)
```

**3. Verify implementation:**

```bash
poetry run python src/train_model.py
# Check model summary for BatchNormalization layers
```

#### Phase 4: Bidirectional LSTM (Optional)

**1. Enable in CONFIG:**

```python
CONFIG = {
    'use_bidirectional': True,
}
```

**2. Or run grid search:**

```bash
poetry run python src/grid_search.py --grid minimal
```

**3. Compare in MLflow:**

```bash
mlflow ui
# Filter by uses_bidirectional
```

#### Phase 5: Hyperparameter Tuning

**1. Quick experiments:**

```python
# Experiment 1: window_size=45
CONFIG = {'window_size': 45, 'overlap': 22}

# Experiment 2: learning_rate=0.0005
CONFIG = {'learning_rate': 0.0005}

# Experiment 3: Both
CONFIG = {'window_size': 45, 'overlap': 22, 'learning_rate': 0.0005}
```

**2. Or systematic grid search:**

```bash
poetry run python src/grid_search.py --grid custom
```

**3. Compare in MLflow:**

```bash
mlflow ui
# Click "Compare", select all runs
# Sort by test_accuracy
```

### Combined Workflow Command Sequence

```bash
# Phase 1: Class Balancing
poetry run python src/train_model.py
# Note: test_accuracy, backhand recall

# Phase 2: Feature Selection
# (Update landmarks, same command)
poetry run python src/train_model.py
# Compare: improved? faster?

# Phase 3: Architecture Improvements
# (Update model architecture, same command)
poetry run python src/train_model.py
# Compare: more stable? better accuracy?

# Phase 4: Bidirectional LSTM
poetry run python src/grid_search.py --grid minimal
# MLflow: compare with/without bidirectional

# Phase 5: Hyperparameter Tuning
# Try window_size=45
poetry run python src/train_model.py
# Try learning_rate=0.0005
poetry run python src/train_model.py
# Try both
poetry run python src/train_model.py

# Final: Compare all in MLflow
mlflow ui
# Select best run
```

### Validation Strategy

#### Before & After Comparison

```
BASELINE (Original):
  - Test accuracy: 54%
  - Backhand recall: 16%
  - Fronthand recall: 94%
  - Training time: ~5 minutes

AFTER CLASS WEIGHTS:
  - Test accuracy: 68%
  - Backhand recall: 56%
  - Fronthand recall: 74%
  - Training time: ~5 minutes

AFTER FEATURE SELECTION:
  - Test accuracy: 70%
  - Backhand recall: 58%
  - Fronthand recall: 75%
  - Training time: ~2.5 minutes  ← Faster!

AFTER ARCHITECTURE:
  - Test accuracy: 73%
  - Backhand recall: 62%
  - Fronthand recall: 76%
  - Training time: ~3 minutes

AFTER HYPERPARAMETER TUNING:
  - Test accuracy: 78-85%
  - Backhand recall: 70-80%
  - Fronthand recall: 75-85%
  - Training time: Variable
```

#### Use MLflow to Compare

```bash
mlflow ui
# http://localhost:5000
```

**Steps:**
1. Select all runs (checkboxes on left)
2. Click "Compare" button
3. View comparison:
   - **Parameters**: See which settings differed
   - **Metrics**: See accuracy, precision, recall side-by-side
   - **Charts**: Visualize improvements
4. Click best run
5. **Artifacts**: Download confusion matrix, training history

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

3. **Reduce batch size:**
   ```python
   'batch_size': 16,
   ```

4. **Add gradient clipping:**
   ```python
   optimizer = keras.optimizers.Adam(
       learning_rate=learning_rate,
       clipnorm=1.0
   )
   ```

### Accuracy Still Low After All Optimizations

**Possible causes:**
1. **Not enough training data:**
   - Need 100+ samples per class
   - Currently have ~400 fronthand, ~350 backhand

2. **Annotations are poor quality:**
   - Verify manually on videos
   - Check for inconsistent labeling

3. **Videos too different from each other:**
   - Training: indoor court, good lighting
   - Testing: outdoor, different angles
   - Solution: Collect more diverse data

4. **Classes are inherently similar:**
   - Backhand and fronthand might look similar in your footage
   - Solution: Review stroke definitions with domain expert

### Model Detects Nothing in Real Videos

**Possible causes:**
1. **Model confidence too low:**
   ```python
   # In detect_strokes.py
   'confidence_threshold': 0.3,  # Lowered
   ```

2. **Detection thresholds too strict:**
   ```python
   'min_stroke_duration': 5,  # More permissive
   ```

3. **Model not trained on similar videos:**
   - Include similar angles/lighting in training data

**Solutions:**
- Lower confidence_threshold gradually
- Check model predictions in debug mode
- Retrain on videos more similar to test videos

---

## Summary

### Key Optimizations & Expected Impact

| Optimization | Primary Benefit | When to Apply | Impact |
|--------------|-----------------|---------------|---------|
| **Class Weights** | Fixes prediction imbalance | First | +10-20% |
| **Feature Selection** | Reduces noise, faster training | Early | +5-10% |
| **Architecture** | Better feature learning | After Phase 1-2 | +5-15% |
| **Bidirectional LSTM** | Temporal context | Refinement | +2-3% |
| **Hyperparameter Tuning** | Fine-tuning | Final phase | +5-10% |

### Recommended Implementation Order

1. **Start**: Apply class weights (easiest, biggest impact)
2. **Next**: Apply feature selection (easy, good speedup)
3. **Then**: Improve architecture (more complex, significant gains)
4. **Optional**: Add bidirectional LSTM (if resources allow)
5. **Finally**: Tune hyperparameters (time-consuming, fine-tuning)

### Expected Final Results

**Baseline:** 54% accuracy, 16% backhand recall
**After optimization:** 78-85% accuracy, 70-80% backhand recall

**That's a 30-45% absolute improvement!**

### Next Steps

1. Implement class weights → Test
2. Implement feature selection → Test
3. Update architecture → Test
4. Consider bidirectional LSTM → Test
5. Grid search hyperparameters → Compare in MLflow
6. Select best model

Happy optimizing! 🎾

---

## See Also

Related Documentation:
- **GRID_SEARCH_GUIDE.md** - Automated hyperparameter optimization
- **GRID_SEARCH_README.md** - Grid search basics and configuration
- **MLFLOW_ENHANCED_LOGGING.md** - Tracking and comparing experiments
- **POSE_EXTRACTION_README.md** - Pose detection and landmark extraction
- **DISABLING_FPS_SCALING.md** - Video processing configuration

Code Files Referenced:
- `src/train_model.py` - Main training script with CONFIG
- `src/grid_search.py` - Automated parameter search
- `src/grid_search_configs.py` - Grid search configurations
- `src/detect_strokes.py` - Inference and stroke detection
- `src/extract_poses.py` - Pose landmark extraction
