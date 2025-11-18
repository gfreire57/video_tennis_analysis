# Model Improvements - Fixing Class Imbalance

## Problem Identified

Your training results showed **severe class imbalance in predictions**:

```
Confusion Matrix:
[[12 63]  ← Backhand: Only 12 correct, 63 misclassified as fronthand
 [ 5 83]] ← Fronthand: 83 correct, 5 misclassified

Precision/Recall:
- Backhand:  71% precision, but only 16% recall (barely detected!)
- Fronthand: 57% precision, 94% recall (over-detected!)
```

**Translation:** The model predicts fronthand 88% of the time, regardless of actual stroke type.

This explains the "absurd" overlapping predictions in your videos - the model is biased toward fronthand.

---

## Root Causes

1. **Class imbalance in training data** - Likely more fronthand examples than backhand
2. **No class weighting** - Model wasn't penalized for ignoring minority class
3. **Insufficient model capacity** - Architecture couldn't discriminate well enough
4. **Insufficient training** - 100 epochs may not be enough for convergence

---

## Improvements Made

### 1. Class Weights (Critical Fix)

**Added automatic class weight calculation:**

```python
# Calculate balanced weights based on class frequency
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

# Apply during training
history = model.fit(
    X_train, y_train,
    class_weight=class_weights,  # Forces model to learn both classes equally
    ...
)
```

**What this does:**
- If you have 200 fronthand samples and 100 backhand samples
- Backhand gets weight ~2.0, fronthand gets weight ~1.0
- Model penalized 2× more for missing backhand examples
- **Forces balanced learning** regardless of data imbalance

**Expected improvement:** Backhand recall should increase from 16% → 60%+

---

### 2. Improved Architecture

**Old architecture:**
```python
LSTM(128) → Dropout → LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(32) → Dense(num_classes)
```

**New architecture:**
```python
LSTM(128) → BatchNorm → Dropout(0.4) →
LSTM(96)  → BatchNorm → Dropout(0.4) →
LSTM(64)  → BatchNorm → Dropout(0.3) →
Dense(64) → BatchNorm → Dropout(0.3) →
Dense(32) → BatchNorm → Dropout(0.2) →
Dense(num_classes)
```

**Key improvements:**
- **BatchNormalization**: Stabilizes learning, improves gradient flow
- **More capacity**: 128→96→64 LSTM units (better feature extraction)
- **Deeper dense layers**: 64→32 units (better classification)
- **Higher dropout**: 0.4 in LSTM layers (prevents overfitting to majority class)

**Expected improvement:** Better feature discrimination between classes

---

### 3. Training Configuration

**Old settings:**
```python
learning_rate: 0.001
epochs: 100
early_stopping patience: 15
lr_reduction patience: 5
```

**New settings:**
```python
learning_rate: 0.0005  # Reduced for stability
epochs: 150            # Increased for convergence
early_stopping patience: 20  # More patience
lr_reduction patience: 7     # More patience
```

**Rationale:**
- Lower learning rate prevents overshooting optima
- More epochs allows full convergence
- Increased patience prevents premature stopping

**Expected improvement:** Model reaches better local minimum

---

## Expected Results

### Before (Current):
```
Test Accuracy: 58%
Backhand:  71% precision, 16% recall  ← Barely detected!
Fronthand: 57% precision, 94% recall  ← Over-detected!
```

### After (Expected):
```
Test Accuracy: 75-85%
Backhand:  75-85% precision, 65-80% recall  ← Much better!
Fronthand: 75-85% precision, 75-85% recall  ← Balanced!
```

### Real-world detection:
- **Before:** Overlapping predictions, mostly fronthand
- **After:** Clear, distinct detections of both stroke types

---

## How to Test

### 1. Retrain the Model

```bash
poetry run python src/train_model.py
```

**Watch for:**
- Class weight output showing balance factors
- Training should show balanced accuracy for both classes
- Confusion matrix should be more diagonal

### 2. Test on Your Video

```bash
poetry run python src/detect_strokes.py D:\path\to\your\video.mp4
```

**Expected improvements:**
- No more overlapping fronthand/backhand detections
- Both stroke types detected more equally
- Fewer false positives for fronthand

### 3. Check MLflow

```bash
poetry run mlflow ui
```

**Compare:**
- Old run: backhand recall ~16%, imbalanced confusion matrix
- New run: backhand recall ~65-80%, balanced confusion matrix

---

## If Results Are Still Poor

### Problem: Still predicting mostly fronthand

**Possible causes:**
1. **Severe data imbalance** (e.g., 10:1 ratio)
   - Solution: Collect more backhand examples
   - Or use data augmentation (time stretching, noise)

2. **Classes are too similar** (model can't distinguish)
   - Solution: Check if annotations are correct
   - Review videos - are backhands truly distinct from fronthands?

### Problem: Accuracy doesn't improve much

**Try:**

1. **Increase window size** (more context):
   ```python
   'window_size': 45,  # 1.5 seconds instead of 1.0
   'overlap': 22,
   ```

2. **Lower learning rate further**:
   ```python
   'learning_rate': 0.0002,  # Even slower, more stable
   ```

3. **Add more dropout** (if overfitting):
   ```python
   keras.layers.Dropout(0.5),  # Increase from 0.4
   ```

### Problem: Training is unstable (loss jumps around)

**Try:**
1. **Reduce batch size**:
   ```python
   'batch_size': 16,  # Smaller batches = more stable gradients
   ```

2. **Add gradient clipping**:
   ```python
   optimizer = keras.optimizers.Adam(
       learning_rate=learning_rate,
       clipnorm=1.0  # Clip gradients
   )
   ```

---

## Technical Details

### Why Class Weights Work

Without class weights:
```
Loss = 0.5 × (backhand_error + fronthand_error)
```
Model minimizes total loss → focuses on majority class (fronthand)

With class weights:
```
Loss = 2.0 × backhand_error + 1.0 × fronthand_error
```
Model forced to care equally about both classes

### Why BatchNormalization Helps

- Normalizes activations between layers
- Prevents "covariate shift" (changing distributions during training)
- Acts as mild regularization
- Allows higher learning rates (but we kept it low for safety)
- Improves gradient flow through deep networks

### Why More Epochs Matter

Your confusion matrix shows the model "learned" to always predict fronthand - this is a local minimum. More epochs + lower learning rate helps escape this.

---

## Summary

**Three critical fixes:**
1. ✅ **Class weights** - Forces balanced learning (most important)
2. ✅ **Better architecture** - BatchNormalization + more capacity
3. ✅ **More training** - 150 epochs with lower learning rate

**Expected outcome:**
- Balanced predictions (no more 88% fronthand)
- Much higher backhand recall (16% → 65-80%)
- Clear, non-overlapping detections in videos

**Next step:** Retrain and test!

```bash
poetry run python src/train_model.py
poetry run python src/detect_strokes.py your_video.mp4
```

Good luck! 🎾
