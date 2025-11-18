# Class Weights Explained

## What Are Class Weights?

Class weights are a technique to handle **imbalanced datasets** where some classes have many more examples than others.

---

## The Problem: Class Imbalance

### Your Current Results (Without Proper Balancing):

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

---

## The Solution: Class Weights

### How It Works

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

### Example Calculation:

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

### During Training:

**Without class weights:**
```
Loss calculation per sample:
  Misclassify backhand  → Loss = 1.0
  Misclassify fronthand → Loss = 1.0

Batch loss:
  86 backhand errors  → Total loss = 86 × 1.0 = 86
  26 fronthand errors → Total loss = 26 × 1.0 = 26
  Combined: 112

Gradient update: Focuses more on reducing fronthand errors (smaller count)
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

**Result:** Model learns to pay equal attention to both classes, not just the majority.

---

## Implementation in Your Code

### Step 1: Calculate Class Weights

**Location:** [src/train_model.py:550-565](src/train_model.py#L550-L565)

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate balanced weights based on training data distribution
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),  # [0, 1] for backhand, fronthand
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

### Step 2: Apply During Training

**Location:** [src/train_model.py:588-596](src/train_model.py#L588-L596)

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    class_weight=class_weights,  # ← CRITICAL: Apply weights here!
    callbacks=callbacks,
    verbose=1
)
```

**What happens:**
- During backpropagation, loss for each sample is multiplied by its class weight
- Minority class errors get amplified → Model pays more attention
- Majority class errors get dampened → Model doesn't over-focus

---

## Checking If It's Working

### 1. Check Training Output

Look for this section in your training logs:

```
======================================================================
CLASS WEIGHTS (to balance training)
======================================================================
  backhand: 1.663 (n=430)
  fronthand: 0.715 (n=1000)
```

If you see this, class weights ARE being calculated.

### 2. Check MLflow

```bash
mlflow ui
```

Navigate to your run → Check parameters:
- Should see individual class weights logged
- Check if backhand_recall improved over epochs

### 3. Check Confusion Matrix

**Before class weights (your current result):**
```
Confusion Matrix:
                Predicted
                Backhand  Fronthand
Actual Backhand    38        86       ← 31% recall
       Fronthand   26       127       ← 83% recall

Heavily biased toward fronthand!
```

**After class weights (expected):**
```
Confusion Matrix (expected improvement):
                Predicted
                Backhand  Fronthand
Actual Backhand    70        54       ← 56% recall (much better!)
       Fronthand   40       113       ← 74% recall (slight decrease OK)

More balanced!
```

**Key metrics:**
- Backhand recall should increase significantly (31% → 50-60%)
- Fronthand recall may decrease slightly (83% → 70-75%)
- Overall accuracy may stay similar or improve slightly
- **Balance** is the goal, not just raw accuracy

---

## Common Issues

### Issue 1: Class Weights Not Helping

**Symptoms:**
- Backhand recall still low (<40%)
- Model still biased toward fronthand

**Possible causes:**

1. **Extreme imbalance (>90/10):**
   ```
   Solution: Collect more minority class data
   Or: Use data augmentation (horizontal flip: fronthand → backhand)
   ```

2. **Weights too conservative:**
   ```python
   # Manually increase minority weight
   class_weights[0] = class_weights[0] * 1.5  # Boost backhand weight by 50%
   ```

3. **Other imbalance issues:**
   - Annotation quality (backhand annotations incomplete?)
   - Feature representation (backhand harder to capture with current landmarks?)

### Issue 2: Training Unstable

**Symptoms:**
- Loss oscillates wildly
- Validation loss much higher than training loss

**Cause:** Class weights too extreme (e.g., backhand weight = 5.0)

**Solution:**
```python
# Cap maximum weight
max_weight = 2.0
class_weights = {k: min(v, max_weight) for k, v in class_weights.items()}
```

### Issue 3: Can't Find Class Weights in Training Output

**Check:**
```bash
# Search training logs for "CLASS WEIGHTS"
grep "CLASS WEIGHTS" training_log.txt
```

If not found:
- Class weights calculation may have failed
- Check if `compute_class_weight` import is present
- Verify `class_weight=class_weights` in `model.fit()`

---

## Expected Training Output

### With Class Weights Working Correctly:

```
Epoch 1/150
49/49 ━━━━━━━━━━━━━━━━━━━━ 3s - loss: 1.2345 - accuracy: 0.5234
  ↑
  Loss initially higher because minority class errors count more

Epoch 50/150
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s - loss: 0.6789 - accuracy: 0.6543
  ↑
  Loss decreases as model learns to balance both classes

Epoch 100/150
49/49 ━━━━━━━━━━━━━━━━━━━━ 2s - loss: 0.5432 - accuracy: 0.7012
  ↑
  Model converges with balanced performance

======================================================================
EVALUATION
======================================================================
Test Accuracy: 0.7012

Classification Report:
              precision    recall  f1-score   support

    backhand       0.65      0.58      0.61       124
   fronthand       0.73      0.78      0.75       153

    accuracy                           0.70       277
   macro avg       0.69      0.68      0.68       277
weighted avg       0.70      0.70      0.69       277

Confusion Matrix:
                Predicted
                Backhand  Fronthand
Actual Backhand    72        52       ← 58% recall (balanced!)
       Fronthand   34       119       ← 78% recall (good!)
```

**Notice:**
- Backhand recall improved: 31% → 58% ✅
- Fronthand recall decreased slightly: 83% → 78% (acceptable trade-off)
- Overall more balanced predictions

---

## Summary

### What Class Weights Do:

✅ **Make the model care equally about all classes** (not just the majority)
✅ **Increase loss for minority class errors** (forces model to learn)
✅ **Decrease loss for majority class errors** (prevents over-focus)
✅ **Result: More balanced predictions**

### How to Check If It's Working:

1. ✅ Training output shows "CLASS WEIGHTS" section
2. ✅ `model.fit()` has `class_weight=class_weights` parameter
3. ✅ Backhand recall improves over epochs
4. ✅ Confusion matrix more balanced

### Expected Results:

```
Before:
  Backhand recall: 31% ❌
  Fronthand recall: 83%

After:
  Backhand recall: 50-60% ✅
  Fronthand recall: 70-80% ✅
```

### Your Code Already Implements This! ✅

**Location:** [src/train_model.py](src/train_model.py)
- Lines 550-565: Calculate weights
- Lines 561-565: Display weights
- Line 593: Apply in `model.fit()`

**The implementation is correct.** If backhand recall is still low, the issue is likely:
1. Data imbalance too extreme (>80/20)
2. Need more backhand training data
3. Backhand sequences too short (window_size mismatch)
4. Feature representation doesn't capture backhand well

Next step: Run `poetry run python src/analyze_annotations.py` to check data balance.
