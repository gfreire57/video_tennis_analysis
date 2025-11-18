# MLflow Classification Metrics Guide

## Overview

The training script now logs **comprehensive classification metrics** to MLflow, including:

✅ **Per-class metrics:** Precision, Recall, F1-Score, Support
✅ **Average metrics:** Macro and Weighted averages
✅ **Confusion matrix:** Both visual (PNG) and text format
✅ **Classification report:** Complete sklearn report

This allows you to compare model performance across different experiments in MLflow UI.

---

## Installation

First, install the new dependency (seaborn for confusion matrix visualization):

```bash
poetry install
```

This will install seaborn based on the updated `pyproject.toml`.

---

## What's Logged to MLflow

### 1. Per-Class Metrics

For **each class** (fronthand, backhand, etc.):

- `{class_name}_precision` - How many predicted strokes were correct
- `{class_name}_recall` - How many actual strokes were detected
- `{class_name}_f1_score` - Harmonic mean of precision and recall
- `{class_name}_support` - Number of test samples for this class

**Example metrics in MLflow:**
```
fronthand_precision: 0.85
fronthand_recall: 0.78
fronthand_f1_score: 0.81
fronthand_support: 88

backhand_precision: 0.72
backhand_recall: 0.65
backhand_f1_score: 0.68
backhand_support: 75
```

### 2. Average Metrics

**Macro Averages** (simple average across classes):
- `macro_avg_precision` - Average precision across all classes
- `macro_avg_recall` - Average recall across all classes
- `macro_avg_f1_score` - Average F1-score across all classes

**Weighted Averages** (weighted by class support):
- `weighted_avg_precision` - Precision weighted by class size
- `weighted_avg_recall` - Recall weighted by class size
- `weighted_avg_f1_score` - F1-score weighted by class size

**Why both?**
- **Macro avg** treats all classes equally (good for balanced evaluation)
- **Weighted avg** accounts for class imbalance (reflects overall performance)

### 3. Confusion Matrix

**Visual confusion matrix (`confusion_matrix.png`):**
- Heatmap showing predictions vs actual labels
- Color-coded (darker blue = more samples)
- Annotated with counts
- Labeled with class names

**Text confusion matrix (`confusion_matrix.txt`):**
- Raw numpy array format
- Rows = True labels
- Columns = Predicted labels

### 4. Classification Report

**Complete classification report (`classification_report.txt`):**
```
              precision    recall  f1-score   support

    backhand       0.72      0.65      0.68        75
   fronthand       0.85      0.78      0.81        88

    accuracy                           0.79       163
   macro avg       0.79      0.72      0.75       163
weighted avg       0.79      0.72      0.75       163
```

---

## Viewing Metrics in MLflow UI

### Step 1: Start MLflow UI

```bash
cd video_tennis_analysis
poetry run mlflow ui
```

Open http://localhost:5000 in your browser.

### Step 2: View Metrics

1. **Click on your experiment** (tennis_stroke_recognition)
2. **Select a run** to view details
3. **Metrics tab** shows all logged metrics:
   - Scroll through to see per-class metrics
   - Check average metrics
   - View test accuracy, loss, etc.

### Step 3: Compare Runs

**Select multiple runs** (checkbox on left), then:
- Click **"Compare"** button
- View side-by-side metrics comparison
- Sort by any metric
- Create plots comparing metrics across runs

### Step 4: View Artifacts

Click **"Artifacts" tab** to download:
- `confusion_matrix.png` - Visual confusion matrix
- `confusion_matrix.txt` - Text confusion matrix
- `classification_report.txt` - Full classification report
- `training_history.png` - Training curves
- `tennis_stroke_model.keras` - Trained model

---

## Interpreting Metrics

### Precision

**Definition:** Of all strokes predicted as class X, how many were actually class X?

**Formula:** `True Positives / (True Positives + False Positives)`

**Example:**
- Model predicted 100 fronthands
- 85 were actually fronthands
- Precision = 85/100 = 0.85 (85%)

**High precision (>0.8):** Few false positives, predictions are trustworthy
**Low precision (<0.6):** Many false positives, model over-predicts this class

### Recall

**Definition:** Of all actual class X strokes, how many did the model detect?

**Formula:** `True Positives / (True Positives + False Negatives)`

**Example:**
- Video had 88 actual fronthands
- Model detected 78 of them
- Recall = 78/88 = 0.89 (89%)

**High recall (>0.8):** Few false negatives, catches most strokes
**Low recall (<0.6):** Many false negatives, misses many strokes

### F1-Score

**Definition:** Harmonic mean of precision and recall (balanced metric)

**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`

**When to use:**
- Best overall metric for imbalanced classes
- Combines precision and recall
- Prefer F1 over accuracy for imbalanced data

**Good F1 (>0.75):** Balanced precision and recall
**Poor F1 (<0.60):** Either precision or recall (or both) is low

### Support

**Definition:** Number of test samples for this class

**Example:**
- backhand_support: 75 (75 backhand samples in test set)
- fronthand_support: 88 (88 fronthand samples in test set)

**Use for:**
- Understanding class distribution
- Weighting metrics appropriately
- Identifying classes with insufficient data

---

## Example Analysis

### Scenario 1: Good Model

```
Metrics:
  fronthand_precision: 0.85
  fronthand_recall: 0.87
  fronthand_f1_score: 0.86

  backhand_precision: 0.83
  backhand_recall: 0.81
  backhand_f1_score: 0.82

  macro_avg_f1_score: 0.84
```

**Interpretation:**
✅ Both classes have high precision and recall
✅ F1-scores >0.80 indicate good performance
✅ Macro average shows balanced performance
✅ Model is ready for deployment!

### Scenario 2: Imbalanced Predictions

```
Metrics:
  fronthand_precision: 0.57
  fronthand_recall: 0.94  ← High recall
  fronthand_f1_score: 0.71

  backhand_precision: 0.71
  backhand_recall: 0.16   ← Low recall
  backhand_f1_score: 0.26  ← Very low F1

  macro_avg_f1_score: 0.49
```

**Interpretation:**
❌ Model over-predicts fronthand (high recall, low precision)
❌ Model under-predicts backhand (low recall)
❌ Class weights may not be working
❌ Need to retrain with better class balancing

**Solution:** Check class weights are applied correctly

### Scenario 3: Insufficient Training Data

```
Metrics:
  fronthand_precision: 0.62
  fronthand_recall: 0.58
  fronthand_support: 15   ← Very few samples!

  backhand_precision: 0.65
  backhand_recall: 0.60
  backhand_support: 12    ← Very few samples!
```

**Interpretation:**
❌ Low support indicates insufficient test data
❌ Metrics may not be reliable
❌ Need more training data

**Solution:** Collect more annotated videos

---

## Comparing Experiments

### Use Case: Did Class Weights Help?

**Before (without class weights):**
```
Run 1:
  backhand_recall: 0.16
  fronthand_recall: 0.94
  macro_avg_f1_score: 0.49
```

**After (with class weights):**
```
Run 2:
  backhand_recall: 0.75
  fronthand_recall: 0.82
  macro_avg_f1_score: 0.78
```

**Result:** Class weights significantly improved backhand recall!

### Use Case: Different Window Sizes

Compare `macro_avg_f1_score` across runs:

| Run | Window Size | Macro Avg F1 |
|-----|-------------|--------------|
| 1   | 20 frames   | 0.72         |
| 2   | 30 frames   | 0.78         |
| 3   | 45 frames   | 0.75         |

**Result:** 30 frames gives best F1-score

---

## Confusion Matrix Interpretation

### Reading the Confusion Matrix

```
Confusion Matrix:
              Predicted
              BH    FH
Actual  BH   [65    10]
        FH   [ 8    80]

BH = Backhand, FH = Fronthand
```

**Diagonal (good):**
- 65 backhands correctly predicted as backhand
- 80 fronthands correctly predicted as fronthand

**Off-diagonal (errors):**
- 10 backhands incorrectly predicted as fronthand (false negatives for backhand)
- 8 fronthands incorrectly predicted as backhand (false negatives for fronthand)

### Ideal Confusion Matrix

```
              Predicted
              BH    FH
Actual  BH   [75     0]  ← All correct!
        FH   [ 0    88]  ← All correct!
```

**All samples on diagonal = perfect classifier**

### Poor Confusion Matrix

```
              Predicted
              BH    FH
Actual  BH   [12    63]  ← Most backhands misclassified!
        FH   [ 5    83]
```

**Most backhands classified as fronthand = severe imbalance**

---

## Metric Thresholds

### General Guidelines

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Precision | >0.85 | 0.75-0.85 | 0.65-0.75 | <0.65 |
| Recall | >0.85 | 0.75-0.85 | 0.65-0.75 | <0.65 |
| F1-Score | >0.85 | 0.75-0.85 | 0.65-0.75 | <0.65 |

**For tennis stroke detection:**
- Target: F1 >0.75 for production use
- Minimum: F1 >0.65 for testing
- Below 0.65: Need significant improvements

---

## MLflow UI Walkthrough

### View Per-Class Metrics

1. Open MLflow UI: `poetry run mlflow ui`
2. Click on experiment → Click on run
3. Go to **Metrics** tab
4. Scroll to find:
   - `fronthand_precision`
   - `fronthand_recall`
   - `fronthand_f1_score`
   - `backhand_precision`
   - etc.

### Compare Multiple Runs

1. Select runs using checkboxes
2. Click **"Compare"** button
3. View table with all metrics side-by-side
4. Click column headers to sort
5. Create charts: Click **"Chart"** tab

### Download Artifacts

1. Click on run
2. Go to **Artifacts** tab
3. Click file to preview
4. Click download icon to save locally

---

## Troubleshooting

### Seaborn Not Found Error

**Error:**
```
ModuleNotFoundError: No module named 'seaborn'
```

**Solution:**
```bash
poetry install
# or
poetry add seaborn
```

### Metrics Not Showing in MLflow

**Check:**
1. MLflow is enabled: `CONFIG['use_mlflow'] = True`
2. Training completed successfully
3. No errors during `mlflow.log_metric()` calls
4. Refresh MLflow UI (F5)

### Confusion Matrix Not Generated

**Check:**
1. Seaborn is installed
2. Check `output/confusion_matrix.png` exists
3. Check terminal for errors during plotting

---

## Summary

**Quick Reference:**

```bash
# Install dependencies
poetry install

# Train model (logs metrics to MLflow)
poetry run python src/train_model.py

# View metrics in MLflow UI
poetry run mlflow ui
# Open http://localhost:5000

# Compare runs:
# 1. Select multiple runs (checkboxes)
# 2. Click "Compare"
# 3. View metrics side-by-side
```

**Metrics logged:**
- ✅ Per-class: precision, recall, f1-score, support
- ✅ Averages: macro and weighted
- ✅ Visual confusion matrix (PNG)
- ✅ Text confusion matrix
- ✅ Classification report

**Artifacts saved:**
- ✅ confusion_matrix.png
- ✅ confusion_matrix.txt
- ✅ classification_report.txt
- ✅ training_history.png
- ✅ Model file

**Use metrics to:**
- 📊 Compare different model architectures
- 📊 Evaluate hyperparameter changes
- 📊 Track improvements over time
- 📊 Identify class-specific issues
- 📊 Make data collection decisions

Happy experimenting! 🎾
