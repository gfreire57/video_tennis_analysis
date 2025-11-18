# MLflow Complete Guide - Experiment Tracking & Metrics

## Overview

MLflow is integrated into the training pipeline to track experiments, parameters, metrics, and model artifacts. This guide covers installation, basic usage, advanced metrics, and interpretation.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [What Gets Tracked](#what-gets-tracked)
3. [Basic Usage](#basic-usage)
4. [MLflow UI Features](#mlflow-ui-features)
5. [Advanced Metrics](#advanced-metrics)
6. [Interpreting Metrics](#interpreting-metrics)
7. [Comparing Experiments](#comparing-experiments)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Install MLflow

```bash
poetry install
```

This installs MLflow along with all dependencies (including seaborn for confusion matrix visualization).

### Verify Installation

Training with MLflow is enabled by default in `train_model.py`:

```python
CONFIG = {
    'use_mlflow': True,
    'mlflow_experiment_name': 'tennis_stroke_recognition',
}
```

Data is stored locally in `./mlruns` directory.

---

## What Gets Tracked

### Parameters

All hyperparameters and dataset info:

- `window_size` - Sequence window size
- `overlap` - Frame overlap
- `num_classes` - Number of stroke types
- `num_features` - Number of features per frame
- `num_landmarks` - Number of selected landmarks
- `total_sequences` - Total training sequences
- `train_samples` - Training set size
- `test_samples` - Test set size
- `batch_size` - Training batch size
- `epochs` - Maximum epochs
- `learning_rate` - Optimizer learning rate
- `group_classes` - Whether classes were grouped

### Basic Metrics

- `test_accuracy` - Final test accuracy
- `test_loss` - Final test loss
- `final_train_accuracy` - Last epoch train accuracy
- `final_val_accuracy` - Last epoch validation accuracy
- `final_train_loss` - Last epoch train loss
- `final_val_loss` - Last epoch validation loss
- `epochs_trained` - Actual epochs trained (with early stopping)

### Per-Class Metrics

For **each class** (fronthand, backhand, etc.):

- `{class_name}_precision` - How many predicted strokes were correct
- `{class_name}_recall` - How many actual strokes were detected
- `{class_name}_f1_score` - Harmonic mean of precision and recall
- `{class_name}_support` - Number of test samples for this class
- `{class_name}_count` - Number of training samples

**Example:**
```
fronthand_precision: 0.85
fronthand_recall: 0.78
fronthand_f1_score: 0.81
fronthand_support: 88
fronthand_count: 420
```

### Average Metrics

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

### Artifacts

Files saved with each run:

- **Model** - Trained model (`.keras` file)
- **Training history** - Training curves (`.png`)
- **Confusion matrix** - Visual heatmap (`.png`)
- **Confusion matrix text** - Raw numpy array (`.txt`)
- **Classification report** - Complete sklearn report (`.txt`)
- **Label classes** - Mapping (`.npy`)

---

## Basic Usage

### 1. Train with MLflow Tracking

```bash
poetry run python src/train_model.py
```

MLflow tracking is enabled by default. Watch for this in the output:

```
MLflow tracking enabled: experiment 'tennis_stroke_recognition'
MLflow UI: Run 'mlflow ui' to view experiments
```

### 2. View Experiments in MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### 3. Disable MLflow (if needed)

Edit `CONFIG` in `train_model.py`:

```python
'use_mlflow': False,  # Disable MLflow tracking
```

---

## MLflow UI Features

### Experiment Page

Main page showing all training runs:

- **Runs list** - All experiments chronologically
- **Sortable columns** - Click headers to sort by any metric
- **Filter** - Search runs by parameters or metrics
- **Select** - Checkbox to select runs for comparison

### Run Details Page

Click any run to see:

**Parameters Tab:**
- All hyperparameters used
- Dataset statistics
- Model configuration

**Metrics Tab:**
- All logged metrics (basic + per-class + averages)
- Scroll to find specific class metrics

**Artifacts Tab:**
- Download models, plots, reports
- Preview images inline
- View text files

**Notes & Tags:**
- Add custom notes about the run
- Tag runs for organization

### Compare Runs

1. **Select multiple runs** (checkboxes)
2. **Click "Compare"** button
3. **View comparison:**
   - Side-by-side parameter table
   - Side-by-side metrics table
   - Chart tab for visual comparisons
4. **Sort by metric** to find best model

---

## Advanced Metrics

### Per-Class Metrics Explained

#### Precision

**Definition:** Of all strokes predicted as class X, how many were actually class X?

**Formula:** `True Positives / (True Positives + False Positives)`

**Example:**
- Model predicted 100 fronthands
- 85 were actually fronthands
- Precision = 85/100 = 0.85 (85%)

**Interpretation:**
- **High precision (>0.8):** Few false positives, predictions are trustworthy
- **Low precision (<0.6):** Many false positives, model over-predicts this class

#### Recall

**Definition:** Of all actual class X strokes, how many did the model detect?

**Formula:** `True Positives / (True Positives + False Negatives)`

**Example:**
- Video had 88 actual fronthands
- Model detected 78 of them
- Recall = 78/88 = 0.89 (89%)

**Interpretation:**
- **High recall (>0.8):** Few false negatives, catches most strokes
- **Low recall (<0.6):** Many false negatives, misses many strokes

#### F1-Score

**Definition:** Harmonic mean of precision and recall (balanced metric)

**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`

**When to use:**
- Best overall metric for imbalanced classes
- Combines precision and recall
- Prefer F1 over accuracy for imbalanced data

**Interpretation:**
- **Good F1 (>0.75):** Balanced precision and recall
- **Poor F1 (<0.60):** Either precision or recall (or both) is low

#### Support

**Definition:** Number of test samples for this class

**Example:**
- backhand_support: 75 (75 backhand samples in test set)
- fronthand_support: 88 (88 fronthand samples in test set)

**Use for:**
- Understanding class distribution
- Weighting metrics appropriately
- Identifying classes with insufficient data

---

## Interpreting Metrics

### Confusion Matrix

#### Reading the Matrix

```
Confusion Matrix:
              Predicted
              BH    FH
Actual  BH   [65    10]
        FH   [ 8    80]
```

**Diagonal (good):**
- 65 backhands correctly predicted as backhand
- 80 fronthands correctly predicted as fronthand

**Off-diagonal (errors):**
- 10 backhands incorrectly predicted as fronthand
- 8 fronthands incorrectly predicted as backhand

#### Ideal Confusion Matrix

```
              Predicted
              BH    FH
Actual  BH   [75     0]  ← All correct!
        FH   [ 0    88]  ← All correct!
```

All samples on diagonal = perfect classifier

#### Poor Confusion Matrix

```
              Predicted
              BH    FH
Actual  BH   [12    63]  ← Most backhands misclassified!
        FH   [ 5    83]
```

Most backhands classified as fronthand = severe imbalance

### Metric Thresholds

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

## Comparing Experiments

### Example Analysis Scenarios

#### Scenario 1: Good Model

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

#### Scenario 2: Imbalanced Predictions

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

#### Scenario 3: Insufficient Training Data

```
Metrics:
  fronthand_support: 15   ← Very few samples!
  backhand_support: 12    ← Very few samples!
```

**Interpretation:**
❌ Low support indicates insufficient test data
❌ Metrics may not be reliable
❌ Need more training data

**Solution:** Collect more annotated videos

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

## Best Practices

### Naming & Organization

Runs are automatically named with timestamp: `lstm_run_YYYYMMDD_HHMMSS`

You can create additional experiments for different model types:

```python
'mlflow_experiment_name': 'tennis_stroke_recognition_v2',
```

### Tracking Custom Metrics

To add custom metrics, edit the MLflow section in `train_model.py`:

```python
if CONFIG['use_mlflow']:
    mlflow.log_metric("custom_metric", value)
```

### Version Control Integration

Add to `.gitignore`:
```
mlruns/
```

This prevents tracking MLflow data in git (it can get large).

### Example Workflow

**Experiment 1: Baseline**
```python
CONFIG = {
    'window_size': 30,
    'overlap': 15,
}
```
Train and note the accuracy.

**Experiment 2: Larger Window**
```python
CONFIG = {
    'window_size': 45,  # Increased
    'overlap': 15,
}
```
Train again.

**Compare in MLflow UI:**
1. Open http://localhost:5000
2. Select both runs
3. Click "Compare"
4. See which parameters improved performance

---

## Troubleshooting

### MLflow UI Not Starting

```bash
# Check if port 5000 is in use
# Use a different port:
mlflow ui --port 5001
```

### Can't See Experiments

```bash
# Make sure you're in the project directory
cd D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis
mlflow ui
```

### Metrics Not Showing

**Check:**
1. MLflow is enabled: `CONFIG['use_mlflow'] = True`
2. Training completed successfully
3. No errors during `mlflow.log_metric()` calls
4. Refresh MLflow UI (F5)

### Confusion Matrix Not Generated

**Check:**
1. Seaborn is installed: `poetry install`
2. Check `output/confusion_matrix.png` exists
3. Check terminal for errors during plotting

### Seaborn Not Found Error

```bash
poetry install
# or
poetry add seaborn
```

### Storage Location

MLflow data is stored in `./mlruns` by default. To change:

```python
mlflow.set_tracking_uri("file:./my_custom_mlruns")
```

---

## Advanced: Remote Tracking Server

For team collaboration, set up a remote MLflow server:

```bash
# On server
mlflow server --host 0.0.0.0 --port 5000

# In train_model.py
mlflow.set_tracking_uri("http://server-ip:5000")
```

---

## Summary

### Quick Commands

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

### What's Logged

- ✅ **Parameters:** window_size, learning_rate, batch_size, etc.
- ✅ **Basic metrics:** test_accuracy, test_loss, epochs_trained
- ✅ **Per-class metrics:** precision, recall, f1-score, support for each class
- ✅ **Average metrics:** macro and weighted averages
- ✅ **Artifacts:** model, plots, confusion matrix, reports

### Use MLflow To

- 📊 Compare different model architectures
- 📊 Evaluate hyperparameter changes
- 📊 Track improvements over time
- 📊 Identify class-specific issues
- 📊 Make data collection decisions

---

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)

Happy experimenting! 🎾
