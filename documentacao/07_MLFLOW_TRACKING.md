# MLflow Tracking - Comprehensive Guide

Complete reference for experiment tracking, metrics logging, and performance analysis with MLflow.

---

## Table of Contents

1. [Overview and Setup](#overview-and-setup)
2. [What Gets Logged](#what-gets-logged)
3. [Enhanced Logging Features](#enhanced-logging-features)
4. [MLflow UI Usage](#mlflow-ui-usage)
5. [Comparing Experiments](#comparing-experiments)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [See Also](#see-also)

---

## Overview and Setup

### What is MLflow?

MLflow is integrated into the training pipeline to track experiments, parameters, metrics, and model artifacts. It provides a centralized system to:

- Track all hyperparameters and model configurations
- Log performance metrics across training epochs
- Store trained models and visualization artifacts
- Compare experiments side-by-side
- Understand model behavior through detailed metrics

### Installation & Verification

MLflow is installed via the dependency management system:

```bash
poetry install
```

This installs MLflow along with all dependencies (including seaborn for confusion matrix visualization).

### Verify MLflow is Enabled

Training with MLflow is enabled by default in `train_model.py`:

```python
CONFIG = {
    'use_mlflow': True,
    'mlflow_experiment_name': 'tennis_stroke_recognition',
}
```

Data is stored locally in `./mlruns` directory.

### Disable MLflow (if needed)

Edit `CONFIG` in `train_model.py`:

```python
'use_mlflow': False,  # Disable MLflow tracking
```

---

## What Gets Logged

### Dataset Configuration Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `window_size` | Sequence window size in frames | 45 |
| `overlap` | Overlap between windows in frames | 15 |
| `reference_fps` | Reference FPS for temporal scaling | 30 |
| `min_annotation_length` | Minimum annotation length in frames | 15 |
| `num_classes` | Number of stroke classes | 2 |
| `num_features` | Features per frame (landmarks × 4) | 60 |
| `num_landmarks` | Number of selected pose landmarks | 15 |
| `total_sequences` | Total sequences created | 1430 |
| `train_samples` | Training set size | 1144 |
| `test_samples` | Test set size | 286 |
| `group_classes` | Whether classes were grouped | True/False |

### Training Configuration Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `batch_size` | Training batch size | 32 |
| `epochs` | Maximum epochs (before early stopping) | 100 |
| `learning_rate` | Initial learning rate | 0.001 |
| `input_shape` | Model input shape | (45, 60) |

### Model Architecture Parameters

#### Layer Counts
| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `num_lstm_layers` | Number of LSTM layers | 2 |
| `num_dense_layers` | Number of Dense layers (excluding output) | 1 |
| `uses_batch_normalization` | Whether BatchNorm is used | True/False |
| `uses_bidirectional` | Whether Bidirectional LSTM is used | True/False |

#### LSTM Layer Details
| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `lstm_layer_1_units` | Units in first LSTM layer | 64 |
| `lstm_layer_2_units` | Units in second LSTM layer | 32 |
| `lstm_layer_3_units` | Units in third LSTM layer (if exists) | 64 |

**Note**: For Bidirectional layers, the units value shown is the **total output units** (wrapped LSTM units × 2).

#### Dense Layer Details
| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `dense_layer_1_units` | Units in first Dense layer | 32 |
| `dense_layer_2_units` | Units in second Dense layer (if exists) | 64 |

**Note**: Output layer (with softmax) is excluded from this count.

#### Dropout Rates
| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `dropout_1_rate` | First dropout rate | 0.3 |
| `dropout_2_rate` | Second dropout rate | 0.3 |
| `dropout_3_rate` | Third dropout rate | 0.2 |

#### Architecture Summary
| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `architecture_summary` | Human-readable architecture string | `LSTM[64, 32] → Dense[32]` |

**Possible formats**:
- `LSTM[64, 32] → Dense[32]` - Standard architecture
- `LSTM[64, 32] → Dense[32] + BatchNorm` - With BatchNormalization
- `Bidirectional-LSTM[128, 64] → Dense[64]` - With Bidirectional wrapper
- `Bidirectional-LSTM[128, 64] → Dense[64] + BatchNorm` - Both features

### Class Weights

For each class, the balanced weight is logged:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `class_weight_backhand` | Weight for backhand class | 1.663 |
| `class_weight_fronthand` | Weight for fronthand class | 0.715 |
| `class_weight_serve` | Weight for serve class (if exists) | 1.250 |

**Formula**: `n_samples / (n_classes * n_samples_for_class)`

---

## Enhanced Logging Features

### Training Time Tracking

Complete training duration is now logged with millisecond precision:

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `training_time_seconds` | Total training time in seconds | 156.7 |
| `training_time_minutes` | Total training time in minutes | 2.61 |

**Note**: Time measured from `model.fit()` start to completion (includes all epochs, early stopping, validation).

### Performance Metrics

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `test_accuracy` | Final test set accuracy | 0.7483 |
| `test_loss` | Final test set loss | 0.5234 |
| `final_train_accuracy` | Last epoch training accuracy | 0.8912 |
| `final_val_accuracy` | Last epoch validation accuracy | 0.7654 |
| `final_train_loss` | Last epoch training loss | 0.3421 |
| `final_val_loss` | Last epoch validation loss | 0.5123 |
| `epochs_trained` | Actual epochs trained (with early stopping) | 67 |

### Class Distribution

For each class, the training sample count is logged:

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `class_backhand_count` | Number of backhand samples in training | 430 |
| `class_fronthand_count` | Number of fronthand samples in training | 1000 |

### Per-Class Metrics

For **each class** (backhand, fronthand, serve, etc.):

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `{class}_precision` | Precision for this class | 0.85 |
| `{class}_recall` | Recall for this class | 0.78 |
| `{class}_f1_score` | F1-score for this class | 0.81 |
| `{class}_support` | Number of test samples for this class | 88 |

**Example for fronthand**:
```
fronthand_precision: 0.85
fronthand_recall: 0.78
fronthand_f1_score: 0.81
fronthand_support: 88
```

### Average Metrics

#### Macro Averages (simple mean across classes)

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `macro_avg_precision` | Average precision across all classes | 0.82 |
| `macro_avg_recall` | Average recall across all classes | 0.79 |
| `macro_avg_f1_score` | Average F1-score across all classes | 0.80 |

**Use case**: Treats all classes equally, good for evaluating balanced performance.

**Why macro average?** Provides balanced evaluation regardless of class distribution.

#### Weighted Averages (weighted by class support)

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `weighted_avg_precision` | Precision weighted by class size | 0.84 |
| `weighted_avg_recall` | Recall weighted by class size | 0.81 |
| `weighted_avg_f1_score` | F1-score weighted by class size | 0.82 |

**Use case**: Accounts for class imbalance, reflects overall performance on all samples.

**Why both?**
- **Macro avg** treats all classes equally (good for balanced evaluation)
- **Weighted avg** accounts for class imbalance (reflects overall performance)

### Logged Artifacts

Files saved with each run:

| Artifact | Description | Format |
|----------|-------------|--------|
| `model` | Trained Keras model | `.keras` |
| `training_history.png` | Training curves (loss & accuracy) | `.png` |
| `confusion_matrix.png` | Visual confusion matrix heatmap | `.png` |
| `confusion_matrix.txt` | Raw confusion matrix array | `.txt` |
| `classification_report.txt` | Complete sklearn classification report | `.txt` |
| `label_classes.npy` | Label encoder classes mapping | `.npy` |

---

## MLflow UI Usage

### Starting the UI

Train with MLflow tracking to generate data:

```bash
poetry run python src/train_model.py
```

MLflow tracking is enabled by default. Watch for this in the output:

```
MLflow tracking enabled: experiment 'tennis_stroke_recognition'
MLflow UI: Run 'mlflow ui' to view experiments
```

View experiments in the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

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
- Architecture details
- Class weights

**Metrics Tab:**
- All logged metrics (basic + per-class + averages)
- Scroll to find specific class metrics
- Training time in seconds and minutes

**Artifacts Tab:**
- Download models, plots, reports
- Preview images inline
- View text files

**Notes & Tags:**
- Add custom notes about the run
- Tag runs for organization

---

## Comparing Experiments

### Comparing Runs

1. **Select multiple runs** (checkboxes)
2. **Click "Compare"** button
3. **View comparison:**
   - Side-by-side parameter table
   - Side-by-side metrics table
   - Chart tab for visual comparisons
4. **Sort by metric** to find best model

### Filtering and Sorting

#### Find Best Model by F1-Score
1. Click column header: `macro_avg_f1_score`
2. Sort descending (highest first)
3. Top run = best overall model

#### Find Fastest Training
1. Click column header: `training_time_minutes`
2. Sort ascending (lowest first)
3. Compare with F1-score to find speed/performance trade-off

#### Filter by Architecture
Use the search bar:
- `params.uses_batch_normalization = "True"` - Only runs with BatchNorm
- `params.num_lstm_layers = "2"` - Only 2-layer architectures
- `params.lstm_layer_1_units = "128"` - Only runs with 128 units in first layer

#### Filter by Performance
- `metrics.macro_avg_f1_score > 0.75` - Only runs with F1 > 0.75
- `metrics.training_time_minutes < 5` - Only fast runs (<5 min)

### Understanding Per-Class Metrics

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

### Reading the Confusion Matrix

#### Visual Interpretation

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
- Both classes have high precision and recall
- F1-scores >0.80 indicate good performance
- Macro average shows balanced performance
- Model is ready for deployment!

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
- Model over-predicts fronthand (high recall, low precision)
- Model under-predicts backhand (low recall)
- Class weights may not be working
- Need to retrain with better class balancing

**Solution:** Check class weights are applied correctly

#### Scenario 3: Insufficient Training Data

```
Metrics:
  fronthand_support: 15   ← Very few samples!
  backhand_support: 12    ← Very few samples!
```

**Interpretation:**
- Low support indicates insufficient test data
- Metrics may not be reliable
- Need more training data

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

### Architecture Comparison Example

Systematic testing with detailed logging enables architecture optimization:

| Run | Architecture Summary | Training Time | Macro F1 | Backhand Recall | Fronthand Recall |
|-----|---------------------|---------------|----------|-----------------|------------------|
| 1 | LSTM[64, 32] → Dense[32] | 2.5 min | 0.71 | 0.65 | 0.78 |
| 2 | LSTM[128, 64] → Dense[64] | 4.2 min | 0.78 | 0.75 | 0.82 |
| 3 | LSTM[128, 64] → Dense[64] + BatchNorm | 4.5 min | 0.82 | 0.79 | 0.85 |
| 4 | Bidirectional-LSTM[128, 64] → Dense[64] + BatchNorm | 7.1 min | 0.84 | 0.82 | 0.87 |

**Analysis**:
- Run 2: Better F1 (+0.07) but slower (+1.7 min) → Worth it
- Run 3: BatchNorm adds +0.04 F1 for only +0.3 min → Excellent trade-off
- Run 4: Bidirectional adds +0.02 F1 but +2.6 min → May not be worth it

---

## Advanced Features

### Systematic Testing Strategy

#### Test 1: Baseline
**Goal**: Establish current performance

```python
# Current V2 architecture
LSTM(64) → LSTM(32) → Dense(32)
```

**Expected logs**:
- `architecture_summary`: "LSTM[64, 32] → Dense[32]"
- `num_lstm_layers`: 2
- `uses_batch_normalization`: False

#### Test 2: Larger Units
**Goal**: Test if more capacity helps

```python
# Increase all units
LSTM(128) → LSTM(64) → Dense(64)
```

**Expected logs**:
- `architecture_summary`: "LSTM[128, 64] → Dense[64]"
- `lstm_layer_1_units`: 128
- `lstm_layer_2_units`: 64

#### Test 3: Add BatchNormalization
**Goal**: Test if normalization improves training

```python
# Uncomment BatchNorm layers
LSTM(128) → BatchNorm → LSTM(64) → BatchNorm → Dense(64)
```

**Expected logs**:
- `architecture_summary`: "LSTM[128, 64] → Dense[64] + BatchNorm"
- `uses_batch_normalization`: True

#### Test 4: Bidirectional
**Goal**: Test if both-direction processing helps

```python
# Add Bidirectional wrapper to first layer
Bidirectional(LSTM(64)) → LSTM(64) → Dense(64) + BatchNorm
```

**Expected logs**:
- `architecture_summary`: "Bidirectional-LSTM[128, 64] → Dense[64] + BatchNorm"
- `uses_bidirectional`: True
- `lstm_layer_1_units`: 128 (64 × 2)

### Interpreting Results

#### Key Metrics for Architecture Comparison

1. **macro_avg_f1_score** - Primary metric (balanced performance)
2. **training_time_minutes** - Efficiency consideration
3. **backhand_recall / fronthand_recall** - Per-class balance
4. **test_accuracy** - Overall correctness

#### Decision Matrix

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| Higher F1, reasonable time increase (<2x) | Better architecture | **Adopt** |
| Higher F1, excessive time increase (>3x) | Overly complex | **Reject** or tune |
| Same F1, faster training | More efficient | **Adopt** |
| Lower F1 | Worse architecture | **Reject** |
| Higher F1 but unbalanced recall | Potential overfitting | **Investigate** confusion matrix |

#### Red Flags

⚠️ **Overfitting indicators**:
- `final_train_accuracy` >> `test_accuracy` (gap > 15%)
- `final_val_loss` >> `test_loss`
- High F1 on one class, low on another

⚠️ **Underfitting indicators**:
- Low `test_accuracy` (<60%)
- Similar low performance on train and test
- Need more capacity or different architecture

⚠️ **Class imbalance issues**:
- Large gap between class recalls (>20% difference)
- Check if `class_weight_{class}` is being applied
- May need to adjust weights manually

### Custom Metrics and Tracking

#### Tracking Custom Metrics

To add custom metrics, edit the MLflow section in `train_model.py`:

```python
if CONFIG['use_mlflow']:
    mlflow.log_metric("custom_metric", value)
```

#### Creating Additional Experiments

You can create additional experiments for different model types:

```python
'mlflow_experiment_name': 'tennis_stroke_recognition_v2',
```

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

### Remote Tracking Server

For team collaboration, set up a remote MLflow server:

```bash
# On server
mlflow server --host 0.0.0.0 --port 5000

# In train_model.py
mlflow.set_tracking_uri("http://server-ip:5000")
```

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

### Version Control Integration

Add to `.gitignore`:
```
mlruns/
```

This prevents tracking MLflow data in git (it can get large).

---

## Quick Reference Commands

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

# Start on different port (if 5000 is busy)
mlflow ui --port 5001

# Filter/Sort in UI:
# - Click column header to sort
# - Use search bar with syntax:
#   params.{param_name} = "value"
#   metrics.{metric_name} > threshold
```

---

## Summary

### What Gets Logged

- ✅ **Parameters:** window_size, learning_rate, batch_size, architecture details, class weights
- ✅ **Dataset info:** total sequences, train/test split, class distribution
- ✅ **Basic metrics:** test_accuracy, test_loss, epochs_trained
- ✅ **Per-class metrics:** precision, recall, f1-score, support for each class
- ✅ **Average metrics:** macro and weighted averages across classes
- ✅ **Training time:** seconds and minutes for duration analysis
- ✅ **Artifacts:** model, plots, confusion matrix, reports, label mappings

### Use MLflow To

- 📊 Compare different model architectures
- 📊 Evaluate hyperparameter changes
- 📊 Track improvements over time
- 📊 Identify class-specific issues
- 📊 Make data collection decisions
- 📊 Analyze speed vs performance trade-offs
- 📊 Reproduce best configurations

### Benefits of Enhanced Logging

1. **Systematic Experimentation**: Test different architectures and immediately see impact
2. **Easy Comparison**: Side-by-side comparison of all parameters and metrics
3. **Reproducibility**: All settings logged for reproducing best models
4. **Time Tracking**: Make informed decisions about speed vs performance
5. **Architecture Analysis**: Understand which components (BatchNorm, Bidirectional, units) matter most

### Next Steps

1. Train current baseline to establish reference metrics
2. Test recommended architecture improvements (larger units, BatchNorm)
3. Compare in MLflow UI to find optimal architecture
4. Document best configuration for production use

---

## See Also

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- See `06_GRID_SEARCH_GUIDE.md` for hyperparameter tuning strategies
- See `05_BIDIRECTIONAL_LSTM_GUIDE.md` for architecture details
- See `04_POSE_EXTRACTION_README.md` for data preparation

---

Happy experimenting! 🎾
