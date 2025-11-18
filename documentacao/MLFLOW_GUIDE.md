# MLflow Experiment Tracking Guide

## Overview

MLflow is now integrated into the training pipeline to track experiments, parameters, metrics, and model artifacts.

## Installation

```bash
poetry install
```

This will install MLflow along with all other dependencies.

## What Gets Tracked

### Parameters
- `window_size` - Sequence window size
- `overlap` - Frame overlap
- `num_classes` - Number of stroke types
- `total_sequences` - Total training sequences
- `train_samples` - Training set size
- `test_samples` - Test set size
- `batch_size` - Training batch size
- `epochs` - Maximum epochs
- `learning_rate` - Optimizer learning rate

### Metrics
- `test_accuracy` - Final test accuracy
- `test_loss` - Final test loss
- `final_train_accuracy` - Last epoch train accuracy
- `final_val_accuracy` - Last epoch validation accuracy
- `final_train_loss` - Last epoch train loss
- `final_val_loss` - Last epoch validation loss
- `epochs_trained` - Actual epochs trained (with early stopping)
- `class_<name>_count` - Sample count per class

### Artifacts
- Trained model (`.keras` file)
- Training history plot (`.png`)
- Label classes mapping (`.npy`)
- Confusion matrix (`.txt`)

## Usage

### 1. Train with MLflow Tracking

```bash
poetry run python src/train_model.py
```

MLflow tracking is enabled by default. Data is stored in `./mlruns` directory.

### 2. View Experiments in MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### 3. Compare Experiments

In the MLflow UI, you can:
- Compare multiple training runs side-by-side
- Visualize metrics over time
- Filter runs by parameters
- Download artifacts
- Load previous models

### 4. Disable MLflow (if needed)

Edit `CONFIG` in `train_model.py`:

```python
'use_mlflow': False,  # Disable MLflow tracking
```

## MLflow UI Features

### Experiment Page
- Lists all training runs
- Shows metrics, parameters, and timestamps
- Sortable and filterable columns

### Run Details Page
For each run, you can see:
- **Parameters**: All hyperparameters used
- **Metrics**: Performance metrics
- **Artifacts**: Download models, plots, etc.
- **Notes**: Add custom notes about the run
- **Tags**: Add custom tags for organization

### Compare Runs
- Select multiple runs
- View side-by-side comparison
- Generate comparison charts
- Export comparison tables

## Best Practices

### Naming Runs
Runs are automatically named with timestamp: `lstm_run_YYYYMMDD_HHMMSS`

### Organizing Experiments
All runs are tracked under the experiment: `tennis_stroke_recognition`

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

## Example Workflow

### Experiment 1: Baseline
```python
CONFIG = {
    'window_size': 30,
    'overlap': 15,
    ...
}
```

Train and note the accuracy.

### Experiment 2: Larger Window
```python
CONFIG = {
    'window_size': 45,  # Increased
    'overlap': 15,
    ...
}
```

Train again.

### Compare in MLflow UI
1. Open http://localhost:5000
2. Select both runs
3. Click "Compare"
4. See which parameters improved performance

## Troubleshooting

### MLflow UI not starting
```bash
# Check if port 5000 is in use
# Use a different port:
mlflow ui --port 5001
```

### Can't see experiments
```bash
# Make sure you're in the project directory
cd D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis
mlflow ui
```

### Storage location
MLflow data is stored in `./mlruns` by default. To change:

```python
mlflow.set_tracking_uri("file:./my_custom_mlruns")
```

## Advanced: Remote Tracking Server

For team collaboration, set up a remote MLflow server:

```bash
# On server
mlflow server --host 0.0.0.0 --port 5000

# In train_model.py
mlflow.set_tracking_uri("http://server-ip:5000")
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
