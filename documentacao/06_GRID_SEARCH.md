# Grid Search - Hyperparameter Optimization

Systematic hyperparameter and architecture optimization for tennis stroke recognition model.

---

## Table of Contents

1. [Overview and Quick Start](#overview-and-quick-start)
2. [Complete Guide](#complete-guide)
3. [Predefined Grids](#predefined-grids)
4. [Usage Examples](#usage-examples)
5. [Customization](#customization)
6. [Result Interpretation](#result-interpretation)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [See Also](#see-also)

---

## Overview and Quick Start

### What is Grid Search?

Grid search systematically tests different combinations of model parameters to find the optimal configuration. Instead of manually trying different settings, the script automatically:

1. **Generates** all combinations from specified parameter ranges
2. **Trains** a model for each combination
3. **Tracks** all results in MLflow
4. **Identifies** the best performing model
5. **Exports** results for analysis

### Why Use Grid Search?

✅ **Systematic exploration**: Test all parameter combinations methodically
✅ **Time-saving**: Automate hours of manual experimentation
✅ **Reproducible**: All parameters and results tracked in MLflow
✅ **Data-driven decisions**: Compare models objectively with metrics
✅ **Find optimal configuration**: Discover best architecture/hyperparameters for your data

### What Gets Optimized?

**Model Architecture:**
- LSTM layer sizes (e.g., [128, 64] vs [256, 128])
- Number of LSTM layers (2 vs 3 vs 4)
- Dense layer units
- Dropout rates
- BatchNormalization (on/off)
- Bidirectional LSTM (on/off)

**Training Hyperparameters:**
- Learning rate
- Batch size

**Data Configuration:**
- Window size
- Overlap
- FPS scaling (on/off)

### Implementation Overview

**`src/grid_search_configs.py`** - Parameter Grid Definitions

Predefined parameter grids for different use cases:

| Grid | Combinations | Time | Purpose |
|------|--------------|------|---------|
| **minimal** | 4 | ~10 min | Test setup |
| **small** | 20-30 | ~1-2 hours | Quick exploration |
| **medium** | 80-120 | ~6-12 hours | Systematic search |
| **large** | 300+ | ~24-48 hours | Exhaustive search |
| **architecture** | ~100 | ~8-12 hours | Focus on model architecture |
| **hyperparameter** | ~150 | ~4-8 hours | Focus on training params |
| **data** | ~60 | ~4-6 hours | Focus on window/overlap |

**`src/grid_search.py`** - Main Grid Search Script

Features:
- ✅ Full grid search (test all combinations)
- ✅ Random search (sample random combinations)
- ✅ Automatic MLflow tracking
- ✅ Progress monitoring with time estimates
- ✅ CSV export of all results
- ✅ Automatic best model identification
- ✅ Integration with existing `train_model.py` code

---

## Quick Start Steps

### Step 1: Test Setup (10 minutes)

```bash
poetry run python src/grid_search.py --grid minimal
```

This runs 4 quick tests to verify everything works.

### Step 2: Quick Exploration (1-2 hours)

```bash
poetry run python src/grid_search.py --grid small
```

Tests 20-30 parameter combinations to find promising directions.

### Step 3: View Results

```bash
mlflow ui
```

Open http://localhost:5000 to view all runs, compare models, and analyze results.

---

## Complete Guide

### Getting Started

#### Minimal Test (4 runs - ~10 minutes)

Test the grid search setup:

```bash
poetry run python src/grid_search.py --grid minimal
```

#### Small Grid (20-30 runs - ~1-2 hours)

Quick exploration of key parameters:

```bash
poetry run python src/grid_search.py --grid small
```

#### View Grid Without Running

See what will be tested before running:

```bash
poetry run python src/grid_search.py --grid small --show-grid
```

#### Random Search (30 samples - ~1-2 hours)

Sample random combinations instead of full grid:

```bash
poetry run python src/grid_search.py --random --n-samples 30
```

---

## Predefined Grids

### MINIMAL (4 combinations)

**Purpose**: Test grid search implementation
**Time**: ~10 minutes
**Parameters**:
- LSTM layers: 2 options
- Learning rate: 2 options
- Everything else: fixed

**When to use**: Verifying setup, testing before long runs

```bash
poetry run python src/grid_search.py --grid minimal
```

---

### SMALL (20-30 combinations)

**Purpose**: Quick exploration of key parameters
**Time**: ~1-2 hours
**Parameters**:
- LSTM layers: `[128,64]`, `[64,32]`
- Dense units: `32`, `64`
- Dropout: 2 configurations
- Learning rate: `0.001`, `0.0005`
- Batch size: `32` (fixed)
- Window/overlap: fixed

**When to use**:
- Initial exploration
- Limited time/resources
- Validating that grid search works

```bash
poetry run python src/grid_search.py --grid small
```

---

### MEDIUM (80-120 combinations)

**Purpose**: Balanced systematic exploration
**Time**: ~6-12 hours
**Parameters**:
- LSTM layers: 4 configurations (2-layer and 3-layer)
- Dense units: `32`, `64`, `128`
- Dropout: 3 configurations
- Learning rate: `0.0001`, `0.0005`, `0.001`
- Batch size: `16`, `32`, `64`
- BatchNorm: on/off
- Window size: `30`, `45`, `60`
- Overlap: `10`, `15`, `20`

**When to use**:
- Finding optimal architecture
- Have moderate computational resources
- Systematic parameter exploration

```bash
poetry run python src/grid_search.py --grid medium
```

---

### LARGE (300+ combinations)

**Purpose**: Comprehensive exhaustive search
**Time**: ~24-48 hours (GPU recommended)
**Parameters**: All parameters with 3-6 values each

**⚠️ WARNING**: This will take significant time!

**When to use**:
- Final optimization
- GPU cluster available
- Exhaustive parameter search needed

```bash
poetry run python src/grid_search.py --grid large
```

---

### ARCHITECTURE_FOCUSED (~100 combinations)

**Purpose**: Focus on model architecture
**Time**: ~8-12 hours
**Fixed**: Learning rate, batch size, window/overlap
**Varied**: LSTM configurations, dense units, BatchNorm, Bidirectional

**When to use**:
- Already tuned hyperparameters
- Want to find optimal architecture
- Comparing layer configurations

```bash
poetry run python src/grid_search.py --grid architecture
```

---

### HYPERPARAMETER_FOCUSED (~150 combinations)

**Purpose**: Focus on training hyperparameters
**Time**: ~4-8 hours
**Fixed**: Architecture (current V2: LSTM[128,64])
**Varied**: Learning rate, batch size, window size, overlap

**When to use**:
- Already have good architecture
- Optimizing training parameters
- Fine-tuning data configuration

```bash
poetry run python src/grid_search.py --grid hyperparameter
```

---

### DATA_FOCUSED (~60 combinations)

**Purpose**: Focus on data parameters
**Time**: ~4-6 hours
**Fixed**: Architecture and training hyperparameters
**Varied**: Window size (11 values), overlap (6 values), FPS scaling

**When to use**:
- Finding optimal sequence length
- Testing temporal window configurations
- Evaluating FPS scaling impact

```bash
poetry run python src/grid_search.py --grid data
```

---

## Usage Examples

### Example 1: Quick Exploration

```bash
# Run small grid search
poetry run python src/grid_search.py --grid small

# View results in MLflow
mlflow ui
```

**What happens**:
1. Tests 20-30 parameter combinations
2. Each combination trains a full model
3. Results logged to MLflow
4. CSV exported with all results
5. Best model identified

**Output files**:
- `output/grid_search_results_TIMESTAMP.csv` - All results
- MLflow runs in `mlruns/` directory

---

### Example 2: Limit Number of Runs

```bash
# Run only first 20 combinations of medium grid
poetry run python src/grid_search.py --grid medium --max-runs 20
```

**When to use**: Test subset of larger grid to save time

---

### Example 3: Random Search

```bash
# Sample 50 random combinations
poetry run python src/grid_search.py --random --n-samples 50
```

**Advantages over full grid**:
- Much faster for large parameter spaces
- Often finds good solutions with fewer runs
- Good for exploratory analysis

---

### Example 4: Preview Grid Before Running

```bash
# See what will be tested
poetry run python src/grid_search.py --grid medium --show-grid
```

**Output**:
```
======================================================================
GRID CONFIGURATION: MEDIUM
======================================================================
Total combinations: 108

Parameters:
  lstm_layers: 4 values
  dense_units: 3 values
    → [32, 64, 128]
  dropout_rates: 3 values
  use_batch_norm: 2 values
    → [False, True]
  ...
======================================================================
```

---

### Example 5: View All Available Grids

```bash
# List all predefined grids
poetry run python src/grid_search_configs.py
```

**Output**:
```
Available grid configurations:

  minimal         -    4 combinations
  small           -   24 combinations
  medium          -  108 combinations
  large           -  432 combinations
  architecture    -   96 combinations
  hyperparameter  -  150 combinations
  data            -   66 combinations
```

---

## Output Files and Results

### Console Output

During execution, you'll see:

```
======================================================================
GRID SEARCH RUN 5/24
======================================================================
Parameters:
  lstm_layers: [128, 64]
  dense_units: 64
  dropout_rates: [0.3, 0.3, 0.2]
  use_batch_norm: False
  learning_rate: 0.001
  batch_size: 32
======================================================================

Loading data...
Found 10 annotation files
...
Training...
Epoch 1/150
...

✅ Run 5/24 complete:
   Test Accuracy: 0.7654
   Macro F1-Score: 0.7521
   Training Time: 3.45 min
   MLflow Run ID: a1b2c3d4e5f6...

📊 Progress: 5/24 (20.8%)
⏱️  Elapsed: 17.3 min | Estimated remaining: 65.7 min
```

### Final Summary

After all runs complete:

```
======================================================================
GRID SEARCH COMPLETE!
======================================================================

✅ Results saved to: output/grid_search_results_20250118_143022.csv

🏆 BEST MODEL:
   Run: 18
   Macro F1-Score: 0.8234
   Test Accuracy: 0.8156
   MLflow Run ID: x1y2z3...

   Parameters:
     lstm_layers: [256, 128]
     dense_units: 128
     dropout_rates: [0.2, 0.2, 0.1]
     learning_rate: 0.0005
     batch_size: 32

📈 TOP 5 MODELS (by Macro F1-Score):
    18. F1=0.8234, Acc=0.8156, Time=5.2min
    12. F1=0.8156, Acc=0.8091, Time=4.8min
    21. F1=0.8103, Acc=0.7987, Time=3.9min
    15. F1=0.8045, Acc=0.7923, Time=4.5min
     7. F1=0.7998, Acc=0.7856, Time=3.2min

⏱️  Total grid search time: 1.85 hours
✅ View all results in MLflow: mlflow ui
======================================================================
```

### CSV Export

The results CSV contains all parameters and metrics:

| run_number | run_id | test_accuracy | macro_f1_score | training_time_minutes | lstm_layers | dense_units | learning_rate | batch_size |
|------------|--------|---------------|----------------|-----------------------|-------------|-------------|---------------|------------|
| 1 | a1b2c3... | 0.7234 | 0.7123 | 3.45 | [128, 64] | 64 | 0.001 | 32 |
| 2 | d4e5f6... | 0.7456 | 0.7345 | 3.78 | [128, 64] | 128 | 0.001 | 32 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Use for**:
- Sorting by any metric
- Filtering by parameter values
- Creating custom visualizations
- Statistical analysis

### MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000

**Filter runs**:
```
# Show only grid search runs
tags.mlflow.runName LIKE '%grid_search%'

# Filter by F1-score
metrics.macro_avg_f1_score > 0.75

# Filter by parameter
params.uses_batch_normalization = "True"
```

**Compare runs**:
1. Select multiple runs (checkboxes)
2. Click "Compare"
3. View parameters and metrics side-by-side
4. Sort by any column

---

## Customization

### Option 1: Modify Existing Grid

Edit `src/grid_search_configs.py`:

```python
SMALL_GRID = {
    'lstm_layers': [
        [128, 64],     # Keep these
        [256, 128],    # Add this one
    ],
    'dense_units': [64, 128],  # Add 128
    'learning_rate': [0.001],   # Fix this
    # ... other parameters
}
```

### Option 2: Create Custom Grid

In `src/grid_search_configs.py`, add:

```python
MY_CUSTOM_GRID = {
    'lstm_layers': [[128, 64]],  # Fixed architecture
    'dense_units': [64],
    'dropout_rates': [[0.3, 0.3, 0.2]],
    'use_batch_norm': [True],  # Test with BatchNorm
    'use_bidirectional': [False],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],  # Focus on LR
    'batch_size': [16, 32, 64, 128],  # And batch size
    'window_size': [45],
    'overlap': [15],
}
```

Then update `get_grid()` function:

```python
def get_grid(grid_name):
    grids = {
        # ... existing grids ...
        'custom': MY_CUSTOM_GRID,  # Add your grid
    }
    # ...
```

Run it:

```bash
poetry run python src/grid_search.py --grid custom
```

### Option 3: Modify Random Search Ranges

In `src/grid_search_configs.py`:

```python
RANDOM_SEARCH_RANGES = {
    'lstm_layers': [
        [64, 32],
        [128, 64],
        [256, 128],
    ],
    'dense_units': (32, 256, 'int'),  # Random int between 32-256
    'learning_rate': (0.0001, 0.01, 'log'),  # Log-uniform sampling
    'batch_size': [16, 32, 64],
    # ...
}
```

**Sampling types**:
- **List**: Random choice from discrete values
- **Tuple (min, max, 'int')**: Random integer in range
- **Tuple (min, max, 'float')**: Random float in range
- **Tuple (min, max, 'log')**: Log-uniform (good for learning rates)

---

## Result Interpretation

### Metrics to Compare

**Primary metric**:
- **`macro_avg_f1_score`**: Best overall metric (balances precision/recall across classes)

**Secondary metrics**:
- **`test_accuracy`**: Overall correctness
- **`training_time_minutes`**: Efficiency consideration
- **Per-class recall**: Check for balanced performance

### Interpreting Metric Values

**Primary Metric: `macro_avg_f1_score`**

- **>0.80**: Excellent
- **0.70-0.80**: Good
- **0.60-0.70**: Acceptable
- **<0.60**: Needs improvement

### Decision Matrix

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| F1 >> current baseline | Better model found | **Adopt** this configuration |
| F1 similar, faster training | More efficient | **Consider** adopting |
| F1 higher, much slower (>3x) | Diminishing returns | **Evaluate** trade-off |
| F1 lower | Worse configuration | **Reject** |
| High train F1, low test F1 | Overfitting | **Add** regularization |
| Low F1 across all runs | Fundamental issue | **Check** data quality |

### Identifying Patterns

**In MLflow UI**:

1. **Sort by `macro_avg_f1_score`** (descending)
2. **Look at top 10 runs**
3. **Find common patterns**:
   - Do they all use BatchNorm?
   - Do they prefer certain LSTM sizes?
   - Is there an optimal learning rate range?

**Example patterns**:
```
Top 5 runs all have:
- LSTM units: 128-256 (not 64)
- Learning rate: 0.0005-0.001 (not higher)
- BatchNorm: True
- Dropout: 0.2-0.3 (not higher)

→ Conclusion: Moderate-large LSTM, medium LR, BatchNorm helpful
```

### Trade-offs to Consider

**F1-Score vs Training Time**:
```
Run 18: F1=0.82, Time=5.2min
Run 12: F1=0.81, Time=3.2min  ← 2% worse F1, 38% faster
```

**Question**: Is 2% improvement worth 60% more training time?

**Class Balance**:
```
Run 18: backhand_recall=0.85, fronthand_recall=0.82  ← Balanced
Run 15: backhand_recall=0.92, fronthand_recall=0.71  ← Unbalanced
```

**Question**: Prefer balanced or optimize for specific class?

---

## Best Practices

### 1. Start Small, Then Expand

**Recommended workflow**:

1. **Test setup** (10 min):
   ```bash
   poetry run python src/grid_search.py --grid minimal
   ```

2. **Quick exploration** (1-2 hours):
   ```bash
   poetry run python src/grid_search.py --grid small
   ```

3. **Focused search** based on small grid results (4-8 hours):
   - If architecture matters most: `--grid architecture`
   - If hyperparameters matter most: `--grid hyperparameter`

4. **Final optimization** (6-12 hours):
   ```bash
   poetry run python src/grid_search.py --grid medium
   ```

### 2. Use Random Search for Large Spaces

If full grid has > 200 combinations:

```bash
# Sample 50 random combinations instead
poetry run python src/grid_search.py --random --n-samples 50
```

**Benefits**:
- Faster results
- Good coverage of parameter space
- Often finds near-optimal solutions

### 3. Monitor Progress

Grid search can take hours. Monitor with:

```bash
# In another terminal
mlflow ui
```

- Check intermediate results
- Identify patterns early
- Stop early if results are poor

### 4. Limit Expensive Parameters

**Expensive (slow training)**:
- Large LSTM units (512, 1024)
- Many layers (4+)
- Very large batch sizes
- Very long window sizes

**Cheap (fast training)**:
- Smaller LSTM units (64, 128)
- 2-3 layers
- Moderate batch sizes (16-64)
- Moderate window sizes (30-60)

**Strategy**: Test expensive parameters sparingly in initial grids

### 5. Set Reasonable Max Runs

```bash
# Limit medium grid to 50 runs
poetry run python src/grid_search.py --grid medium --max-runs 50
```

**When to use**:
- Testing overnight (limit to ~8-hour runtime)
- Limited computational resources
- Want preliminary results quickly

### 6. Check Results Incrementally

Don't wait for all runs to complete:

```bash
# After 10-20 runs, check MLflow
mlflow ui
```

**Look for**:
- Clear winners emerging
- Parameters that don't matter
- Signs of overfitting (low test accuracy)

**Action**: Adjust grid or stop early if needed

---

## Troubleshooting

### Problem: Grid search fails immediately

**Error**: `No module named 'train_model'`

**Solution**:
```bash
# Make sure you're in project directory
cd d:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis

# Run with poetry
poetry run python src/grid_search.py --grid small
```

---

### Problem: "No sequences created" for all runs

**Cause**: Window size or overlap incompatible with data

**Solution**:
1. Check annotation lengths in data
2. Use smaller window sizes
3. Use data-focused grid to find optimal window size:
   ```bash
   poetry run python src/grid_search.py --grid data
   ```

---

### Problem: All F1-scores very low (< 0.50)

**Possible causes**:
1. **Data quality**: Annotations incorrect or insufficient
2. **Class imbalance**: Severe imbalance not handled by weights
3. **Wrong parameters**: All tested configurations inadequate

**Solutions**:
1. Check annotation quality manually
2. Review class distribution in MLflow
3. Try simpler grid with proven baseline:
   ```bash
   # Test known-good configuration
   poetry run python src/train_model.py
   ```

---

### Problem: Grid search taking too long

**Solution 1**: Limit runs
```bash
poetry run python src/grid_search.py --grid medium --max-runs 30
```

**Solution 2**: Use random search
```bash
poetry run python src/grid_search.py --random --n-samples 30
```

**Solution 3**: Use smaller grid
```bash
poetry run python src/grid_search.py --grid small
```

---

### Problem: Want to resume interrupted grid search

**Current limitation**: Grid search doesn't support resume (yet)

**Workaround**:
1. Check MLflow to see which runs completed
2. Note the run number that failed
3. Modify grid to exclude completed combinations
4. Re-run with `--max-runs` to limit

**Future enhancement**: Add `--resume` flag

---

### Problem: Out of memory during training

**Error**: `ResourceExhaustedError: OOM when allocating tensor`

**Solutions**:
1. **Reduce batch size**: Edit grid to use smaller batches
2. **Smaller models**: Limit LSTM units to < 256
3. **Close other programs**: Free up GPU memory

---

## Quick Reference Summary

### Commands Reference

| Task | Command |
|------|---------|
| Test setup | `poetry run python src/grid_search.py --grid minimal` |
| Quick exploration | `poetry run python src/grid_search.py --grid small` |
| Architecture-focused | `poetry run python src/grid_search.py --grid architecture` |
| Hyperparameter-focused | `poetry run python src/grid_search.py --grid hyperparameter` |
| Data-focused | `poetry run python src/grid_search.py --grid data` |
| Systematic search | `poetry run python src/grid_search.py --grid medium` |
| Random search | `poetry run python src/grid_search.py --random --n-samples 50` |
| Preview grid | `poetry run python src/grid_search.py --grid small --show-grid` |
| View results | `mlflow ui` |
| List all grids | `poetry run python src/grid_search_configs.py` |

### Recommended Workflow

1. **Start**: `--grid minimal` (verify setup)
2. **Explore**: `--grid small` (find promising directions)
3. **Focus**: `--grid architecture` or `--grid hyperparameter`
4. **Optimize**: `--grid medium` (final optimization)
5. **Analyze**: MLflow UI → identify best model
6. **Deploy**: Update `train_model.py` CONFIG with best parameters

### Key Takeaways

✅ Grid search automates hyperparameter exploration
✅ Start small, expand based on results
✅ Use MLflow to track and compare all experiments
✅ Focus on `macro_avg_f1_score` as primary metric
✅ Consider training time vs performance trade-offs
✅ Random search is efficient for large parameter spaces
✅ Predefined grids cover different optimization strategies

---

## See Also

- **04_BIDIRECTIONAL_LSTM_GUIDE.md** - Deep dive into bidirectional LSTM architecture and when to use it
- **05_POSE_EXTRACTION_README.md** - Understanding pose data extraction and sequence creation
- **07_MLFLOW_ENHANCED_LOGGING.md** - Advanced experiment tracking and visualization
- **08_DISABLING_FPS_SCALING.md** - Data preprocessing options for sequence creation
- **src/train_model.py** - Main training script with configurable parameters
- **src/grid_search.py** - Grid search implementation
- **src/grid_search_configs.py** - Predefined grid configurations

---

**Happy optimizing!**
