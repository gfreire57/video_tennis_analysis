# Suppressing TensorFlow Warnings

## The Problem

When running TensorFlow scripts, you may see informational messages like:

```
2025-11-16 22:21:10.223233: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on.
You may see slightly different numerical results due to floating-point round-off errors from
different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
```

These are **harmless info messages**, not errors. They don't affect functionality but can clutter output.

---

## Solution Applied

I've added warning suppression to all TensorFlow scripts:

### Scripts Updated:
- ✅ [src/train_model.py](src/train_model.py)
- ✅ [src/detect_strokes.py](src/detect_strokes.py)
- ✅ [src/hyperparameter_tuning.py](src/hyperparameter_tuning.py)

### What Was Added:

```python
import os
# Suppress TensorFlow warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import tensorflow as tf  # Import AFTER setting environment variables
```

**Important:** These lines must come **before** importing TensorFlow!

---

## What Each Setting Does

### `TF_CPP_MIN_LOG_LEVEL`

Controls TensorFlow C++ logging level:
- `0` = All messages (default, very verbose)
- `1` = Filter out INFO messages
- `2` = Filter out INFO + WARNING messages (recommended)
- `3` = Filter out everything except ERROR messages

**We use `2`** - shows errors but hides info/warnings.

### `TF_ENABLE_ONEDNN_OPTS`

Controls oneDNN optimizations:
- `1` = Enabled (default) - shows info messages about optimizations
- `0` = Disabled - suppresses oneDNN messages

**We use `0`** - disables the messages without affecting performance significantly.

---

## For hyperparameter_tuning.py

The script also filters output from subprocess calls:

```python
# Filter out TensorFlow warnings from subprocess output
filtered_lines = [
    line for line in stdout_lines
    if not any(phrase in line for phrase in [
        'oneDNN custom operations',
        'TF_ENABLE_ONEDNN_OPTS',
        'tensorflow/core/util/port.cc'
    ])
]
```

This ensures clean output even when running training scripts via subprocess.

---

## Running Scripts Now

**All scripts now run cleanly:**

```bash
# Training (no warnings)
poetry run python src/train_model.py

# Detection (no warnings)
poetry run python src/detect_strokes.py video.mp4

# Hyperparameter tuning (no warnings)
poetry run python src/hyperparameter_tuning.py
```

---

## If You Still See Warnings

### Method 1: Set Environment Variables Globally (Windows)

**PowerShell:**
```powershell
$env:TF_CPP_MIN_LOG_LEVEL="2"
$env:TF_ENABLE_ONEDNN_OPTS="0"
poetry run python src/train_model.py
```

**Command Prompt:**
```cmd
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
poetry run python src/train_model.py
```

### Method 2: Add to Shell Profile (Permanent)

**PowerShell Profile** (`$PROFILE`):
```powershell
$env:TF_CPP_MIN_LOG_LEVEL="2"
$env:TF_ENABLE_ONEDNN_OPTS="0"
```

**Command Prompt** (set system environment variables):
1. Search "Environment Variables" in Windows
2. Add:
   - `TF_CPP_MIN_LOG_LEVEL` = `2`
   - `TF_ENABLE_ONEDNN_OPTS` = `0`

---

## Other Warnings You Might See

### CUDA/GPU Warnings

If you see GPU-related warnings but don't have a GPU:
```
Could not load dynamic library 'cudart64_110.dll'
```

**This is normal!** TensorFlow will automatically use CPU. No action needed.

### NumPy Warnings

```
np.float is deprecated, use float instead
```

**This is a library compatibility issue.** Doesn't affect functionality. Update libraries if it bothers you:
```bash
poetry update numpy
```

---

## Summary

✅ All scripts now suppress TensorFlow info messages
✅ Hyperparameter tuning runs cleanly in loops
✅ Only actual errors will be shown
✅ No performance impact

The warnings were purely informational and are now hidden for cleaner output.
