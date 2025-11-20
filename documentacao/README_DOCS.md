# Documentation Index

**Complete guide to the Tennis Stroke Recognition System documentation**

---

## Quick Navigation

### New Users - Start Here
1. **[00_GETTING_STARTED.md](00_GETTING_STARTED.md)** - Installation, first training run, and quick start

### Understanding the System
2. **[01_ARCHITECTURE.md](01_ARCHITECTURE.md)** - Complete system architecture and data flow
3. **[02_POSE_FEATURES.md](02_POSE_FEATURES.md)** - How pose features are extracted and used

### Using the System
4. **[03_USAGE_GUIDE.md](03_USAGE_GUIDE.md)** - Step-by-step usage examples and workflows

### Configuration & Optimization
5. **[04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md)** - Pose extraction, sequences, FPS scaling, alignment
6. **[05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md)** - Class weights, features, architecture, tuning
7. **[06_GRID_SEARCH.md](06_GRID_SEARCH.md)** - Automated hyperparameter search
8. **[07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md)** - Experiment tracking and comparison

### Advanced Topics
9. **[08_VIDEO_PREPROCESSING.md](08_VIDEO_PREPROCESSING.md)** - Video enhancement and preprocessing
10. **[09_DEVELOPMENT_NOTES.md](09_DEVELOPMENT_NOTES.md)** - Design decisions and lessons learned

---

## Documentation by Topic

### Getting Started

| Document | Purpose | Who Should Read |
|----------|---------|-----------------|
| [00_GETTING_STARTED.md](00_GETTING_STARTED.md) | Quick start guide | Everyone (start here!) |
| [03_USAGE_GUIDE.md](03_USAGE_GUIDE.md) | Detailed usage examples | New users |

### System Understanding

| Document | Purpose | Who Should Read |
|----------|---------|-----------------|
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | Complete architecture explanation | Developers, researchers |
| [02_POSE_FEATURES.md](02_POSE_FEATURES.md) | Pose estimation technical details | Advanced users, researchers |
| [09_DEVELOPMENT_NOTES.md](09_DEVELOPMENT_NOTES.md) | Design decisions, what worked/didn't | Contributors, researchers |

### Configuration

| Document | Purpose | Who Should Read |
|----------|---------|-----------------|
| [04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md) | Pose caching, sequences, FPS scaling | Everyone training models |
| [05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md) | Improving model accuracy | Users with accuracy issues |
| [06_GRID_SEARCH.md](06_GRID_SEARCH.md) | Automated hyperparameter tuning | Advanced users |
| [07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md) | Experiment tracking | Everyone training models |
| [08_VIDEO_PREPROCESSING.md](08_VIDEO_PREPROCESSING.md) | Video quality improvement | Users with poor pose detection |

---

## Reading Paths

### Path 1: Complete Beginner

1. **[00_GETTING_STARTED.md](00_GETTING_STARTED.md)** - Install and first run
2. **[03_USAGE_GUIDE.md](03_USAGE_GUIDE.md)** - Learn the workflow
3. **[07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md)** - Track your experiments
4. **[05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md)** - Improve accuracy if needed

### Path 2: Troubleshooting Low Accuracy

1. **[07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md)** - Compare experiments
2. **[05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md)** - Class weights, features, architecture
3. **[06_GRID_SEARCH.md](06_GRID_SEARCH.md)** - Find optimal hyperparameters
4. **[04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md)** - Check sequence/FPS configuration

### Path 3: Troubleshooting Detection Issues

1. **[00_GETTING_STARTED.md](00_GETTING_STARTED.md#troubleshooting)** - Common issues
2. **[08_VIDEO_PREPROCESSING.md](08_VIDEO_PREPROCESSING.md)** - Improve video quality
3. **[04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md)** - FPS scaling, alignment
4. **[03_USAGE_GUIDE.md](03_USAGE_GUIDE.md#tuning-guide)** - Adjust detection parameters

### Path 4: Understanding How It Works

1. **[01_ARCHITECTURE.md](01_ARCHITECTURE.md)** - System overview
2. **[02_POSE_FEATURES.md](02_POSE_FEATURES.md)** - Feature extraction details
3. **[04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md)** - Data processing pipeline
4. **[09_DEVELOPMENT_NOTES.md](09_DEVELOPMENT_NOTES.md)** - Why these design choices

### Path 5: Advanced Optimization

1. **[05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md)** - All optimization strategies
2. **[06_GRID_SEARCH.md](06_GRID_SEARCH.md)** - Systematic hyperparameter search
3. **[07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md)** - Compare results
4. **[09_DEVELOPMENT_NOTES.md](09_DEVELOPMENT_NOTES.md)** - Learn from past experiments

---

## Common Questions - Where to Look

### Installation and Setup
- **How do I install?** → [00_GETTING_STARTED.md](00_GETTING_STARTED.md#installation)
- **GPU not working?** → [00_GETTING_STARTED.md](00_GETTING_STARTED.md#troubleshooting)
- **First time setup?** → [00_GETTING_STARTED.md](00_GETTING_STARTED.md#quick-start-workflow)

### Training
- **How to train a model?** → [00_GETTING_STARTED.md](00_GETTING_STARTED.md#first-training-run)
- **Detailed training workflow?** → [03_USAGE_GUIDE.md](03_USAGE_GUIDE.md#step-3-train-the-model)
- **What happens during training?** → [01_ARCHITECTURE.md](01_ARCHITECTURE.md#part-2-training-architecture)

### Data Configuration
- **Pose extraction is slow?** → [04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md#pose-extraction-and-caching)
- **What is FPS scaling?** → [04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md#fps-scaling-for-temporal-consistency)
- **How do sequences work?** → [04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md#sequence-creation-with-sliding-windows)
- **Prediction timing is off?** → [04_DATA_CONFIGURATION.md](04_DATA_CONFIGURATION.md#window-alignment-and-prediction-timing)

### Model Performance
- **Accuracy too low (<70%)?** → [05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md)
- **Class imbalance issues?** → [05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md#class-balancing-with-weights)
- **How to tune hyperparameters?** → [05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md#hyperparameter-tuning)
- **What is Bidirectional LSTM?** → [05_MODEL_OPTIMIZATION.md](05_MODEL_OPTIMIZATION.md#bidirectional-lstm)
- **Automated tuning?** → [06_GRID_SEARCH.md](06_GRID_SEARCH.md)

### Detection Issues
- **No strokes detected?** → [00_GETTING_STARTED.md](00_GETTING_STARTED.md#troubleshooting)
- **Too many false positives?** → [03_USAGE_GUIDE.md](03_USAGE_GUIDE.md#tuning-guide)
- **Poor pose detection (<50%)?** → [08_VIDEO_PREPROCESSING.md](08_VIDEO_PREPROCESSING.md)
- **Video is dark/low quality?** → [08_VIDEO_PREPROCESSING.md](08_VIDEO_PREPROCESSING.md)

### Experiment Tracking
- **How to use MLflow?** → [07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md)
- **What metrics are tracked?** → [07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md#what-gets-logged)
- **How to compare experiments?** → [07_MLFLOW_TRACKING.md](07_MLFLOW_TRACKING.md#comparing-experiments)

### Technical Details
- **How does LSTM work?** → [01_ARCHITECTURE.md](01_ARCHITECTURE.md#part-2-training-architecture)
- **What are pose features?** → [02_POSE_FEATURES.md](02_POSE_FEATURES.md)
- **Why skip neutral class?** → [09_DEVELOPMENT_NOTES.md](09_DEVELOPMENT_NOTES.md#what-didnt-work)
- **Design decisions explained?** → [09_DEVELOPMENT_NOTES.md](09_DEVELOPMENT_NOTES.md#design-decisions)

---

## Quick Command Reference

```bash
# Extract poses (once)
poetry run python src/extract_poses.py

# Train model
poetry run python src/train_model.py

# Detect strokes
poetry run python src/detect_strokes.py video.mp4

# MLflow UI
poetry run mlflow ui

# Grid search
poetry run python src/grid_search.py --grid minimal

# Verify annotations
poetry run python src/verify_annotation.py

# Check GPU
poetry run python src/check_gpu.py

# Visualize poses
poetry run python src/visualize_pose.py video.mp4 --max-frames 300

# Preprocess videos
poetry run python src/preprocess_video.py input.mp4 output.mp4 --auto-brighten
```

---

**Start here:** [00_GETTING_STARTED.md](00_GETTING_STARTED.md) 🎾
