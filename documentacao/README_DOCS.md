# Documentation Structure

This folder contains all technical documentation for the Tennis Stroke Recognition System.

## 📚 Documentation Index

### Core Guides

#### 1. **DEVELOPMENT_NOTES.md**
**What:** Project evolution, design decisions, lessons learned
**When to read:** Understanding why the system works the way it does
**Key topics:** Neutral class removal, dependency challenges, architectural choices

#### 2. **ARCHITECTURE_GUIDE.md**
**What:** Visual system architecture and theory
**When to read:** Understanding the complete ML pipeline
**Key topics:** Data flow, LSTM architecture, sliding windows, mathematical foundations

#### 3. **USAGE_GUIDE.md**
**What:** Complete step-by-step usage instructions
**When to read:** First time using the system or learning workflows
**Key topics:** Installation, annotation, training, detection, MLflow

---

### Technical Deep Dives

#### 4. **POSE_FEATURES_EXPLAINED.md**
**What:** Deep technical explanation of pose features
**When to read:** Understanding how MediaPipe features work
**Key topics:** 132 features breakdown, LSTM learning patterns, body geometry

#### 5. **SEQUENCE_CREATION_EXPLAINED.md**
**What:** Sliding window sequence creation
**When to read:** Troubleshooting "0 sequences created" error
**Key topics:** Window size, overlap, majority voting, debugging

#### 6. **FPS_SCALING_GUIDE.md**
**What:** Automatic FPS-based parameter scaling
**When to read:** Working with videos of different frame rates
**Key topics:** Temporal consistency, scale factors, mixed FPS training

#### 7. **PREDICTION_ALIGNMENT_GUIDE.md**
**What:** Fixing "seeing the future" prediction timing
**When to read:** Predictions appear too early in timeline
**Key topics:** Window alignment modes (start/center/end), timing accuracy

---

### Model Optimization

#### 8. **MODEL_OPTIMIZATION_GUIDE.md** ⭐ MERGED
**What:** Complete guide to class balancing and feature selection
**When to read:** Need to improve model performance
**Key topics:** Class weights, feature selection, biomechanical landmarks, combined workflow
**Replaces:** CLASS_WEIGHTS_EXPLAINED.md + FEATURE_SELECTION_GUIDE.md

#### 9. **MODEL_IMPROVEMENTS.md**
**What:** Architectural improvements for better accuracy
**When to read:** Model accuracy is low or imbalanced
**Key topics:** BatchNormalization, class weights, training configuration

---

### Hyperparameter Tuning

#### 11. **TUNING_GUIDE.md**
**What:** Complete parameter tuning reference
**When to read:** Need to improve model performance
**Key topics:** Window size, learning rate, batch size, overlap

#### 12. **QUICK_START_TUNING.md**
**What:** Fast-track tuning for common issues
**When to read:** Quick fixes for low accuracy or poor detection
**Key topics:** Grouped classes, threshold adjustment, common problems

---

### MLflow & Experiment Tracking

#### 11. **MLFLOW_COMPLETE_GUIDE.md** ⭐ MERGED
**What:** Complete MLflow guide - setup, usage, metrics, interpretation
**When to read:** Setting up experiment tracking or understanding metrics
**Key topics:** Installation, UI, per-class metrics, confusion matrix, comparing runs
**Replaces:** MLFLOW_GUIDE.md + MLFLOW_METRICS_GUIDE.md

---

### Video Processing

#### 12. **VIDEO_PREPROCESSING_GUIDE.md**
**What:** Complete preprocessing pipeline
**When to read:** Videos have lighting/zoom/distortion issues
**Key topics:** Auto-brightness, auto-zoom, static zoom, fisheye correction, batch processing

---

### Project Records

#### 13. **MODIFICATIONS_MADE_TO_VIDEOS.md**
**What:** Log of preprocessing commands applied to specific videos
**When to read:** Reference for which videos were modified and how
**Key topics:** Command history, zoom/brightness/fisheye parameters used

---

## 🗺️ Quick Navigation by Task

### First Time Setup
1. README.md (project root) - Start here
2. USAGE_GUIDE.md - Step-by-step instructions
3. MLFLOW_GUIDE.md - Setup experiment tracking

### Improving Model Performance
1. TUNING_GUIDE.md - Parameter optimization
2. CLASS_WEIGHTS_EXPLAINED.md - Fix imbalanced predictions
3. FEATURE_SELECTION_GUIDE.md - Reduce overfitting
4. MODEL_IMPROVEMENTS.md - Architecture enhancements

### Troubleshooting
1. SEQUENCE_CREATION_EXPLAINED.md - "0 sequences created"
2. PREDICTION_ALIGNMENT_GUIDE.md - Timing issues
3. POSE_VISUALIZATION_GUIDE.md - Check pose detection
4. VIDEO_PREPROCESSING_GUIDE.md - Fix video quality

### Understanding the System
1. ARCHITECTURE_GUIDE.md - How everything works
2. POSE_FEATURES_EXPLAINED.md - Feature engineering
3. DEVELOPMENT_NOTES.md - Design decisions

---

## 📁 File Organization

```
documentacao/
├── README_DOCS.md (this file)
├── Core Guides/
│   ├── DEVELOPMENT_NOTES.md
│   ├── ARCHITECTURE_GUIDE.md
│   └── USAGE_GUIDE.md
├── Technical/
│   ├── POSE_FEATURES_EXPLAINED.md
│   ├── SEQUENCE_CREATION_EXPLAINED.md
│   ├── FPS_SCALING_GUIDE.md
│   └── PREDICTION_ALIGNMENT_GUIDE.md
├── Optimization/
│   ├── CLASS_WEIGHTS_EXPLAINED.md
│   ├── FEATURE_SELECTION_GUIDE.md
│   ├── MODEL_IMPROVEMENTS.md
│   ├── TUNING_GUIDE.md
│   └── QUICK_START_TUNING.md
├── MLflow/
│   ├── MLFLOW_GUIDE.md
│   └── MLFLOW_METRICS_GUIDE.md
├── Video Processing/
│   ├── VIDEO_PREPROCESSING_GUIDE.md
│   ├── VIDEO_ANNOTATION_GUIDE.md
│   ├── POSE_VISUALIZATION_GUIDE.md
│   └── STATIC_ZOOM_EXAMPLE.md
└── Records/
    └── MODIFICATIONS_MADE_TO_VIDEOS.md
```

---

## 🎯 Recommended Reading Order

### For New Users:
1. README.md (root)
2. USAGE_GUIDE.md
3. ARCHITECTURE_GUIDE.md
4. POSE_VISUALIZATION_GUIDE.md (verify videos work)

### For Improving Results:
1. QUICK_START_TUNING.md (fast fixes)
2. TUNING_GUIDE.md (comprehensive)
3. CLASS_WEIGHTS_EXPLAINED.md
4. MODEL_IMPROVEMENTS.md

### For Understanding Internals:
1. ARCHITECTURE_GUIDE.md
2. POSE_FEATURES_EXPLAINED.md
3. SEQUENCE_CREATION_EXPLAINED.md
4. DEVELOPMENT_NOTES.md

---

## 🔧 Recently Updated

- **PREDICTION_ALIGNMENT_GUIDE.md** - NEW: Fixes timing offset issues
- **FPS_SCALING_GUIDE.md** - NEW: Handles mixed FPS videos
- **CLASS_WEIGHTS_EXPLAINED.md** - Enhanced with examples
- **MODEL_IMPROVEMENTS.md** - Added BatchNormalization details

---

## 📝 Documentation Standards

All documentation follows these conventions:
- Clear **What/When/Key Topics** sections
- Concrete examples with actual code/commands
- Visual diagrams where helpful
- Troubleshooting sections
- Links to related docs

---

## 🤝 Contributing to Docs

When adding new documentation:
1. Check if content fits in existing file
2. Use clear section headers
3. Include code examples
4. Add entry to this index
5. Link related documents
