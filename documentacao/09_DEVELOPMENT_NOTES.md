# Development Notes

This document captures the design decisions, challenges, and lessons learned during the development of the Tennis Stroke Recognition System.

## Table of Contents

- [Project Evolution](#project-evolution)
- [What Worked](#what-worked)
- [What Didn't Work](#what-didnt-work)
- [Technical Challenges](#technical-challenges)
- [Design Decisions](#design-decisions)
- [Lessons Learned](#lessons-learned)
- [Future Improvements](#future-improvements)

## Project Evolution

### Initial Approach

The project started with a simple goal: classify tennis strokes using pose estimation and deep learning.

**Initial Architecture:**
1. Use MediaPipe to extract pose landmarks
2. Create sequences from annotated video frames
3. Train an LSTM classifier
4. Apply to new videos

**Initial Assumptions:**
- All frames should be classified (stroke or "neutral")
- Sliding window would work for both training and inference
- Simple LSTM would be sufficient

### Final Architecture

The architecture evolved into a **two-stage system**:

**Stage 1: Training (Stroke Classification Only)**
- Train LSTM on annotated stroke segments only
- No "neutral" class
- Focus on distinguishing between stroke types

**Stage 2: Inference (Continuous Video Analysis)**
- Sliding window detection with configurable stride
- Confidence thresholding to filter predictions
- Stroke merging to combine fragmented detections
- Timeline generation and frequency analysis

## What Worked

### 1. MediaPipe Pose Estimation

**Success:** MediaPipe proved extremely robust for extracting body keypoints from tennis videos.

**Benefits:**
- Works well even with varying camera angles
- Handles partial occlusions reasonably well
- Provides temporal smoothing for stable landmarks
- 132 features (33 landmarks × 4 values) are sufficient for stroke recognition

**Configuration that worked:**
```python
mp_pose.Pose(
    static_image_mode=False,      # Video mode for temporal smoothing
    model_complexity=1,            # Balance between speed and accuracy
    smooth_landmarks=True,         # Reduce jitter
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### 2. LSTM Architecture

**Success:** 3-layer LSTM with dropout captured temporal patterns effectively.

**Architecture:**
```python
LSTM(128, return_sequences=True) → Dropout(0.3)
LSTM(64, return_sequences=True) → Dropout(0.3)
LSTM(32) → Dropout(0.3)
Dense(num_classes, softmax)
```

**Why this worked:**
- 30-frame window (≈1 second at 30 fps) captures full stroke motion
- Stacked LSTMs learn hierarchical temporal features
- Dropout prevents overfitting on small datasets
- 15-frame overlap provides enough training examples

### 3. Skipping Neutral Class

**Success:** Removing the "neutral" class was the breakthrough for achieving good performance.

**Before (with neutral):**
```
Classification Report:
      backhand       0.00      0.00      0.00        69
     fronthand       0.00      0.00      0.00        84
       neutral       0.91      1.00      0.95      1803
```

**After (without neutral):**
```
Classification Report:
              precision    recall  f1-score   support
    backhand       0.83      0.89      0.86       112
   fronthand       0.88      0.82      0.85       135
```

**Implementation:**
```python
# train_model.py, line 195
if majority_count > window_size * 0.5 and majority_label != 'neutral':
    X.append(window)
    y.append(majority_label)
```

### 4. Two-Stage Detection System

**Success:** Separating training (classification) from inference (detection) solved the continuous video problem.

**Training:** Focus on "what is this stroke?"
**Inference:** Focus on "when do strokes occur?"

**Key parameters for detection:**
```python
stride=5              # Check every 5 frames (6 predictions/sec at 30fps)
confidence=0.7        # Only accept high-confidence predictions
min_duration=13       # Filter brief false positives
merge_nearby=15       # Combine fragmented detections
```

### 5. MLflow Integration

**Success:** MLflow made experiment tracking effortless and reproducible.

**Benefits:**
- Compare different window sizes (30 vs 45 vs 60 frames)
- Track impact of overlap changes (10 vs 15 vs 20 frames)
- Monitor class distribution across experiments
- Recover best models from any run

**Critical metrics tracked:**
- Test accuracy (overall performance)
- Per-class precision/recall (identify weak classes)
- Epochs trained (detect early stopping)
- Training time (optimize efficiency)

### 6. Majority Voting for Label Assignment

**Success:** Assigning window labels based on >50% frame coverage proved robust.

```python
# Count frames per label in window
label_counts = Counter(frame_labels)
majority_label, majority_count = label_counts.most_common(1)[0]

# Use window only if label covers >50%
if majority_count > window_size * 0.5:
    # This is a valid stroke window
```

**Why this worked:**
- Reduces ambiguity at stroke boundaries
- Ensures each window is clearly one stroke type
- Handles slight annotation inaccuracies

## What Didn't Work

### 1. ⚠️ Neutral Class (CRITICAL FAILURE)

**What we tried:** Include a "neutral" class to represent non-stroke frames.

**Implementation:**
```python
# Original approach (FAILED)
for each frame:
    if frame has annotation:
        label = annotation_label
    else:
        label = 'neutral'  # ← Problem here
```

**Why it failed:**

**Data imbalance:**
```
Total sequences: 9780
  neutral: 9012 (91%)  ← Overwhelming majority
  fronthand: 421 (4%)
  backhand: 347 (3%)
```

**Model behavior:**
- Model learned to always predict "neutral" (91% accuracy)
- All actual strokes got 0% precision
- Model effectively became useless

**The fundamental problem:**
- Tennis videos have many non-stroke frames (player waiting, moving, etc.)
- Annotating only strokes means 90%+ frames become "neutral"
- LSTM learns that "neutral" is the safest prediction

**Lesson:** Don't create implicit classes from missing data.

### 2. ⚠️ Including All Frames in Training

**What we tried:** Use every frame from annotated videos, with sliding windows creating sequences from both strokes and non-stroke segments.

**Why it failed:**
- Created massive class imbalance (see above)
- Non-stroke movements are too diverse to learn as one class
- Wasted compute on irrelevant patterns
- Diluted the training signal for actual strokes

**What we learned:** For supervised learning, focus on what you explicitly labeled, not what you didn't label.

### 3. ⚠️ TensorFlow GPU Installation with Poetry

**What we tried:** Install TensorFlow with GPU support using `poetry add tensorflow[and-cuda]`

**Error:**
```
Unable to find installation candidates for nvidia-nccl-cu12 (2.21.5)
wheel(s) were skipped as your project's environment does not support the identified abi tags
```

**Why it failed:**
- Poetry on Windows had trouble with CUDA package wheels
- The `[and-cuda]` extra tried to install platform-specific binaries
- Mismatch between Poetry's environment and CUDA requirements

**Solution:**
- Use standard `tensorflow>=2.17.0,<2.19.0`
- Install CUDA toolkit separately if GPU is needed
- TensorFlow auto-detects GPU if CUDA is present
- Not a problem for CPU-only training

### 4. ⚠️ Protobuf Version Conflict

**What we tried:** Use latest TensorFlow 2.20.0 with MediaPipe 0.10.21

**Error:**
```
Because tensorflow (2.20.0) depends on protobuf (>=5.28.0)
and mediapipe (0.10.21) depends on protobuf (>=4.25.3,<5)
tensorflow (2.20.0) is incompatible with mediapipe (0.10.21)
```

**Why it failed:**
- TensorFlow 2.20+ requires protobuf >=5.28
- MediaPipe 0.10.x requires protobuf <5
- No version satisfies both constraints

**Solution:**
```toml
# Use TensorFlow 2.17-2.18 (compatible with protobuf <5)
"tensorflow>=2.17.0,<2.19.0"
"mediapipe>=0.10.21,<0.11.0"
```

**Lesson:** Check dependency constraints before using "latest" versions.

### 5. ⚠️ Label Studio JSON Format Changes

**What we tried:** Parse Label Studio exports with fixed JSON structure.

**Problem:** Label Studio changed export format between versions.

**Old format:**
```json
{
  "data": {
    "video": "/data/local-files/?d=videos/Bhand_1.MP4"
  },
  "annotations": [{
    "result": [...]
  }]
}
```

**New format:**
```json
{
  "task": {
    "data": {
      "video": "/data/local-files/?d=videos/Bhand_1.MP4"
    }
  },
  "result": [...]
}
```

**Solution:** Handle both formats with try/except:
```python
try:
    video_path = data['task']['data']['video']
except KeyError:
    video_path = data['data']['video']

if 'result' in data and isinstance(data['result'], list):
    result_data = data['result']  # New format
elif 'annotations' in data:
    result_data = data['annotations'][0]['result']  # Old format
```

**Lesson:** External tools evolve; make your code resilient to format changes.

### 6. ⚠️ Video Path Extraction

**What we tried:** Simple string replacement of Label Studio path prefix.

**Problem:** Path duplication and platform differences.

**Label Studio path:**
```
/data/local-files/?d=videos/Bhand_1.MP4
```

**Incorrect approach:**
```python
# WRONG: Assumes specific structure
path = path.replace('/data/local-files/?d=videos/', '')
# Result: 'Bhand_1.MP4'
# Combined with base: 'D:/videos/Bhand_1.MP4' ✓ Works by accident
```

**What broke:** If base path already includes 'videos/' folder:
```python
base = 'D:/data/videos/'
path = 'Bhand_1.MP4'  # Wrong - missing 'videos/' prefix
# Result: 'D:/data/videos/Bhand_1.MP4' ✓ Still works

# But if Label Studio includes 'videos/':
path = 'videos/Bhand_1.MP4'
# Result: 'D:/data/videos/videos/Bhand_1.MP4' ✗ WRONG
```

**Correct solution:**
```python
# 1. Remove Label Studio prefix
path = path.replace('/data/local-files/?d=', '')

# 2. Remove 'videos/' prefix if present (base already has it)
if path.startswith('videos/') or path.startswith('videos\\'):
    path = path[7:]

# 3. Use Path for proper OS handling
full_path = Path(video_base_path) / path
```

**Lesson:** Don't assume path structures; parse and validate explicitly.

## Technical Challenges

### Challenge 1: Class Imbalance

**Problem:** Limited training data with uneven distribution across stroke types.

**Mitigation strategies:**
- Class weights in model training
- Stratified train/test split
- Data augmentation (time warping, not implemented yet)
- Focus on annotating underrepresented classes

**Code:**
```python
# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Apply during training
model.fit(..., class_weight=class_weight_dict)
```

### Challenge 2: Temporal Alignment

**Problem:** Stroke timing varies (fast serves vs slow slices).

**Solution:** Fixed window size with overlap
- 30 frames captures most strokes at 30fps
- 15-frame overlap creates multiple views of each stroke
- Majority voting handles strokes shorter than window

**Trade-off:**
- Larger window: Better for slow strokes, worse for fast ones
- Smaller window: Better for fast strokes, worse for slow ones
- 30 frames (1 second) is a good middle ground

### Challenge 3: Annotation Quality

**Problem:** Human annotation has inconsistencies:
- Stroke boundaries not perfectly aligned
- Some strokes missed or mislabeled
- Subjective interpretation (is this a slice or regular shot?)

**Mitigation:**
- Verification script to catch obvious errors
- Majority voting tolerates boundary inaccuracies
- Confidence thresholding at inference time

### Challenge 4: Continuous Video Detection

**Problem:** Training is on isolated strokes, but inference is on continuous video.

**Solution:** Sliding window with post-processing
- Small stride (5 frames) for fine-grained detection
- Confidence threshold to filter noise
- Merge nearby predictions of same class
- Minimum duration to filter brief false positives

**Parameters to tune:**
```python
stride = 5           # Detection frequency (lower = more precise)
confidence = 0.7     # Certainty required (higher = fewer false positives)
min_duration = 13    # Minimum stroke length (higher = filter noise)
merge_nearby = 15    # Fragmentation tolerance (higher = merge more)
```

## Design Decisions

### Decision 1: LSTM vs CNN vs Transformer

**Options considered:**
1. **3D CNN**: Spatial + temporal convolutions
2. **LSTM**: Sequential modeling
3. **Transformer**: Attention-based temporal modeling

**Choice: LSTM**

**Rationale:**
- Pose data is already spatial features (no need for spatial convolutions)
- Stroke is inherently sequential (LSTM's strength)
- Small dataset (Transformers need more data)
- LSTM is interpretable and well-understood

**Trade-offs:**
- LSTMs can be slow to train
- Transformers might be better with more data
- Could revisit if dataset grows significantly

### Decision 2: Window Size (30 frames)

**Options considered:**
- 15 frames (0.5 sec): Too short for full stroke
- 30 frames (1.0 sec): Captures most strokes
- 45 frames (1.5 sec): Longer context
- 60 frames (2.0 sec): Very long context

**Choice: 30 frames**

**Rationale:**
- Tennis strokes typically last 0.5-1.5 seconds
- 30 frames captures the "core" motion
- Small enough to fit in memory
- Large enough for meaningful temporal patterns

**How to test other sizes:**
```python
# In train_model.py CONFIG
'window_size': 45,  # Try different values
```

MLflow will track results for comparison.

### Decision 3: Skip Neutral vs Other Approaches

**Options considered:**

**Option 1: Skip neutral (CHOSEN)**
- Train only on annotated strokes
- Use confidence thresholding at inference

**Option 2: Balanced sampling**
- Undersample neutral to match stroke classes
- Requires many more annotated videos

**Option 3: Two-stage classifier**
- Stage 1: Stroke vs non-stroke (binary)
- Stage 2: Stroke type (multi-class)
- More complex pipeline

**Option 4: Annotations only (no neutral)**
- Manually annotate full videos including neutral
- Time-consuming and subjective

**Why skip neutral:**
- Simplest implementation
- Works with existing annotations
- Good results (83-88% precision)
- Fits user's use case (detect when strokes occur)

### Decision 4: Majority Voting Threshold (50%)

**Options considered:**
- 30%: Allow more boundary cases
- 50%: Clear majority
- 70%: Very strict
- 100%: Entire window must be one label

**Choice: 50% (majority)**

**Rationale:**
- Balances purity vs sample size
- Tolerates slight annotation inaccuracies
- Ensures each window is "mostly" one stroke
- If <50% is max label, window is ambiguous (skip it)

### Decision 5: MLflow vs Alternatives

**Options considered:**
- **TensorBoard**: TensorFlow's native tracking
- **Weights & Biases**: Cloud-based tracking
- **MLflow**: Open-source, local-first tracking
- **Neptune**: Cloud-based with free tier

**Choice: MLflow**

**Rationale:**
- Open source and free
- Runs locally (no account needed)
- Simple integration
- Good UI for comparison
- Can upgrade to remote server later

## Lessons Learned

### 1. Start with the Simplest Thing That Could Work

**Mistake:** Originally tried to handle all frames, all cases, all possibilities.

**Learning:** Start with annotated strokes only. Add complexity only when needed.

**Application:**
- First: Get basic classifier working (strokes only)
- Then: Add continuous video detection (sliding window)
- Later: Could add neutral class if needed (still not needed)

### 2. Class Imbalance is More Serious Than It Seems

**Mistake:** Thought 91% accuracy with neutral class was decent.

**Learning:** Check per-class metrics, not just overall accuracy.

**Application:**
```python
# ALWAYS print classification report
print(classification_report(y_true, y_pred, target_names=label_classes))

# NOT just accuracy
print(f"Accuracy: {accuracy}")  # Can be misleading!
```

### 3. Visualize Data Distribution Early

**Mistake:** Didn't notice neutral class dominance until training failed.

**Learning:** Always check class distribution before training.

**Application:**
```python
# In train_model.py, lines 210-212
for label_name, count in label_counts.items():
    percentage = (count / len(X_all) * 100)
    print(f"  {label_name}: {count} ({percentage:.1f}%)")
```

### 4. Make Code Resilient to External Changes

**Mistake:** Assumed Label Studio format wouldn't change.

**Learning:** External tools evolve; handle multiple formats.

**Application:**
```python
# Try new format first, fall back to old
try:
    data = new_format(json)
except KeyError:
    data = old_format(json)
```

### 5. Separate Training from Inference Concerns

**Mistake:** Tried to use same approach for training and detection.

**Learning:** Training optimizes classification; inference needs detection.

**Application:**
- **Training**: Focus on "what is this stroke?" (pure classification)
- **Inference**: Focus on "when do strokes occur?" (detection + classification)

### 6. Experiment Tracking is Worth It

**Mistake:** Initially didn't use MLflow, had to remember what parameters worked.

**Learning:** Track everything from the start.

**Application:**
- Every training run logged
- Easy to see what worked
- Can recover best model anytime
- Compare approaches objectively

### 7. Document What Didn't Work

**Mistake:** Almost forgot why neutral class was removed.

**Learning:** Document failures as prominently as successes.

**Application:** This document! Future developers won't repeat the same mistakes.

## Future Improvements

### Short Term (Easy Wins)

1. **Data augmentation:**
   ```python
   # Time warping: Slightly speed up/slow down sequences
   # Add noise to landmarks
   # Flip left/right (backhand ↔ fronthand for opposite-handed players)
   ```

2. **Ensemble methods:**
   - Train multiple models with different random seeds
   - Average predictions for robustness

3. **Attention mechanism:**
   - Add attention to LSTM to focus on key frames
   - Visualize which frames are most important

### Medium Term (More Data Needed)

4. **Stroke quality assessment:**
   - Not just "what stroke" but "how good is it"
   - Requires expert-labeled quality ratings

5. **Player identification:**
   - Track multiple players in doubles
   - Assign strokes to specific player

6. **Ball tracking integration:**
   - Combine pose with ball position
   - Improve stroke boundary detection

### Long Term (Significant Development)

7. **Real-time inference:**
   - Optimize for live video streaming
   - Reduce latency (currently ~5fps for detection)

8. **Mobile deployment:**
   - Convert model to TensorFlow Lite
   - Run on smartphones for instant feedback

9. **Transfer learning:**
   - Pre-train on large sports dataset
   - Fine-tune on tennis strokes
   - Improve generalization

### Research Directions

10. **Stroke prediction:**
    - Predict stroke type before it happens (from setup)
    - Could help with coaching

11. **Comparative analysis:**
    - Compare player's stroke to professional players
    - Identify technique differences

12. **Automated highlight generation:**
    - Detect "good shots" based on stroke quality + rally context
    - Create video highlights automatically

## Summary

### Key Takeaways

1. **Skip the neutral class** - Focus on what you annotated, not what you didn't
2. **Check class distribution** - Before training, not after
3. **Separate training and inference** - Different objectives need different approaches
4. **Track everything with MLflow** - You'll thank yourself later
5. **Handle format variations** - External tools change, be resilient
6. **Visualize before optimizing** - See the data, understand the problem

### What Made This Project Work

- **MediaPipe**: Robust pose estimation out of the box
- **LSTM**: Perfect fit for sequential stroke data
- **Skipping neutral**: The breakthrough that made everything work
- **Two-stage approach**: Training on clean data, inference with post-processing
- **MLflow**: Made experimentation systematic and reproducible

### What We'd Do Differently

- Start with class distribution analysis (would've caught neutral problem early)
- Use MLflow from day one (not added midway)
- Test inference scenario earlier (would've informed training approach)
- Document decisions as we go (not reconstruct later)

### For Future Developers

If you're continuing this project:

1. **Read this document first** - Avoid repeating our mistakes
2. **Check [USAGE_GUIDE.md](USAGE_GUIDE.md)** - For step-by-step examples
3. **Review MLflow runs** - See what worked in practice
4. **Start small** - Get something working before adding complexity
5. **Question assumptions** - Just because we did it doesn't mean it's optimal

Good luck! 🎾
