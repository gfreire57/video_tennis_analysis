# Pose Features Explained: How the Model Uses Body Geometry

## Overview

This document explains **exactly how the model uses pose estimation** to classify tennis strokes. It answers:

- ✅ What features are extracted from MediaPipe pose estimation?
- ✅ How does body geometry (arms, legs, torso) factor into training?
- ✅ How does the LSTM learn stroke patterns from body positions?
- ✅ What does the model actually "see" when classifying strokes?

---

## The Big Picture

```
Video Frame → MediaPipe Pose → 33 Body Landmarks → 132 Features per Frame
                                                           ↓
                                        20 Frames (0.67 seconds of motion)
                                                           ↓
                                        Shape: (20, 132) = Sequence
                                                           ↓
                                        LSTM Neural Network
                                                           ↓
                                        Learns: "This sequence of body positions = FRONTHAND"
```

**Key insight:** The model learns **temporal patterns of body geometry**. It recognizes that a fronthand stroke has a specific sequence of body positions over ~0.67 seconds.

---

## Part 1: What Features Are Extracted?

### MediaPipe Pose Landmarks

MediaPipe detects **33 body landmarks** on every frame:

```
Landmarks (33 total):

HEAD/FACE:
0  - Nose
1  - Left eye (inner)
2  - Left eye
3  - Left eye (outer)
4  - Right eye (inner)
5  - Right eye
6  - Right eye (outer)
7  - Left ear
8  - Right ear
9  - Mouth (left)
10 - Mouth (right)

UPPER BODY:
11 - Left shoulder
12 - Right shoulder
13 - Left elbow
14 - Right elbow
15 - Left wrist
16 - Right wrist

HANDS:
17 - Left pinky
18 - Right pinky
19 - Left index
20 - Right index
21 - Left thumb
22 - Right thumb

LOWER BODY:
23 - Left hip
24 - Right hip
25 - Left knee
26 - Right knee
27 - Left ankle
28 - Right ankle

FEET:
29 - Left heel
30 - Right heel
31 - Left foot index
32 - Right foot index
```

### Each Landmark Has 4 Values

For **each landmark**, MediaPipe provides:

1. **x** - Horizontal position (0.0 = left edge, 1.0 = right edge)
2. **y** - Vertical position (0.0 = top edge, 1.0 = bottom edge)
3. **z** - Depth (distance from camera, approximate)
4. **visibility** - How confident MediaPipe is (0.0 = hidden, 1.0 = clearly visible)

**Total features per frame:** 33 landmarks × 4 values = **132 features**

### Example: Right Wrist During Fronthand

```
Frame 1 (start of stroke):
  right_wrist (landmark 16):
    x = 0.30 (left side of body)
    y = 0.45 (mid-height)
    z = -0.2 (close to body)
    visibility = 0.95 (clearly visible)

Frame 10 (mid-stroke):
  right_wrist:
    x = 0.55 (moving right)
    y = 0.50 (slightly higher)
    z = 0.3 (extended away from body)
    visibility = 0.98

Frame 20 (end of stroke):
  right_wrist:
    x = 0.75 (far right)
    y = 0.60 (higher)
    z = 0.5 (fully extended)
    visibility = 0.92
```

**The model sees:** Right wrist moved from left (0.30) to right (0.75), rising from mid-height to higher, and extending away from body. This pattern = FRONTHAND.

---

## Part 2: How Body Geometry Is Used

### The Model Learns Geometric Patterns

The LSTM doesn't just see isolated positions. It learns **how body parts move together over time**.

### Fronthand Stroke Pattern (What the Model Learns)

```
Geometric sequence detected:

Frame 1-5 (Preparation):
  - Right shoulder (landmark 12): Rotates backward (x decreases)
  - Right elbow (14): Bends (distance to shoulder decreases)
  - Right wrist (16): Pulled back (x = 0.25-0.30)
  - Hips (23, 24): Rotate backward (right hip x > left hip x)

Frame 6-12 (Forward swing):
  - Right shoulder: Rotates forward (x increases)
  - Right elbow: Extends (distance to wrist increases)
  - Right wrist: Accelerates forward (x = 0.30 → 0.60)
  - Hips: Rotate forward (left hip x > right hip x)
  - Left foot (31): Plants (y stable)
  - Right foot (32): Pushes (y changes)

Frame 13-20 (Follow-through):
  - Right wrist: Continues across body (x = 0.60 → 0.75)
  - Right elbow: Fully extended (z increases)
  - Shoulders: Fully rotated (right shoulder x > left shoulder x)
  - Torso: Leaning forward (shoulder y > hip y)
```

**Key insight:** The model learns that THIS specific sequence of body positions = FRONTHAND.

### Backhand Stroke Pattern

```
Geometric sequence (DIFFERENT from fronthand):

Preparation:
  - LEFT shoulder (11): Rotates backward
  - LEFT elbow (13): Bends
  - LEFT wrist (15): Pulled back (x = 0.70-0.75)
  - Hips: Rotate opposite direction

Forward swing:
  - LEFT arm extends (not right)
  - Body rotates OPPOSITE direction
  - Weight shifts to RIGHT foot (opposite of fronthand)

Follow-through:
  - LEFT wrist crosses body (x decreases)
  - Shoulders rotate opposite
```

**Key difference:** Backhand uses LEFT arm extending from right to left, while fronthand uses RIGHT arm extending from left to right.

### How the Model Distinguishes Strokes

The LSTM learns **discriminative patterns**:

| Feature Pattern | Fronthand | Backhand |
|----------------|-----------|----------|
| Primary arm used | Right (landmarks 12, 14, 16) | Left (landmarks 11, 13, 15) |
| Wrist motion direction | Left → Right (x increases) | Right → Left (x decreases) |
| Hip rotation | Right hip back → forward | Left hip back → forward |
| Shoulder rotation | Right shoulder leads | Left shoulder leads |
| Weight transfer | Left foot → Right foot | Right foot → Left foot |

The model doesn't need to be explicitly told these rules. The LSTM **automatically discovers** these patterns from the training data.

---

## Part 3: The Sliding Window Approach

### Why 20 Frames?

**20 frames at 30 FPS = 0.67 seconds**

This captures:
- Preparation (backswing): ~5-7 frames
- Forward swing (contact): ~6-10 frames
- Follow-through: ~5-7 frames

**Total: Complete stroke motion**

### How Sequences Are Created

**Code location:** [src/train_model.py:134-223](src/train_model.py#L134-L223)

```python
def create_sequences_from_frames(all_frames, all_labels, window_size=20, overlap=10):
    """
    Creates sliding window sequences from pose features

    Input:
      all_frames: List of frames, each frame = 132 features (33 landmarks × 4 values)
      all_labels: List of labels for each frame
      window_size: 20 frames (0.67 seconds)
      overlap: 10 frames (50% overlap for better detection)

    Output:
      sequences: Shape (num_sequences, 20, 132)
                 Each sequence = 20 frames of body positions
      labels: One label per sequence (majority vote)
    """
```

### Example: Creating a Fronthand Sequence

```
Video has 100 frames total:

Frames 0-19 (first window):
  - Each frame has 132 features
  - Stack into shape (20, 132)
  - Label: If >10 frames labeled "fronthand" → sequence label = FRONTHAND

Frames 10-29 (second window, 50% overlap):
  - Stack into shape (20, 132)
  - Label: Majority vote again

Frames 20-39 (third window):
  - Stack into shape (20, 132)
  - ...and so on
```

**Result:** One 100-frame video → Multiple (20, 132) sequences for training

### Visualization of a Sequence

```
Sequence shape: (20, 132)

Row 0:  [x0, y0, z0, vis0, x1, y1, z1, vis1, ..., x32, y32, z32, vis32]  ← Frame 0
Row 1:  [x0, y0, z0, vis0, x1, y1, z1, vis1, ..., x32, y32, z32, vis32]  ← Frame 1
Row 2:  [x0, y0, z0, vis0, x1, y1, z1, vis1, ..., x32, y32, z32, vis32]  ← Frame 2
...
Row 19: [x0, y0, z0, vis0, x1, y1, z1, vis1, ..., x32, y32, z32, vis32]  ← Frame 19

                       ↓
               Fed into LSTM
                       ↓
       LSTM learns: "This pattern = FRONTHAND"
```

---

## Part 4: How the LSTM Learns

### LSTM Architecture

**Code location:** [src/train_model.py:301-339](src/train_model.py#L301-L339)

```python
model = keras.Sequential([
    keras.layers.Input(shape=(20, 132)),  # 20 frames, 132 features per frame

    # First LSTM layer (128 units)
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    # Second LSTM layer (96 units)
    keras.layers.LSTM(96, return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    # Third LSTM layer (64 units)
    keras.layers.LSTM(64, return_sequences=False),  # Outputs single vector
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    # Dense classification layers
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')  # Output: [prob_backhand, prob_fronthand]
])
```

### What Each LSTM Layer Learns

**Layer 1 (128 units):**
- Learns basic motion patterns
- Examples:
  - "Wrist moving right"
  - "Elbow bending"
  - "Hip rotating"
- Outputs: 128 features per timestep

**Layer 2 (96 units):**
- Learns combined patterns
- Examples:
  - "Wrist moving right WHILE elbow extends"
  - "Hip rotation WHILE shoulder rotates"
  - "Weight shift from left to right foot"
- Outputs: 96 features per timestep

**Layer 3 (64 units):**
- Learns complete stroke patterns
- Examples:
  - "Right-arm dominant stroke with hip rotation" = FRONTHAND
  - "Left-arm dominant stroke with opposite rotation" = BACKHAND
- Outputs: Single 64-dimensional vector (summary of entire sequence)

**Dense Layers:**
- Maps the 64-dimensional summary to class probabilities
- Output: `[0.15, 0.85]` → 15% backhand, 85% fronthand

### How LSTM Processes Temporal Information

```
Input sequence (20 frames):

Frame 0  → LSTM reads position 0 → Hidden state updates
Frame 1  → LSTM reads position 1 → Hidden state updates (remembers frame 0)
Frame 2  → LSTM reads position 2 → Hidden state updates (remembers frames 0-1)
...
Frame 19 → LSTM reads position 19 → Final hidden state (remembers ALL frames)

Final hidden state = Compressed representation of entire motion
                   ↓
              Dense layers
                   ↓
           [prob_backhand, prob_fronthand]
```

**Key capability:** LSTM **remembers early frames** when processing later frames. This allows it to learn patterns like:

- "Wrist was LEFT (frame 0), now moving RIGHT (frame 10), reached FAR RIGHT (frame 19)" = FRONTHAND
- "Wrist was RIGHT (frame 0), now moving LEFT (frame 10), reached FAR LEFT (frame 19)" = BACKHAND

---

## Part 5: Complete Data Flow Example

### From Video to Prediction: Step by Step

**1. Video Input**

```
video.mp4 (30 FPS, 3 minutes)
  ↓
5400 total frames
```

**2. Pose Extraction**

**Code:** [src/train_model.py:246-298](src/train_model.py#L246-L298) - `PoseExtractor.extract_landmarks()`

```
For each frame:
  MediaPipe Pose detects 33 landmarks
  Extract x, y, z, visibility for each

Frame 0: [132 features]
Frame 1: [132 features]
...
Frame 5399: [132 features]

Result: Array of shape (5400, 132)
```

**3. Sequence Creation**

**Code:** [src/train_model.py:134-223](src/train_model.py#L134-L223) - `create_sequences_from_frames()`

```
Sliding window (size=20, overlap=10):

Sequence 0: Frames 0-19   → Shape (20, 132)
Sequence 1: Frames 10-29  → Shape (20, 132)
Sequence 2: Frames 20-39  → Shape (20, 132)
...

Result: ~540 sequences of shape (20, 132)
```

**4. Training**

**Code:** [src/train_model.py:561-587](src/train_model.py#L561-L587)

```
For each sequence:
  Input: (20, 132) array
  Label: "fronthand" or "backhand"

Model learns:
  "When I see THIS pattern of 132 features changing over 20 frames → Label = FRONTHAND"
  "When I see THAT pattern of 132 features changing over 20 frames → Label = BACKHAND"
```

**5. Prediction** (during detection)

**Code:** [src/detect_strokes.py:158-249](src/detect_strokes.py#L158-L249) - `detect_strokes()`

```
New video frame → Extract 132 features → Add to buffer

When buffer has 20 frames:
  Create sequence of shape (20, 132)
  Feed to model
  Model outputs: [0.12, 0.88] → 12% backhand, 88% fronthand
  Prediction: FRONTHAND (confidence = 88%)
```

---

## Part 6: What Makes Different Strokes Distinctive?

### Body Geometry Patterns the Model Learns

**Fronthand vs Backhand (Key Differences):**

| Body Part | Fronthand Pattern | Backhand Pattern |
|-----------|-------------------|------------------|
| **Right wrist (16)** | x increases (0.3 → 0.7) | x stable or decreases |
| **Left wrist (15)** | x stable or decreases | x decreases (0.7 → 0.3) |
| **Right elbow (14)** | Extends (z increases) | Stays bent |
| **Left elbow (13)** | Stays bent | Extends (z increases) |
| **Right shoulder (12)** | Rotates forward (x increases) | Rotates backward |
| **Left shoulder (11)** | Rotates backward | Rotates forward |
| **Hips (23, 24)** | Right hip forward | Left hip forward |
| **Torso rotation** | Clockwise (top view) | Counter-clockwise |
| **Weight transfer** | Left → Right foot | Right → Left foot |

### Serve Pattern (Different from Both)

```
Distinctive features:

Frame 0-8 (Ball toss):
  - Right wrist (16): Moves UP (y decreases from 0.6 → 0.2)
  - Right arm: Fully extended upward (z increases)
  - Left wrist (15): Releases ball (visibility may drop)

Frame 9-15 (Racket drop):
  - Right elbow (14): Bends sharply (z decreases)
  - Right wrist: Behind head (x = 0.4-0.5, y = 0.2-0.3)
  - Body: Arches backward (shoulders y < hips y)

Frame 16-20 (Contact):
  - Right wrist: Explosively rises (y decreases to minimum)
  - Right arm: Fully extends upward (z maximum)
  - Body: Extends upward (all y values decrease)
  - Left foot: May leave ground (ankle visibility drops)
```

**Key difference from groundstrokes:** Vertical motion (y-axis) dominates, rather than horizontal (x-axis).

---

## Part 7: Code Walkthrough

### Extract Pose Features

**File:** [src/train_model.py](src/train_model.py#L246-L298)

```python
class PoseExtractor:
    def extract_landmarks(self, image):
        """
        Extract 132 features from one frame

        Returns:
          features: List of 132 floats
                   [x0, y0, z0, vis0, x1, y1, z1, vis1, ..., x32, y32, z32, vis32]
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            features = []
            for landmark in results.pose_landmarks.landmark:
                features.extend([
                    landmark.x,          # Horizontal position
                    landmark.y,          # Vertical position
                    landmark.z,          # Depth
                    landmark.visibility  # Confidence
                ])
            return features  # Length: 33 landmarks × 4 = 132
        else:
            return None  # No pose detected
```

### Create Training Sequences

**File:** [src/train_model.py](src/train_model.py#L134-L223)

```python
def create_sequences_from_frames(all_frames, all_labels, window_size=20, overlap=10):
    """
    Create (20, 132) sequences from individual frames

    Args:
      all_frames: List of frames, each frame = [132 features]
      all_labels: List of labels, each label = "fronthand" or "backhand"
      window_size: 20 frames
      overlap: 10 frames (50% overlap)

    Returns:
      X: Array of shape (num_sequences, 20, 132)
      y: Array of labels (num_sequences,)
    """
    sequences = []
    labels = []

    for start_idx in range(0, len(all_frames) - window_size + 1, window_size - overlap):
        end_idx = start_idx + window_size

        # Extract 20 frames
        sequence = all_frames[start_idx:end_idx]  # Shape: (20, 132)

        # Get majority label
        window_labels = all_labels[start_idx:end_idx]
        majority_label = max(set(window_labels), key=window_labels.count)

        sequences.append(sequence)
        labels.append(majority_label)

    return np.array(sequences), np.array(labels)
```

### Model Training

**File:** [src/train_model.py](src/train_model.py#L561-L587)

```python
# Training loop
history = model.fit(
    X_train,  # Shape: (num_sequences, 20, 132)
    y_train,  # Shape: (num_sequences,) - one label per sequence
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    class_weight=class_weights,  # Balance fronthand/backhand
    callbacks=[early_stopping, reduce_lr]
)

# What happens during training:
# 1. Model receives batch of 32 sequences, each shape (20, 132)
# 2. LSTM processes each sequence, learning patterns
# 3. Model outputs predictions: [prob_backhand, prob_fronthand]
# 4. Loss calculated: How far off from true labels?
# 5. Backpropagation: Adjust weights to reduce loss
# 6. Repeat for 150 epochs
```

### Stroke Detection

**File:** [src/detect_strokes.py](src/detect_strokes.py#L158-L249)

```python
def detect_strokes(video_path, model, label_classes, config=CONFIG):
    """
    Detect strokes in video using trained model
    """
    pose_extractor = PoseExtractor()
    frame_buffer = []  # Will hold 20 frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract pose features
        features = pose_extractor.extract_landmarks(frame)  # 132 features

        if features is not None:
            frame_buffer.append(features)

            # When we have 20 frames, make prediction
            if len(frame_buffer) >= CONFIG['window_size']:
                sequence = np.array(frame_buffer[-20:])  # Shape: (20, 132)
                sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 20, 132)

                # Model prediction
                prediction = model.predict(sequence, verbose=0)
                # prediction = [[0.15, 0.85]] → 15% backhand, 85% fronthand

                class_idx = np.argmax(prediction[0])
                confidence = prediction[0][class_idx]

                if confidence >= CONFIG['confidence_threshold']:
                    # Detected a stroke!
                    stroke_type = label_classes[class_idx]
                    # ... record stroke ...
```

---

## Part 8: Summary

### What Features Are Extracted?

**132 features per frame:**
- 33 body landmarks (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles, etc.)
- 4 values per landmark (x, y, z, visibility)
- **Total: 33 × 4 = 132 features**

### How Body Geometry Is Used?

**The model learns patterns in how body parts move together:**
- Fronthand: Right arm extends left → right, hips rotate, weight shifts left → right
- Backhand: Left arm extends right → left, opposite hip rotation, opposite weight shift
- Serve: Vertical motion dominates, arm goes up then extends upward

**LSTM learns these patterns automatically** from training data - no manual feature engineering needed.

### How Does the LSTM Work?

**Temporal pattern recognition:**
1. Receives sequence of 20 frames (0.67 seconds of motion)
2. Each frame = 132 features (body positions)
3. LSTM learns: "THIS sequence of positions = FRONTHAND"
4. Outputs probability: [12% backhand, 88% fronthand]

### The Complete Pipeline

```
Video → MediaPipe (33 landmarks) → 132 features per frame → Sliding window (20 frames)
  → Shape (20, 132) → LSTM → Probability [backhand, fronthand] → Prediction
```

---

## Part 9: Visualizing What the Model Sees

### Understanding Your Annotated Videos

When you run detection with visualization enabled:

```bash
poetry run python src/detect_strokes.py video.mp4
```

You can **literally see** what features the model uses:

**Skeleton overlay shows:**
- Green lines = Landmark connections (MediaPipe detections)
- These ARE the features (33 landmark positions)

**During a fronthand:**
- Watch right wrist (green dot) move from left to right
- Watch right elbow extend
- Watch hips rotate (left hip moves backward, right hip forward)

**During a backhand:**
- Watch LEFT wrist move from right to left
- Watch LEFT elbow extend
- Watch hips rotate opposite direction

**The model is learning:** "When I see wrist landmark 16 moving in THIS pattern while hip landmarks 23-24 rotate like THAT → It's a FRONTHAND"

### Try This Experiment

1. **Create annotated video:**
   ```bash
   poetry run python src/detect_strokes.py your_video.mp4
   ```

2. **Watch the skeleton during detected strokes:**
   - Pause at stroke start
   - Play slowly through the stroke
   - Notice how landmarks move

3. **You'll see exactly what the model sees:**
   - The 33 landmark positions
   - How they change frame by frame
   - The patterns that distinguish fronthand from backhand

---

## Part 10: Why This Approach Works

### Advantages of Pose-Based Features

**1. View-invariant:** Works from any camera angle
   - Landmarks are relative positions (0.0-1.0), not absolute pixels
   - Camera on left or right doesn't matter

**2. Player-size invariant:** Works regardless of player distance
   - Landmarks normalized to frame size
   - Small player or large player gives same features

**3. Lighting-robust:** Works in dark or bright conditions
   - MediaPipe uses ML for detection, not simple color/brightness
   - As long as pose is visible, features extracted correctly

**4. Captures biomechanics:** Models actual movement patterns
   - Not just "image looks like fronthand"
   - Learns "body moves like fronthand stroke"

### Comparison to Other Approaches

**Image-based CNN (not used):**
- Needs thousands of images
- Sensitive to camera angle, lighting, background
- Learns: "This IMAGE looks like fronthand"

**Pose-based LSTM (what we use):**
- Needs fewer examples (body geometry patterns are consistent)
- Robust to camera angle, lighting, background
- Learns: "This MOTION looks like fronthand"

---

## Questions & Answers

### Q: Does the model understand tennis?

**A:** No. The model doesn't "know" what tennis is. It learns:
- "Pattern A in landmarks 11-16 over 20 frames = Label: fronthand"
- "Pattern B in landmarks 11-16 over 20 frames = Label: backhand"

It discovers that fronthand has right-arm dominant motion, backhand has left-arm dominant motion - but it doesn't "understand" this conceptually.

### Q: What if the player is left-handed?

**A:** The model will **struggle** because left-handed players have opposite patterns:
- Left-handed fronthand = Right-handed backhand pattern
- Left-handed backhand = Right-handed fronthand pattern

**Solution:** Either:
1. Train separate model on left-handed data
2. Add left-handed examples to training data
3. Use mirror-augmentation (flip landmarks horizontally)

### Q: Why 20 frames specifically?

**A:** Empirical choice. At 30 FPS:
- 10 frames = 0.33s (too short, misses follow-through)
- 20 frames = 0.67s (captures full stroke)
- 30 frames = 1.00s (too long, includes non-stroke movement)

You can experiment with different window sizes via CONFIG parameter.

### Q: Can the model detect half-swings or incomplete strokes?

**A:** It depends on training data:
- If trained only on full strokes → Will miss partial swings
- If trained on varied strokes (full, half, practice) → Better generalization

The model learns what patterns you show it in training data.

---

## Next Steps

### To Better Understand Your Model:

1. **Watch annotated videos:**
   ```bash
   poetry run python src/detect_strokes.py video.mp4
   ```
   Observe skeleton during detected strokes.

2. **Check confusion matrix in MLflow:**
   - Are fronthands confused with backhands?
   - This tells you if body geometry patterns are distinctive enough

3. **Experiment with features:**
   - Try removing hand landmarks (17-22) - does accuracy drop?
   - Try using only upper body (11-16) - does it still work?
   - This tells you which landmarks are most important

4. **Visualize pose during different strokes:**
   ```bash
   poetry run python src/visualize_pose.py video.mp4 --max-frames 300
   ```
   See if fronthand/backhand LOOK geometrically different to you.

---

## Technical References

### Code Locations:

- **Pose extraction:** [src/train_model.py:246-298](src/train_model.py#L246-L298)
- **Sequence creation:** [src/train_model.py:134-223](src/train_model.py#L134-L223)
- **Model architecture:** [src/train_model.py:301-339](src/train_model.py#L301-L339)
- **Training loop:** [src/train_model.py:561-587](src/train_model.py#L561-L587)
- **Stroke detection:** [src/detect_strokes.py:158-249](src/detect_strokes.py#L158-L249)

### Key Configuration:

```python
CONFIG = {
    'window_size': 20,        # Number of frames per sequence
    'overlap': 10,            # Overlap between windows
    'input_features': 132,    # 33 landmarks × 4 values
}
```

### MediaPipe Documentation:

- [Pose Landmark Detection](https://google.github.io/mediapipe/solutions/pose.html)
- [Landmark Index Reference](https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d)

---

**You now understand exactly how pose estimation features are being used to train the stroke classification model!** 🎾
