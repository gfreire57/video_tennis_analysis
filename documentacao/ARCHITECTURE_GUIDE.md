# Architecture & Theory Guide

This document provides a visual explanation of how the Tennis Stroke Recognition system works, including the data flow, model architecture, and algorithms.

## Table of Contents

- [System Overview](#system-overview)
- [Part 1: Data Preparation](#part-1-data-preparation)
- [Part 2: Training Architecture](#part-2-training-architecture)
- [Part 3: Inference Architecture](#part-3-inference-architecture)
- [Mathematical Foundations](#mathematical-foundations)
- [Why This Architecture Works](#why-this-architecture-works)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TENNIS STROKE RECOGNITION                    │
│                         PIPELINE                                │
└─────────────────────────────────────────────────────────────────┘

INPUT: Annotated Videos
   ↓
┌──────────────────────────┐
│  POSE EXTRACTION         │  ← MediaPipe extracts body landmarks
│  (MediaPipe)             │
└──────────┬───────────────┘
           │ Output: Sequence of 132-dim vectors (one per frame)
           ↓
┌──────────────────────────┐
│  SLIDING WINDOW          │  ← Create fixed-length sequences
│  (window_size=30)        │
└──────────┬───────────────┘
           │ Output: Many 30-frame windows
           ↓
┌──────────────────────────┐
│  LABEL ASSIGNMENT        │  ← Assign stroke labels to windows
│  (Majority Voting)       │
└──────────┬───────────────┘
           │ Output: (sequence, label) pairs
           ↓
┌──────────────────────────┐
│  LSTM TRAINING           │  ← Learn temporal patterns
│  (3-layer network)       │
└──────────┬───────────────┘
           │ Output: Trained model
           ↓
┌──────────────────────────┐
│  INFERENCE               │  ← Detect strokes in new videos
│  (Sliding + Merging)     │
└──────────┬───────────────┘
           │
           ▼
OUTPUT: Stroke timeline, statistics, JSON
```

---

## Part 1: Data Preparation

### 1.1 Pose Extraction with MediaPipe

MediaPipe detects 33 body landmarks in each video frame:

```
         HEAD (0)
            •
            │
    (11)•───┼───•(12)  ← Shoulders
        │   │   │
    (13)•   │   •(14)  ← Elbows
        │   │   │
    (15)•   │   •(16)  ← Wrists
            │
    (23)•───┼───•(24)  ← Hips
        │   │   │
    (25)•   │   •(26)  ← Knees
        │   │   │
    (27)•   │   •(28)  ← Ankles

Each landmark has 4 values:
  - x: horizontal position (0-1)
  - y: vertical position (0-1)
  - z: depth (relative)
  - visibility: confidence (0-1)

Total per frame: 33 landmarks × 4 values = 132 features
```

**Example raw data for one frame:**
```python
Frame 0: [x₀, y₀, z₀, v₀, x₁, y₁, z₁, v₁, ..., x₃₂, y₃₂, z₃₂, v₃₂]
         ↑                                                        ↑
      Nose (landmark 0)                          Right ankle (landmark 32)

Shape: (132,)  # One vector per frame
```

**For entire video:**
```python
Video (300 frames @ 30fps = 10 seconds):
[
  Frame 0:   [132 features],
  Frame 1:   [132 features],
  Frame 2:   [132 features],
  ...
  Frame 299: [132 features]
]

Shape: (300, 132)  # 300 frames × 132 features
```

### 1.2 Annotation Mapping to Frames

Label Studio annotations are in **seconds**, we convert to **frame numbers**:

```
Video: 30 FPS, 10 seconds total

Annotations (from Label Studio):
┌─────────────────────────────────────────────────────────────────┐
│ Time:     0s      1s      2s      3s      4s      5s      6s    │
│ Label:    [   fronthand   ] [   backhand    ] [ fronthand  ]   │
│           ▲───────────────▲ ▲───────────────▲ ▲────────────▲   │
│         start=1.0      end=2.5           end=4.8                │
└─────────────────────────────────────────────────────────────────┘

Convert to frame numbers (fps = 30):
┌─────────────────────────────────────────────────────────────────┐
│ Frames:   0    30    60    90   120   150   180               │
│           │     │     │     │     │     │     │                │
│ Label:    [  fronthand ] [  backhand   ] [fronthand]           │
│           ▲────────────▲ ▲─────────────▲ ▲──────────▲          │
│         fr=30        fr=75          fr=144                      │
└─────────────────────────────────────────────────────────────────┘

Frame-level labels:
Frame 0-29:   neutral (no annotation)
Frame 30-75:  fronthand
Frame 76-89:  neutral
Frame 90-144: backhand
Frame 145-165: neutral
Frame 166-195: fronthand
Frame 196-299: neutral
```

### 1.3 Sliding Window Approach (Training)

We create **fixed-length sequences** from the continuous video using a sliding window:

```
Parameters:
  window_size = 30 frames  (≈1 second @ 30fps)
  overlap = 15 frames      (50% overlap)
  stride = window_size - overlap = 15 frames

Video frames with labels:
┌──────────────────────────────────────────────────────────────────┐
Frame:  0   15  30  45  60  75  90  105 120 135 150 165 180 195
Label:  N   N   F   F   F   F   B   B   B   B   N   N   F   F
        └───────┬───────────┘
                │
        Window 1: Frames 0-29 (majority: F=fronthand)
                    └───────┬───────────┘
                            │
                    Window 2: Frames 15-44 (majority: F=fronthand)
                                └───────┬───────────┘
                                        │
                                Window 3: Frames 30-59 (majority: F=fronthand)

Legend: N=neutral, F=fronthand, B=backhand
```

**Detailed sliding window visualization:**

```
Window 1: Frames 0-29
┌─────────────────────────────────────────────────────────┐
│ Frames: [0,1,2,...,27,28,29]                           │
│ Labels: [N,N,N,...,F, F, F ]                           │
│                                                         │
│ Count:  neutral=10, fronthand=20                       │
│ Majority: fronthand (20/30 = 67% > 50% ✓)             │
│                                                         │
│ BUT: Skip because majority_label == 'neutral' would    │
│      be False here (majority is fronthand)             │
│                                                         │
│ Action: CREATE TRAINING SAMPLE                         │
│   X = [30 frames × 132 features] shape: (30, 132)     │
│   y = 'fronthand'                                      │
└─────────────────────────────────────────────────────────┘

Move forward by stride=15 frames ↓

Window 2: Frames 15-44
┌─────────────────────────────────────────────────────────┐
│ Frames: [15,16,17,...,42,43,44]                        │
│ Labels: [N, F, F,..., F, F, F ]                        │
│                                                         │
│ Count:  neutral=5, fronthand=25                        │
│ Majority: fronthand (25/30 = 83% > 50% ✓)             │
│                                                         │
│ Action: CREATE TRAINING SAMPLE                         │
│   X = [30 frames × 132 features]                       │
│   y = 'fronthand'                                      │
└─────────────────────────────────────────────────────────┘

Move forward by stride=15 frames ↓

... continue until end of video ...

Window N: Frames 270-299
┌─────────────────────────────────────────────────────────┐
│ Frames: [270,271,272,...,297,298,299]                  │
│ Labels: [N,  N,  N,...,  N,  N,  N ]                   │
│                                                         │
│ Count:  neutral=30                                     │
│ Majority: neutral (30/30 = 100% > 50% ✓)              │
│                                                         │
│ BUT: majority_label == 'neutral'                       │
│                                                         │
│ Action: SKIP (don't create training sample)            │
│         This is the key fix!                           │
└─────────────────────────────────────────────────────────┘
```

**Why 50% overlap?**
```
Without overlap (stride = 30):
┌──────────┐          ┌──────────┐          ┌──────────┐
│ Window 1 │          │ Window 2 │          │ Window 3 │
└──────────┘          └──────────┘          └──────────┘
0         29         30         59         60         89

Problem: Miss stroke if it starts at frame 20 and ends at frame 40
         (split between two windows, neither has majority)

With overlap (stride = 15):
┌──────────┐
│ Window 1 │
└──────────┘
0         29
     ┌──────────┐
     │ Window 2 │
     └──────────┘
    15         44
          ┌──────────┐
          │ Window 3 │
          └──────────┘
         30         59

Benefit: Stroke from frame 20-40 is captured by Window 2
         (has 25/30 frames = majority)
         More robust + more training samples
```

### 1.4 Label Assignment (Majority Voting)

For each window, we assign a label based on which stroke appears most frequently:

```python
# Pseudocode for label assignment

window_frames = [frame_30, frame_31, ..., frame_59]  # 30 frames
window_labels = ['F', 'F', 'F', ..., 'B', 'B', 'B']  # Labels for each frame

# Count labels
count_fronthand = 18
count_backhand = 12
count_neutral = 0

# Find majority
majority_label = 'fronthand'  # Has most frames (18/30)
majority_count = 18

# Check if majority is >50% and not neutral
if majority_count > 15 and majority_label != 'neutral':  # 15 = 30 * 0.5
    # Use this window for training
    X.append(window_frames)
    y.append('fronthand')
else:
    # Skip this window
    pass
```

**Visual example of majority voting:**

```
Window with clear majority:
┌────────────────────────────────────────────────────────┐
│ Frame: [ 0  1  2  3  4 ... 25 26 27 28 29]           │
│ Label: [ F  F  F  F  F ... F  F  F  F  F ]           │
│                                                        │
│ Counts: fronthand=30, others=0                        │
│ Majority: fronthand (30/30 = 100%)                    │
│ Result: ✓ Use for training with label='fronthand'     │
└────────────────────────────────────────────────────────┘

Window with borderline case:
┌────────────────────────────────────────────────────────┐
│ Frame: [ 0  1  2  3  4 ... 25 26 27 28 29]           │
│ Label: [ F  F  F  F  F ... F  B  B  B  B ]           │
│                                                        │
│ Counts: fronthand=20, backhand=10                     │
│ Majority: fronthand (20/30 = 67%)                     │
│ Result: ✓ Use for training with label='fronthand'     │
│         (>50% threshold met)                          │
└────────────────────────────────────────────────────────┘

Window with no clear majority:
┌────────────────────────────────────────────────────────┐
│ Frame: [ 0  1  2  3  4 ... 25 26 27 28 29]           │
│ Label: [ F  F  F  F  F ... B  B  B  B  B ]           │
│                                                        │
│ Counts: fronthand=14, backhand=16                     │
│ Majority: backhand (16/30 = 53%)                      │
│ Result: ✓ Use for training with label='backhand'      │
│         (>50% threshold met, but close to boundary)   │
└────────────────────────────────────────────────────────┘

Window at stroke boundary (ambiguous):
┌────────────────────────────────────────────────────────┐
│ Frame: [ 0  1  2  3  4 ... 25 26 27 28 29]           │
│ Label: [ N  N  N  N  N ... F  F  F  F  F ]           │
│                                                        │
│ Counts: neutral=20, fronthand=10                      │
│ Majority: neutral (20/30 = 67%)                       │
│ Result: ✗ SKIP (majority is neutral)                  │
│         This window is dropped from training          │
└────────────────────────────────────────────────────────┘
```

### 1.5 Final Training Dataset

After processing all videos with sliding windows:

```
Original data:
  Video 1: 3600 frames → ~240 windows
  Video 2: 5400 frames → ~360 windows
  Total windows: ~600

After filtering (remove neutral):
  Windows with fronthand: 420 (70%)
  Windows with backhand: 348 (30%)  [some skipped due to neutral majority]
  Total training samples: 768

Shape of training data:
  X_train: (768, 30, 132)
           │    │   └─ 132 features (pose landmarks)
           │    └───── 30 frames per window
           └────────── 768 training samples

  y_train: (768,)
           └─ 768 labels (one per window)
              Values: [0=backhand, 1=fronthand]
```

---

## Part 2: Training Architecture

### 2.1 LSTM Network Architecture

```
INPUT: Sequence of pose data
  Shape: (batch_size, 30, 132)
         │           │   └─ 132 pose features
         │           └───── 30 time steps (frames)
         └───────────────── Multiple samples in batch

    ↓ Feed into network ↓

┌─────────────────────────────────────────────────────────────┐
│                      LSTM LAYER 1                           │
│  Units: 128                                                 │
│  Input:  (batch, 30, 132)                                   │
│  Output: (batch, 30, 128)  ← Returns sequences             │
│                                                             │
│  [Processes temporal patterns in pose data]                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (batch, 30, 128)
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                     DROPOUT (0.3)                           │
│  Randomly drops 30% of connections                          │
│  [Prevents overfitting]                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (batch, 30, 128)
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                      LSTM LAYER 2                           │
│  Units: 64                                                  │
│  Input:  (batch, 30, 128)                                   │
│  Output: (batch, 30, 64)  ← Returns sequences              │
│                                                             │
│  [Learns higher-level temporal features]                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (batch, 30, 64)
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                     DROPOUT (0.3)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (batch, 30, 64)
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                      LSTM LAYER 3                           │
│  Units: 32                                                  │
│  Input:  (batch, 30, 64)                                    │
│  Output: (batch, 32)  ← Returns ONLY final state           │
│                                                             │
│  [Compresses temporal sequence into fixed representation]  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (batch, 32)
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                     DROPOUT (0.3)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (batch, 32)
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   DENSE LAYER (OUTPUT)                      │
│  Units: num_classes (e.g., 2 for backhand/fronthand)       │
│  Activation: Softmax                                        │
│  Input:  (batch, 32)                                        │
│  Output: (batch, 2)                                         │
│                                                             │
│  Output: [P(backhand), P(fronthand)]                       │
│          Example: [0.12, 0.88] → Predict fronthand         │
└─────────────────────────────────────────────────────────────┘

OUTPUT: Class probabilities
  Shape: (batch_size, num_classes)
  Values: Probability distribution (sums to 1.0)
```

### 2.2 LSTM Cell Internal Mechanics

Each LSTM layer contains cells that process sequences. Here's what happens inside:

```
LSTM Cell at time step t:
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Input at time t:                                         │
│    x_t = pose features for frame t                        │
│    h_{t-1} = hidden state from previous frame             │
│    c_{t-1} = cell state from previous frame               │
│                                                            │
│  ┌──────────────────────────────────────────────┐         │
│  │         FORGET GATE                          │         │
│  │  f_t = σ(W_f·[h_{t-1}, x_t] + b_f)          │         │
│  │  "What to forget from previous state?"       │         │
│  └──────────────────────────────────────────────┘         │
│                      ↓                                     │
│  ┌──────────────────────────────────────────────┐         │
│  │         INPUT GATE                           │         │
│  │  i_t = σ(W_i·[h_{t-1}, x_t] + b_i)          │         │
│  │  "What new information to store?"            │         │
│  │                                               │         │
│  │  c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)       │         │
│  │  "Candidate values to add"                   │         │
│  └──────────────────────────────────────────────┘         │
│                      ↓                                     │
│  ┌──────────────────────────────────────────────┐         │
│  │     UPDATE CELL STATE                        │         │
│  │  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t           │         │
│  │  "Combine old and new information"           │         │
│  └──────────────────────────────────────────────┘         │
│                      ↓                                     │
│  ┌──────────────────────────────────────────────┐         │
│  │         OUTPUT GATE                          │         │
│  │  o_t = σ(W_o·[h_{t-1}, x_t] + b_o)          │         │
│  │  "What to output?"                           │         │
│  │                                               │         │
│  │  h_t = o_t ⊙ tanh(c_t)                       │         │
│  │  "Final hidden state for this time step"     │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  Output:                                                   │
│    h_t → passed to next time step or next layer           │
│    c_t → passed to next time step (internal memory)       │
│                                                            │
└────────────────────────────────────────────────────────────┘

Legend:
  σ = sigmoid function (outputs 0-1)
  tanh = hyperbolic tangent (outputs -1 to 1)
  ⊙ = element-wise multiplication
  W = weight matrices (learned during training)
  b = bias vectors (learned during training)
```

**Example: Processing a fronthand stroke**

```
Time step analysis for a 30-frame fronthand window:

Frame 0 (backswing starts):
  x_0 = [right_wrist_x, right_wrist_y, ...] ← Low position
  h_0 = [initialized to zeros]
  LSTM learns: "Wrist moving backward and down"

Frame 5 (backswing continues):
  x_5 = [right_wrist is further back]
  h_5 = [contains memory of frames 0-4]
  LSTM learns: "Continuous backward motion → likely preparation"

Frame 10 (peak of backswing):
  x_10 = [wrist at maximum backward position]
  h_10 = [remembers entire backswing trajectory]
  LSTM learns: "Reached peak → forward swing coming"

Frame 15 (forward swing):
  x_15 = [wrist accelerating forward]
  h_15 = [knows: back→peak→now forward]
  LSTM learns: "This pattern matches fronthand"

Frame 20 (contact point):
  x_20 = [wrist extended forward, high position]
  h_20 = [complete stroke pattern stored]
  LSTM learns: "Confirm fronthand stroke"

Frame 25-29 (follow-through):
  x_25-29 = [wrist continues forward and up]
  h_25-29 = [full stroke sequence remembered]
  LSTM output: High confidence for fronthand

Final output from LSTM layer 3:
  h_29 = 32-dimensional vector encoding "fronthand stroke pattern"

Dense layer:
  Input:  h_29 (32 dims)
  Output: [0.08, 0.92] → 92% fronthand, 8% backhand
```

### 2.3 Training Process

```
TRAINING LOOP (simplified):

For each epoch (1 to 100):

  1. Shuffle training data
     X_train: (768, 30, 132)
     y_train: (768,) → one-hot encoded to (768, 2)

  2. For each batch (batch_size=32):

     ┌────────────────────────────────────────────────┐
     │ FORWARD PASS                                   │
     ├────────────────────────────────────────────────┤
     │                                                │
     │ Batch input: (32, 30, 132)                     │
     │      ↓                                         │
     │ LSTM layers process sequences                  │
     │      ↓                                         │
     │ Output predictions: (32, 2)                    │
     │      ↓                                         │
     │ Compare with true labels                       │
     │      ↓                                         │
     │ Calculate loss (cross-entropy)                 │
     │   loss = -Σ y_true * log(y_pred)              │
     │                                                │
     └────────────────────────────────────────────────┘
              ↓
     ┌────────────────────────────────────────────────┐
     │ BACKWARD PASS                                  │
     ├────────────────────────────────────────────────┤
     │                                                │
     │ Compute gradients (∂loss/∂weights)            │
     │      ↓                                         │
     │ Update weights using Adam optimizer            │
     │   W_new = W_old - lr * gradient               │
     │      ↓                                         │
     │ Network learns to predict strokes better      │
     │                                                │
     └────────────────────────────────────────────────┘

  3. After all batches, evaluate on validation set

     ┌────────────────────────────────────────────────┐
     │ Validation loss: 0.3456                        │
     │ Validation accuracy: 85.2%                     │
     │                                                │
     │ Compare to previous best:                      │
     │   If improved: Save model                      │
     │   If not improved for 10 epochs: STOP          │
     └────────────────────────────────────────────────┘

RESULT: Trained model that maps sequences → stroke labels
```

**What the model learns:**

```
Fronthand pattern (learned by LSTM):
  Early frames:  Wrist moves backward → Recognize preparation
  Middle frames: Wrist at back peak → Anticipate forward swing
  Late frames:   Wrist accelerates forward → Confirm fronthand

  Pattern: backward → peak → forward (with specific joint angles)

Backhand pattern (learned by LSTM):
  Early frames:  Wrist crosses body → Different preparation
  Middle frames: Shoulders rotate opposite direction
  Late frames:   Wrist extends across body

  Pattern: cross → rotate → extend (different from fronthand)

The LSTM learns these temporal patterns across all joints,
not just the wrist. All 33 landmarks contribute!
```

---

## Part 3: Inference Architecture

### 3.1 Sliding Window Detection (Different from Training!)

When analyzing a new video, we use a **different sliding approach**:

```
TRAINING vs INFERENCE Sliding Window:

TRAINING (stride=15, with overlap):
┌──────────┐
│ Window 1 │
└──────────┘
     ┌──────────┐
     │ Window 2 │  ← 50% overlap
     └──────────┘
          ┌──────────┐
          │ Window 3 │
          └──────────┘
Purpose: Create more training samples
Speed: Doesn't matter (offline)

INFERENCE (stride=5, high overlap):
┌──────────┐
│ Window 1 │
└──────────┘
 ┌──────────┐
 │ Window 2 │  ← 83% overlap!
 └──────────┘
  ┌──────────┐
  │ Window 3 │  ← Checking frequently for strokes
  └──────────┘
   ┌──────────┐
   │ Window 4 │
   └──────────┘
Purpose: Precise stroke timing detection
Speed: Slower, but acceptable for offline analysis
```

**Detailed inference sliding window:**

```
New video: 300 frames (10 seconds @ 30fps)

Parameters:
  window_size = 30
  stride = 5  ← Move only 5 frames each time

Processing:
┌──────────────────────────────────────────────────────────────┐
│ Window 1: Frames 0-29                                        │
│   Model predicts: [0.05, 0.02, 0.93] → neutral (93%)        │
│   Result: confidence < threshold (0.7), skip                 │
├──────────────────────────────────────────────────────────────┤
│ Window 2: Frames 5-34                                        │
│   Model predicts: [0.10, 0.05, 0.85] → neutral (85%)        │
│   Result: Skip (neutral)                                     │
├──────────────────────────────────────────────────────────────┤
│ ... (frames 10-39, 15-44, 20-49 all predict neutral)        │
├──────────────────────────────────────────────────────────────┤
│ Window 7: Frames 30-59                                       │
│   Model predicts: [0.88, 0.10, 0.02] → fronthand (88%)      │
│   Result: ✓ DETECTED (confidence > 0.7)                     │
│   Save: {class: 'fronthand', frame_start: 30,               │
│          frame_end: 59, confidence: 0.88}                    │
├──────────────────────────────────────────────────────────────┤
│ Window 8: Frames 35-64                                       │
│   Model predicts: [0.92, 0.06, 0.02] → fronthand (92%)      │
│   Result: ✓ DETECTED (overlaps with Window 7)               │
│   Save: {class: 'fronthand', frame_start: 35,               │
│          frame_end: 64, confidence: 0.92}                    │
├──────────────────────────────────────────────────────────────┤
│ Window 9: Frames 40-69                                       │
│   Model predicts: [0.85, 0.12, 0.03] → fronthand (85%)      │
│   Result: ✓ DETECTED (still same stroke)                    │
│   Save: {class: 'fronthand', frame_start: 40,               │
│          frame_end: 69, confidence: 0.85}                    │
├──────────────────────────────────────────────────────────────┤
│ Window 10: Frames 45-74                                      │
│   Model predicts: [0.15, 0.10, 0.75] → neutral (75%)        │
│   Result: Skip (neutral → stroke ended)                     │
└──────────────────────────────────────────────────────────────┘

Raw detections (before merging):
  Detection 1: fronthand, frames 30-59,  confidence 88%
  Detection 2: fronthand, frames 35-64,  confidence 92%
  Detection 3: fronthand, frames 40-69,  confidence 85%

These are all the SAME stroke detected multiple times!
```

### 3.2 Post-Processing: Merging Detections

After sliding window, we merge nearby detections of the same class:

```
MERGING ALGORITHM:

Input: List of detections (sorted by frame_start)
Parameter: merge_nearby_strokes = 15 frames

┌─────────────────────────────────────────────────────────────┐
│ Detection 1: fronthand, frames 30-59, conf=0.88            │
│ Detection 2: fronthand, frames 35-64, conf=0.92            │
│                                                             │
│ Check: Same class? ✓                                       │
│        Nearby? 35 - 59 = -24 (overlapping!) ✓              │
│                                                             │
│ Action: MERGE into one detection                           │
│   frame_start = min(30, 35) = 30                           │
│   frame_end = max(59, 64) = 64                             │
│   confidences = [0.88, 0.92]                               │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Merged 1: fronthand, frames 30-64, confs=[0.88, 0.92]     │
│ Detection 3: fronthand, frames 40-69, conf=0.85            │
│                                                             │
│ Check: Same class? ✓                                       │
│        Nearby? 40 - 64 = -24 (overlapping!) ✓              │
│                                                             │
│ Action: MERGE again                                         │
│   frame_start = min(30, 40) = 30                           │
│   frame_end = max(64, 69) = 69                             │
│   confidences = [0.88, 0.92, 0.85]                         │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ FINAL STROKE:                                               │
│   class: fronthand                                          │
│   frames: 30-69 (39 frames ≈ 1.3 seconds)                  │
│   avg_confidence: (0.88 + 0.92 + 0.85) / 3 = 88.3%         │
│   time: 1.0s - 2.3s (frames ÷ fps)                         │
└─────────────────────────────────────────────────────────────┘
```

**Visual merging example:**

```
Before merging (raw detections):
Time:  0s    1s    2s    3s    4s    5s    6s    7s    8s
       │     │     │     │     │     │     │     │     │
       ├─────┤  Fronthand detection 1 (conf=88%)
         ├─────┤  Fronthand detection 2 (conf=92%)
           ├─────┤  Fronthand detection 3 (conf=85%)
                   ├─────┤  Backhand detection 1 (conf=91%)
                     ├─────┤  Backhand detection 2 (conf=87%)
                                ├─────┤  Fronthand detection 4 (conf=90%)

After merging:
Time:  0s    1s    2s    3s    4s    5s    6s    7s    8s
       │     │     │     │     │     │     │     │     │
       ├───────────┤  Fronthand (1.0s-2.3s, avg_conf=88%)
                   ├───────┤  Backhand (2.3s-3.8s, avg_conf=89%)
                                ├─────┤  Fronthand (5.0s-6.0s, conf=90%)

Result: 3 clean strokes instead of 6 overlapping detections
```

### 3.3 Filtering and Quality Control

```
FILTERING PIPELINE:

Raw predictions from model
         ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 1: Confidence Threshold                              │
│                                                            │
│ Keep only predictions with confidence ≥ 0.7               │
│                                                            │
│ Before: 234 predictions                                   │
│ After:  156 predictions (dropped 78 low-confidence)       │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 2: Merge Nearby Detections                           │
│                                                            │
│ Combine overlapping detections of same class              │
│                                                            │
│ Before: 156 detections                                    │
│ After:  24 merged strokes                                 │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 3: Minimum Duration Filter                           │
│                                                            │
│ Keep only strokes ≥ min_stroke_duration (13 frames)       │
│                                                            │
│ Before: 24 strokes                                        │
│ After:  18 strokes (dropped 6 very brief detections)      │
└────────────────────────────────────────────────────────────┘
         ↓
Final output: 18 high-quality stroke detections
```

**Example filtering:**

```
Stroke candidate 1:
  Duration: 39 frames (1.3 seconds)
  Confidence: 88%
  Check confidence: 88% ≥ 70% ✓
  Check duration: 39 ≥ 13 ✓
  Result: ✓ KEEP

Stroke candidate 2:
  Duration: 8 frames (0.27 seconds)
  Confidence: 92%
  Check confidence: 92% ≥ 70% ✓
  Check duration: 8 < 13 ✗
  Result: ✗ REJECT (too brief, likely false positive)

Stroke candidate 3:
  Duration: 25 frames (0.83 seconds)
  Confidence: 65%
  Check confidence: 65% < 70% ✗
  Result: ✗ REJECT (low confidence)
```

### 3.4 Complete Inference Pipeline

```
INPUT: New video (my_match.mp4)
    ↓
┌───────────────────────────────────────────────────────────┐
│ 1. POSE EXTRACTION                                        │
│    Extract landmarks from all 9000 frames                 │
│    Output: (9000, 132) array                              │
└───────────────┬───────────────────────────────────────────┘
                ↓
┌───────────────────────────────────────────────────────────┐
│ 2. SLIDING WINDOW (stride=5)                              │
│    Create windows: 0-29, 5-34, 10-39, ...                │
│    Number of windows: (9000-30)/5 = 1794 windows          │
└───────────────┬───────────────────────────────────────────┘
                ↓
┌───────────────────────────────────────────────────────────┐
│ 3. MODEL PREDICTION                                       │
│    For each window, predict stroke class                  │
│    Output: 1794 predictions with confidences              │
└───────────────┬───────────────────────────────────────────┘
                ↓
┌───────────────────────────────────────────────────────────┐
│ 4. CONFIDENCE FILTERING                                   │
│    Keep predictions with confidence ≥ 0.7                 │
│    1794 → 234 high-confidence predictions                 │
└───────────────┬───────────────────────────────────────────┘
                ↓
┌───────────────────────────────────────────────────────────┐
│ 5. MERGING                                                │
│    Combine overlapping detections of same class           │
│    234 detections → 24 merged strokes                     │
└───────────────┬───────────────────────────────────────────┘
                ↓
┌───────────────────────────────────────────────────────────┐
│ 6. DURATION FILTERING                                     │
│    Remove strokes shorter than 13 frames                  │
│    24 strokes → 18 final strokes                          │
└───────────────┬───────────────────────────────────────────┘
                ↓
┌───────────────────────────────────────────────────────────┐
│ 7. OUTPUT GENERATION                                      │
│    - Timeline visualization                               │
│    - Text report with timestamps                          │
│    - JSON with stroke data                                │
└───────────────────────────────────────────────────────────┘
    ↓
OUTPUT: Analysis results
```

---

## Mathematical Foundations

### Cross-Entropy Loss

The model is trained using categorical cross-entropy loss:

```
For a single sample:

True label: [0, 1]  ← One-hot encoded (class 1 = fronthand)
Prediction: [0.12, 0.88]  ← Model output probabilities

Loss = -Σ(y_true * log(y_pred))
     = -(0 * log(0.12) + 1 * log(0.88))
     = -log(0.88)
     = 0.128

Interpretation:
  Lower loss = better prediction
  If model predicted [0.01, 0.99], loss would be -log(0.99) = 0.010 (better!)
  If model predicted [0.50, 0.50], loss would be -log(0.50) = 0.693 (worse!)

For a batch of 32 samples:
  Total loss = average of individual losses
  This is what the optimizer tries to minimize
```

### Softmax Activation

The final layer uses softmax to convert logits to probabilities:

```
Raw output from dense layer (logits):
  z = [2.1, 3.8]  ← Arbitrary real numbers

Softmax:
  e^z₀ = e^2.1 = 8.17
  e^z₁ = e^3.8 = 44.70

  Sum = 8.17 + 44.70 = 52.87

  P(class 0) = 8.17 / 52.87 = 0.15  (15% backhand)
  P(class 1) = 44.70 / 52.87 = 0.85  (85% fronthand)

Output: [0.15, 0.85]

Properties:
  - All probabilities sum to 1.0
  - Each probability is between 0 and 1
  - Larger logit → higher probability
```

### Gradient Descent (Adam Optimizer)

```
Weight update at each training step:

Current weight: W = 0.5
Gradient: ∂Loss/∂W = -0.3  ← Computed via backpropagation
Learning rate: lr = 0.001

Simple gradient descent:
  W_new = W - lr * gradient
        = 0.5 - 0.001 * (-0.3)
        = 0.5 + 0.0003
        = 0.5003

Adam optimizer (used in our model):
  Keeps track of:
    m_t = moving average of gradients
    v_t = moving average of squared gradients

  Update is adaptive (larger updates when confident, smaller when uncertain)
  This leads to faster, more stable training
```

---

## Why This Architecture Works

### 1. Pose-Based Features vs Raw Pixels

```
Why pose landmarks work better than raw video:

RAW PIXELS:
  Frame size: 1920×1080×3 = 6,220,800 values per frame
  30-frame window: 186,624,000 values
  Problems:
    - Massive input size
    - Background clutter (court, crowd, etc.)
    - Lighting variations
    - Camera angle differences
    - Requires enormous dataset

POSE LANDMARKS:
  Features: 33 × 4 = 132 values per frame
  30-frame window: 3,960 values  ← 47,000× smaller!
  Benefits:
    - Focused on relevant body motion
    - Invariant to background
    - Robust to lighting changes
    - Works across camera angles
    - Needs less training data
```

### 2. Why Window Size = 30 Frames

```
Tennis stroke timing (approximate):

Serve:           0.8 - 1.5 seconds
Forehand:        0.5 - 1.2 seconds
Backhand:        0.5 - 1.2 seconds
Slice:           0.6 - 1.3 seconds

At 30 fps:
  0.5s = 15 frames  ← Too short for backswing
  1.0s = 30 frames  ← Captures most strokes ✓
  1.5s = 45 frames  ← Better for slow strokes, but larger model

Window = 30 frames (1 second) is a good balance:
  - Captures full stroke motion (backswing → contact → follow-through)
  - Small enough to fit in memory
  - Large enough for meaningful temporal patterns
  - Matches typical stroke duration
```

### 3. Why LSTM vs Other Architectures

```
LSTM advantages for stroke recognition:

Temporal patterns in tennis strokes:
  Frame 0-10:   Preparation (backswing)
  Frame 10-15:  Acceleration
  Frame 15-20:  Contact point
  Frame 20-30:  Follow-through

LSTM excels at:
  ✓ Remembering early frames while processing later ones
  ✓ Learning which temporal patterns matter
  ✓ Handling variable-speed strokes
  ✓ Capturing long-range dependencies

Alternatives:
  CNN: Good for spatial patterns, not temporal sequences
  Simple RNN: Struggles with long sequences (vanishing gradient)
  Transformer: Needs more data, computationally expensive

LSTM is the sweet spot for this task!
```

### 4. Why Majority Voting Works

```
Stroke boundaries are fuzzy:

Human annotator marks:
  Start: 1.0s (frame 30)
  End:   2.5s (frame 75)

Actual stroke motion:
  Preparation starts: 0.8s (frame 24)
  Contact:            1.7s (frame 51)
  Follow-through ends: 2.7s (frame 81)

Window at frames 25-54:
  Annotated as fronthand: frames 30-54 (25 frames)
  Not annotated:          frames 25-29 (5 frames)

  Majority: 25/30 = 83% fronthand ✓

Even with imperfect annotation boundaries,
majority voting captures the stroke correctly!

This makes the system robust to:
  - Annotation imprecision (±5 frames)
  - Slight timing variations
  - Gradual transitions between strokes
```

### 5. Why Skip Neutral Class

```
CLASS IMBALANCE PROBLEM:

With neutral class:
┌──────────────────────────────────────────────────────┐
│ Training data:                                       │
│   Neutral:    9012 samples (92%)  ← Overwhelming    │
│   Fronthand:   421 samples (4%)                      │
│   Backhand:    347 samples (4%)                      │
│                                                      │
│ Model learns:                                        │
│   "Just predict neutral every time"                  │
│   → 92% accuracy but 0% for actual strokes!          │
└──────────────────────────────────────────────────────┘

Without neutral class:
┌──────────────────────────────────────────────────────┐
│ Training data:                                       │
│   Fronthand:   420 samples (55%)  ← Balanced        │
│   Backhand:    348 samples (45%)                     │
│                                                      │
│ Model learns:                                        │
│   Actual stroke patterns                             │
│   → 84% accuracy on real strokes ✓                   │
└──────────────────────────────────────────────────────┘

At inference:
  - Model predicts stroke class for every window
  - Low confidence → likely not a stroke (implicit neutral)
  - High confidence → stroke detected!

This two-stage approach works better than trying to
learn "neutral" as an explicit class.
```

### 6. Why High Overlap in Inference (stride=5)

```
STROKE BOUNDARY DETECTION:

True stroke: frames 30-65 (35 frames)

With stride=15 (low overlap):
  Window 1: frames 0-29   → Prediction: neutral
  Window 2: frames 15-44  → Prediction: fronthand (partially overlaps)
  Window 3: frames 30-59  → Prediction: fronthand (good coverage)
  Window 4: frames 45-74  → Prediction: fronthand (partially overlaps)
  Window 5: frames 60-89  → Prediction: neutral

  Problem: Only 3 detections, timing is imprecise

With stride=5 (high overlap):
  Window 1: frames 0-29   → neutral
  Window 2: frames 5-34   → neutral
  Window 3: frames 10-39  → neutral
  Window 4: frames 15-44  → neutral
  Window 5: frames 20-49  → fronthand ← First detection!
  Window 6: frames 25-54  → fronthand
  Window 7: frames 30-59  → fronthand
  Window 8: frames 35-64  → fronthand
  Window 9: frames 40-69  → fronthand ← Last detection
  Window 10: frames 45-74 → neutral

  Result: 5 detections, merged to precise timing (20-69)

Benefit:
  - More precise start/end times
  - More confident (multiple confirmations)
  - Robust to edge cases

Cost:
  - 3× more computation (acceptable for offline analysis)
```

---

## Summary

### Data Flow Recap

```
1. VIDEO + ANNOTATIONS
   ↓ MediaPipe extracts pose
2. POSE SEQUENCES (N frames × 132 features)
   ↓ Sliding window (30 frames, 50% overlap)
3. TRAINING WINDOWS (768 samples × 30 frames × 132 features)
   ↓ Skip windows with neutral majority
4. FILTERED DATASET (only stroke windows)
   ↓ Train LSTM network
5. TRAINED MODEL (weights learned)
   ↓ Apply to new video
6. PREDICTIONS (sliding window with stride=5)
   ↓ Merge and filter
7. STROKE DETECTIONS (timeline + statistics)
```

### Key Design Principles

1. **Pose-based features**: Focus on body motion, ignore irrelevant visual details
2. **Temporal modeling**: LSTM captures stroke patterns over time
3. **Sliding windows**: Convert variable-length videos to fixed-length inputs
4. **Majority voting**: Robust label assignment despite annotation imperfection
5. **Skip neutral**: Avoid class imbalance by training only on strokes
6. **Two-stage approach**: Train on clean data, detect with post-processing
7. **High overlap inference**: Precise timing detection with multiple confirmations

### Why It Works

- **MediaPipe**: Robust pose estimation handles various conditions
- **LSTM**: Perfect for sequential patterns in stroke motion
- **Window size (30)**: Matches typical stroke duration
- **Skipping neutral**: Prevents overwhelming model with non-strokes
- **Merging detections**: Combines fragmented predictions into clean strokes
- **Confidence threshold**: Natural way to distinguish strokes from non-strokes

This architecture balances **accuracy**, **robustness**, and **practicality** for tennis stroke recognition!
