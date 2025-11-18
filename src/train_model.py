"""
Tennis Stroke Recognition using Pose Estimation + LSTM
Processes Label Studio annotations (FRAME-BASED) and trains a model to classify tennis strokes
"""

import os
# Suppress TensorFlow warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'video_base_path': r'D:\Mestrado\redes_neurais\dados_filtrados\videos',  # Base path for videos
    'label_studio_exports': r'D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\label_studio_exports',  # Folder with JSON files
    'output_dir': r'D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\output',

    # TEMPORAL WINDOW CONFIGURATION (time-based, FPS-independent)
    'reference_fps': 30,  # Reference FPS for window calibration
    'window_size': 45,  # Number of frames per sequence at reference_fps (45 frames @ 30fps = 1.5 seconds)
    'overlap': 15,  # Overlap between windows at reference_fps (15 frames @ 30fps = 0.5 seconds)
    'MIN_ANNOTATION_LENGTH': 15,  # Minimum annotation length at reference_fps (15 frames @ 30fps = 0.5 seconds)

    'confidence_threshold': 0.5,  # MediaPipe confidence threshold
    'use_mixed_precision': False,  # Enable mixed precision training for faster GPU training (GPU only)
    'use_mlflow': True,  # Enable MLflow experiment tracking
    'mlflow_experiment_name': 'tennis_stroke_recognition',  # MLflow experiment name
    'group_classes': True,  # Group slice classes with main strokes, ignore saque
    'learning_rate': 0.001,  # Learning rate for Adam optimizer (reduced for stability)
    'batch_size': 32,  # Batch size for training
    'epochs': 150,  # Maximum number of epochs (increased for better convergence)
}

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def setup_gpu():
    """Configure GPU settings for TensorFlow"""
    print("=" * 70)
    print("GPU CONFIGURATION")
    print("=" * 70)

    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"\n✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")

        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n✅ GPU memory growth enabled (dynamic allocation)")

            # Optional: Set memory limit (e.g., use only 4GB)
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
            # )

            print(f"✅ TensorFlow will use GPU for training")

        except RuntimeError as e:
            print(f"\n⚠️  GPU configuration error: {e}")
    else:
        print("\n⚠️  No GPU found. Training will use CPU (slower)")
        print("   To use GPU, install CUDA and cuDNN, then reinstall tensorflow with GPU support")

    # Print TensorFlow build info
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print("=" * 70 + "\n")


# ============================================================================
# STEP 1: EXTRACT POSES FROM VIDEO
# ============================================================================

# FEATURE SELECTION: Use only biomechanically relevant landmarks
# Reduces from 132 features (33 landmarks) to 60 features (15 landmarks)
SELECTED_LANDMARKS = [
    0,   # Nose (head position/orientation)
    11, 12,  # Left/Right shoulders (torso rotation)
    13, 14,  # Left/Right elbows (arm mechanics)
    15, 16,  # Left/Right wrists (racket trajectory)
    23, 24,  # Left/Right hips (body rotation, power)
    25, 26,  # Left/Right knees (weight transfer)
    27, 28,  # Left/Right ankles (stance)
    29, 30,  # Left/Right heels (foot position)
]

class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame):
        """
        Extract pose landmarks from a single frame

        Returns only SELECTED landmarks (60 features instead of 132)
        to focus on biomechanically relevant features for stroke classification
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = []
            # Extract only selected landmarks
            for idx in SELECTED_LANDMARKS:
                lm = results.pose_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(landmarks)  # Length: 60 (15 landmarks × 4 values)
        else:
            # Return zeros if no pose detected
            return np.zeros(len(SELECTED_LANDMARKS) * 4)  # 60 zeros
    
    def extract_from_video(self, video_path):
        """Extract pose landmarks from entire video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {frame_count}")
        
        all_landmarks = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_landmarks(frame)
            all_landmarks.append(landmarks)
            
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        
        return np.array(all_landmarks), fps

# ============================================================================
# STEP 2: PARSE LABEL STUDIO ANNOTATIONS (FRAME-BASED FORMAT)
# ============================================================================

def parse_label_studio_json_frames(json_path):
    """
    Parse Label Studio JSON export with FRAME-BASED annotations

    Returns:
        video_path: Path to video file
        annotations: List of dicts with 'start_frame', 'end_frame', 'label'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both single annotation and list of annotations
    if isinstance(data, list):
        data = data[0]

    # Extract video path - handle both old and new Label Studio formats
    try:
        video_path = data['task']['data']['video']
    except KeyError:
        video_path = data['data']['video']

    # Clean up the video path - remove Label Studio prefix
    # Label Studio format: "/data/local-files/?d=videos/Bhand_1.MP4"
    # We want: "Bhand_1.MP4"
    video_path = video_path.replace('/data/local-files/?d=', '')
    # Remove the 'videos/' prefix if present, since video_base_path already points to the videos directory
    if video_path.startswith('videos/') or video_path.startswith('videos\\'):
        video_path = video_path[7:]  # Remove 'videos/' or 'videos\'

    annotations = []

    # Get annotations - handle both old and new Label Studio formats
    result_data = None
    if 'result' in data and isinstance(data['result'], list):
        # New format: annotations directly in 'result'
        result_data = data['result']
    elif 'annotations' in data and data['annotations'] and len(data['annotations']) > 0:
        # Old format: annotations in 'annotations[0]['result']'
        result_data = data['annotations'][0]['result']

    # Parse each labeled segment
    if result_data:
        for result in result_data:
            if result['type'] == 'timelinelabels':
                # Extract frame ranges and labels
                for range_item in result['value']['ranges']:
                    start_frame = range_item['start']
                    end_frame = range_item['end']
                    label = result['value']['timelinelabels'][0]  # Get first label

                    annotations.append({
                        'start_frame': int(start_frame),
                        'end_frame': int(end_frame),
                        'label': label.lower()  # Normalize to lowercase
                    })

    return video_path, annotations

# ============================================================================
# STEP 3: CREATE TRAINING SEQUENCES
# ============================================================================

def scale_params_for_fps(reference_fps, video_fps, window_size, overlap, min_annotation_length):
    """
    Scale temporal window parameters based on video FPS

    Args:
        reference_fps: Reference FPS used for calibration (e.g., 30)
        video_fps: Actual video FPS (e.g., 60, 48, 30)
        window_size: Window size at reference FPS
        overlap: Overlap at reference FPS
        min_annotation_length: Minimum annotation length at reference FPS

    Returns:
        Scaled parameters for the video's actual FPS
    """
    scale_factor = video_fps / reference_fps

    scaled_window = int(round(window_size * scale_factor))
    scaled_overlap = int(round(overlap * scale_factor))
    scaled_min_length = int(round(min_annotation_length * scale_factor))

    # Calculate temporal durations for verification
    window_duration = window_size / reference_fps
    scaled_window_duration = scaled_window / video_fps

    print(f"\n📐 FPS Scaling:")
    print(f"   Reference: {reference_fps} FPS → Video: {video_fps} FPS (scale factor: {scale_factor:.2f}x)")
    print(f"   Window: {window_size} → {scaled_window} frames ({window_duration:.2f}s → {scaled_window_duration:.2f}s)")
    print(f"   Overlap: {overlap} → {scaled_overlap} frames")
    print(f"   Min annotation: {min_annotation_length} → {scaled_min_length} frames")

    return scaled_window, scaled_overlap, scaled_min_length

def create_sequences_from_frames(landmarks, annotations, fps, window_size=30, overlap=15, group_classes=True, min_annotation_length=15):
    """
    Create fixed-length sequences from pose landmarks with labels
    Uses FRAME-BASED annotations directly

    Args:
        landmarks: Array of shape (num_frames, 132) - pose landmarks
        annotations: List of annotation dicts with 'start_frame', 'end_frame', 'label'
        fps: Video FPS (used to scale parameters from reference FPS)
        window_size: Number of frames per sequence (at reference FPS)
        overlap: Number of overlapping frames between windows (at reference FPS)
        group_classes: If True, groups slice classes and ignores saque
        min_annotation_length: Minimum annotation length (at reference FPS)
    """
    X, y = [], []
    num_frames = len(landmarks)

    # Scale parameters based on video FPS
    reference_fps = CONFIG['reference_fps']
    if fps != reference_fps:
        window_size, overlap, min_annotation_length = scale_params_for_fps(
            reference_fps, fps, window_size, overlap, min_annotation_length
        )
    else:
        print(f"\n📐 Using reference FPS parameters (no scaling needed)")
        print(f"   Window: {window_size} frames, Overlap: {overlap} frames")

    print(f"\nCreating sequences from {num_frames} frames")
    print(f"Window size: {window_size}, Overlap: {overlap}")

    # CLASS GROUPING MAPPING
    # Group slice direita with fronthand, slice esquerda with backhand, ignore saque
    class_mapping = {
        'fronthand': 'fronthand',
        'forehand': 'fronthand',
        'slice direita': 'fronthand',  # ← Absorb into fronthand
        'backhand': 'backhand',
        'slice esquerda': 'backhand',  # ← Absorb into backhand
        'saque': 'ignore',  # ← Ignore saque
        'serve': 'ignore',
        'neutral': 'neutral'
    }

    frame_labels = ['neutral'] * num_frames

    expanded_count = 0
    min_frames = min_annotation_length

    for anno in annotations:
        start_frame = anno['start_frame']
        end_frame = anno['end_frame']
        label = anno['label']

        if group_classes and label in class_mapping:
            mapped_label = class_mapping[label]
            if mapped_label == 'ignore':
                continue
            label = mapped_label

        annotation_length = end_frame - start_frame

        if annotation_length < min_frames:
            needed = min_frames - annotation_length
            before = needed // 2
            after = needed - before
            start_frame = max(0, start_frame - before)
            end_frame = min(num_frames, end_frame + after)
            expanded_count += 1

        end_frame = min(end_frame, num_frames)

        for i in range(start_frame, end_frame):
            frame_labels[i] = label

    if expanded_count > 0:
        print(f"Expanded {expanded_count} annotations shorter than {min_frames} frames")

    
    # Sliding window approach
    stride = window_size - overlap

    for i in range(0, num_frames - window_size, stride):
        window = landmarks[i:i+window_size]

        # Determine label for this window (majority vote)
        window_labels = frame_labels[i:i+window_size]

        # Count non-neutral labels and skip if no other label different than neutral was found
        stroke_labels = [lbl for lbl in window_labels if lbl != 'neutral']

        if len(stroke_labels) == 0:
            continue

        # Count labels excluding 'neutral' label
        label_counts = {}
        for lbl in window_labels:
            if lbl != 'neutral':
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
        print(f"Window {i}-{i+window_size}: label counts: {label_counts}")

        # Use the most common non-neutral label
        majority_label = max(label_counts, key=label_counts.get)
        majority_count = label_counts[majority_label]

        # Append the majority label (here, it is not expected to exists more than one non-neutral label)
        X.append(window)
        y.append(majority_label)

        # # Use the most common label
        # majority_label = max(label_counts, key=label_counts.get)
        # majority_count = label_counts[majority_label]

        # # Only use window if:
        # # 1. Label covers >50% of frames
        # # 2. Label is NOT 'neutral' (skip unannotated segments) 
        # if majority_count > window_size * 0.5 and majority_label != 'neutral':
        #     X.append(window)
        #     y.append(majority_label)

        # majority_label = max(label_counts, key=label_counts.get)
        # print(f"  → Majority label: {majority_label} ({label_counts[majority_label]} frames)")

        # # Only keep if stroke class is majority (>50%)
        # if majority_label != 'neutral':
        #     X.append(window)
        #     y.append(majority_label)

    
    print(f"Created {len(X)} sequences")
    
    return np.array(X), np.array(y)

# ============================================================================
# STEP 4: BUILD AND TRAIN LSTM MODEL
# ============================================================================

def build_model(input_shape, num_classes, learning_rate=0.001):
    """Build improved LSTM model with BatchNormalization for better discrimination"""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        #### V1 - ARQUITETURA ORIGINAL ####
        # # First LSTM layer with BatchNormalization
        # keras.layers.LSTM(128, return_sequences=True),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(0.4),

        # # Second LSTM layer with BatchNormalization
        # keras.layers.LSTM(96, return_sequences=True),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(0.4),

        # # Third LSTM layer with BatchNormalization
        # keras.layers.LSTM(64, return_sequences=False),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(0.3),

        # # Dense layers with BatchNormalization
        # keras.layers.Dense(64, activation='relu'),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(0.3),
        ####

        #### V2 - ARQUITETURA MAIS SIMPLES ####
        # Simpler architecture (2 LSTM layers instead of 3)
        keras.layers.LSTM(64, return_sequences=True),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.LSTM(32, return_sequences=False),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        ####

        keras.layers.Dense(32, activation='relu'),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_training_history(history, output_dir):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png")
    print(f"Training history saved to {output_dir}/training_history.png")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Setup MLflow
    if CONFIG['use_mlflow']:
        mlflow.set_experiment(CONFIG['mlflow_experiment_name'])
        mlflow.set_tracking_uri("file:./mlruns")  # Local tracking
        print(f"MLflow tracking enabled: experiment '{CONFIG['mlflow_experiment_name']}'")
        print(f"MLflow UI: Run 'mlflow ui' to view experiments\n")

    # Setup GPU
    setup_gpu()

    # Enable mixed precision training for faster GPU training
    if CONFIG['use_mixed_precision']:
        print("Enabling mixed precision training for faster GPU performance...")
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled (float16)\n")

    # Create output directory
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TENNIS STROKE RECOGNITION - POSE + LSTM")
    print("FRAME-BASED ANNOTATIONS")
    print("=" * 70)
    
    # Initialize pose extractor
    pose_extractor = PoseExtractor()
    
    # Process all Label Studio JSON files
    all_X, all_y = [], []
    
    json_files = list(Path(CONFIG['label_studio_exports']).glob('*.json'))
    print(f"\nFound {len(json_files)} annotation files")
    
    for json_file in json_files:
        print(f"\n{'='*70}")
        print(f"Processing: {json_file.name}")
        print(f"{'='*70}")
        
        # Parse annotations
        video_filename, annotations = parse_label_studio_json_frames(json_file)
        # Construct full path, ensuring proper path separator
        video_path = str(Path(CONFIG['video_base_path']) / video_filename)
        
        print(f"Video: {video_filename}")
        print(f"Annotations: {len(annotations)}")
        for anno in annotations:
            duration_frames = anno['end_frame'] - anno['start_frame']
            print(f"  - {anno['label']}: frames {anno['start_frame']}-{anno['end_frame']} "
                  f"({duration_frames} frames)")
        
        # Check if video exists
        if not Path(video_path).exists():
            print(f"WARNING: Video not found at {video_path}, skipping...")
            continue
        
        # Extract poses
        landmarks, fps = pose_extractor.extract_from_video(video_path)
        
        if landmarks is None:
            print(f"Skipping {video_filename} - could not process video")
            continue
        
        print(f"Extracted {len(landmarks)} pose sequences")

        # Create training sequences
        X, y = create_sequences_from_frames(
            landmarks,
            annotations,
            fps,
            window_size=CONFIG['window_size'],
            overlap=CONFIG['overlap'],
            group_classes=CONFIG['group_classes'],
            min_annotation_length=CONFIG['MIN_ANNOTATION_LENGTH']
        )

        # Skip videos with no sequences
        if len(X) == 0:
            print("⚠️  WARNING: No sequences created for this video (annotations too short or no valid windows)")
            print("   Skipping this video...")
            continue

        # Print label distribution for this video
        unique, counts = np.unique(y, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")

        all_X.append(X)
        all_y.append(y)
    
    # Combine all data
    if len(all_X) == 0:
        print("\nERROR: No data was processed. Check video paths and annotations.")
        return
    
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    print(f"\n{'='*70}")
    print(f"TOTAL DATASET")
    print(f"{'='*70}")
    print(f"Total sequences: {len(X_all)}")
    print(f"Sequence shape: {X_all.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)
    
    print(f"\nLabel mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == idx)
        print(f"  {label} -> {idx} ({count} samples)")
    
    # Check class balance
    min_samples = min([np.sum(y_encoded == i) for i in range(len(label_encoder.classes_))])
    if min_samples < 20:
        print(f"\n⚠️  WARNING: Some classes have very few samples (min: {min_samples})")
        print("   Consider collecting more data or adjusting window parameters")
    
    # Save label encoder
    np.save(f"{CONFIG['output_dir']}/label_classes.npy", label_encoder.classes_)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build model
    print(f"\n{'='*70}")
    print("BUILDING MODEL")
    print(f"{'='*70}")

    # Calculate input features based on selected landmarks
    num_features = len(SELECTED_LANDMARKS) * 4  # 15 landmarks × 4 values = 60 features
    input_shape = (CONFIG['window_size'], num_features)  # (window_size, 60)
    num_classes = len(label_encoder.classes_)

    print(f"Input shape: {input_shape}")
    print(f"  - Window size: {CONFIG['window_size']} frames")
    print(f"  - Features per frame: {num_features} (from {len(SELECTED_LANDMARKS)} landmarks)")
    print(f"  - Number of classes: {num_classes}")

    model = build_model(input_shape, num_classes, learning_rate=CONFIG['learning_rate'])
    model.summary()

    # Start MLflow run
    if CONFIG['use_mlflow']:
        mlflow.start_run(run_name=f"lstm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Log parameters
        mlflow.log_param("window_size", CONFIG['window_size'])
        mlflow.log_param("overlap", CONFIG['overlap'])
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("num_features", num_features)
        mlflow.log_param("num_landmarks", len(SELECTED_LANDMARKS))
        mlflow.log_param("total_sequences", len(X_all))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("input_shape", str(input_shape))
        mlflow.log_param("batch_size", CONFIG['batch_size'])
        mlflow.log_param("epochs", CONFIG['epochs'])
        mlflow.log_param("learning_rate", CONFIG['learning_rate'])
        mlflow.log_param("group_classes", CONFIG['group_classes'])

        # Log class distribution
        for idx, label in enumerate(label_encoder.classes_):
            count = np.sum(y_encoded == idx)
            mlflow.log_metric(f"class_{label}_count", count)

    # Calculate class weights to handle imbalance

    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights_array))

    print(f"\n{'='*70}")
    print("CLASS WEIGHTS (to balance training)")
    print(f"{'='*70}")
    for idx, weight in class_weights.items():
        class_name = label_encoder.classes_[idx]
        class_count = np.sum(y_train == idx)
        print(f"  {class_name}: {weight:.3f} (n={class_count})")
    print()

    # Train model
    print(f"\n{'='*70}")
    print("TRAINING MODEL")
    print(f"{'='*70}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased from 15
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,      # More patient (was 7)
            min_lr=0.00005,  # Higher minimum (was 0.00001)
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        class_weight=class_weights,  # Apply class weights
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Log metrics to MLflow
    if CONFIG['use_mlflow']:
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("final_train_loss", history.history['loss'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("epochs_trained", len(history.history['loss']))

    # Detailed predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)


    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_classes,
        target_names=label_encoder.classes_
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred_classes,
        labels=range(len(label_encoder.classes_)),
        zero_division=0
    )

    # Save model
    model.save(f"{CONFIG['output_dir']}/tennis_stroke_model.keras")
    print(f"\nModel saved to {CONFIG['output_dir']}/tennis_stroke_model.keras")

    # Plot training history
    plot_training_history(history, CONFIG['output_dir'])

    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix visualization saved to {CONFIG['output_dir']}/confusion_matrix.png")

    # Log metrics to MLflow
    if CONFIG['use_mlflow']:
        # Log per-class metrics
        for idx, class_name in enumerate(label_encoder.classes_):
            mlflow.log_metric(f"{class_name}_precision", float(precision[idx]))
            mlflow.log_metric(f"{class_name}_recall", float(recall[idx]))
            mlflow.log_metric(f"{class_name}_f1_score", float(f1[idx]))
            mlflow.log_metric(f"{class_name}_support", int(support[idx]))

        # Log average metrics (macro and weighted)
        mlflow.log_metric("macro_avg_precision", float(precision.mean()))
        mlflow.log_metric("macro_avg_recall", float(recall.mean()))
        mlflow.log_metric("macro_avg_f1_score", float(f1.mean()))

        # Weighted averages
        total_support = support.sum()
        weighted_precision = (precision * support).sum() / total_support
        weighted_recall = (recall * support).sum() / total_support
        weighted_f1 = (f1 * support).sum() / total_support

        mlflow.log_metric("weighted_avg_precision", float(weighted_precision))
        mlflow.log_metric("weighted_avg_recall", float(weighted_recall))
        mlflow.log_metric("weighted_avg_f1_score", float(weighted_f1))

        # Log model
        mlflow.tensorflow.log_model(model, "model")

        # Log artifacts (plots, label classes)
        mlflow.log_artifact(f"{CONFIG['output_dir']}/training_history.png")
        mlflow.log_artifact(f"{CONFIG['output_dir']}/confusion_matrix.png")
        mlflow.log_artifact(f"{CONFIG['output_dir']}/label_classes.npy")
        mlflow.log_artifact(f"{CONFIG['output_dir']}/tennis_stroke_model.keras")

        # Log confusion matrix as text
        cm_text = f"Confusion Matrix:\n{cm}"
        with open(f"{CONFIG['output_dir']}/confusion_matrix.txt", 'w') as f:
            f.write(cm_text)
        mlflow.log_artifact(f"{CONFIG['output_dir']}/confusion_matrix.txt")

        # Save classification report as text
        report_text = classification_report(
            y_test,
            y_pred_classes,
            target_names=label_encoder.classes_
        )
        with open(f"{CONFIG['output_dir']}/classification_report.txt", 'w') as f:
            f.write(report_text)
        mlflow.log_artifact(f"{CONFIG['output_dir']}/classification_report.txt")

        # End MLflow run
        mlflow.end_run()
        print("\n✅ MLflow run completed. View results with: mlflow ui")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nModel and artifacts saved to: {CONFIG['output_dir']}/")

    if CONFIG['use_mlflow']:
        print(f"\n📊 MLflow Tracking:")
        print(f"   - Run 'mlflow ui' in the project directory")
        print(f"   - Open http://localhost:5000 in your browser")
        print(f"   - Compare experiments, metrics, and models")

    print(f"\nNext steps:")
    print(f"1. Check training_history.png for overfitting")
    print(f"2. Test on new videos: python detect_strokes.py <video_path>")
    print(f"3. If accuracy is low, consider:")
    print(f"   - Adding more training videos")
    print(f"   - Adjusting window_size in CONFIG")
    print(f"   - Balancing classes (equal forehand/backhand samples)")

if __name__ == "__main__":
    main()