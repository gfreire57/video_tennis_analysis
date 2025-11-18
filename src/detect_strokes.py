"""
Continuous Video Stroke Detection and Timeline Analysis
Analyzes a tennis video to detect strokes and generate timeline with statistics
"""

import os
# Suppress TensorFlow warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

from datetime import timedelta
from pathlib import Path
from tensorflow import keras
import argparse
import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import tensorflow as tf


# ============================================================================
# CONFIGURATION: default
# ============================================================================

CONFIG = {
    'model_path': r'.\output\tennis_stroke_model.keras',
    'label_classes_path': r'.\output\label_classes.npy',

    # TEMPORAL WINDOW CONFIGURATION (time-based, FPS-independent)
    'reference_fps': 30,  # Reference FPS for window calibration (must match training)
    'window_size': 45,  # Number of frames at reference_fps (45 frames @ 30fps = 1.5 seconds)
    'overlap': 15,  # Overlap at reference_fps (15 frames @ 30fps = 0.5 seconds)

    # PREDICTION ALIGNMENT (fixes "seeing the future" effect)
    # Options: 'start', 'center', 'end'
    # - 'start': Prediction assigned to start of window (earliest, may appear too early)
    # - 'center': Prediction centered on window (recommended, most accurate)
    # - 'end': Prediction assigned to end of window (latest, may appear too late)
    'prediction_alignment': 'center',

    'confidence_threshold': 0.6,  # Minimum confidence to consider a stroke
    'min_stroke_duration': 10,  # Minimum frames for a valid stroke (at reference_fps)
    'max_stroke_duration_seconds': 1.75,  # Maximum duration for a single stroke in seconds
    'merge_nearby_strokes': 15,  # Merge strokes within this many frames (at reference_fps)
    'visualize_video': True,  # Generate annotated video with pose and predictions
}


# ============================================================================
# POSE EXTRACTION
# ============================================================================

# FEATURE SELECTION: Use only biomechanically relevant landmarks
# MUST MATCH train_model.py selection for consistency!
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
        to match training feature selection
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
            return np.zeros(len(SELECTED_LANDMARKS) * 4)  # 60 zeros

    def extract_from_video(self, video_path, progress_callback=None):
        """Extract pose landmarks from entire video"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"FPS: {fps:.2f}, Total frames: {frame_count}")

        all_landmarks = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.extract_landmarks(frame)
            all_landmarks.append(landmarks)

            frame_idx += 1
            if frame_idx % 250 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
                if progress_callback:
                    progress_callback(frame_idx, frame_count)

        cap.release()

        return np.array(all_landmarks), fps

# ============================================================================
# STROKE DETECTION
# ============================================================================

def scale_params_for_fps(reference_fps, video_fps, window_size, overlap):
    """
    Scale temporal window parameters based on video FPS

    Args:
        reference_fps: Reference FPS used for calibration (e.g., 30)
        video_fps: Actual video FPS (e.g., 60, 48, 30)
        window_size: Window size at reference FPS
        overlap: Overlap at reference FPS

    Returns:
        Scaled window_size and overlap for the video's actual FPS
    """
    scale_factor = video_fps / reference_fps

    scaled_window = int(round(window_size * scale_factor))
    scaled_overlap = int(round(overlap * scale_factor))

    # Calculate temporal durations for verification
    window_duration = window_size / reference_fps
    scaled_window_duration = scaled_window / video_fps

    print(f"\n📐 FPS Scaling:")
    print(f"   Reference: {reference_fps} FPS → Video: {video_fps} FPS (scale factor: {scale_factor:.2f}x)")
    print(f"   Window: {window_size} → {scaled_window} frames ({window_duration:.2f}s → {scaled_window_duration:.2f}s)")
    print(f"   Overlap: {overlap} → {scaled_overlap} frames")

    return scaled_window, scaled_overlap

def detect_strokes_in_video(video_path, model, label_classes, config=CONFIG):
    """
    Detect strokes in a continuous video using sliding window

    Returns:
        strokes: List of detected strokes with timing and confidence
        fps: Video frame rate
    """

    # Extract pose landmarks
    pose_extractor = PoseExtractor()
    landmarks, fps = pose_extractor.extract_from_video(video_path)

    num_frames = len(landmarks)
    window_size = config['window_size']
    overlap = config['overlap']

    # Scale parameters based on video FPS
    reference_fps = config['reference_fps']
    if fps != reference_fps:
        window_size, overlap = scale_params_for_fps(reference_fps, fps, window_size, overlap)
    else:
        print(f"\n📐 Using reference FPS parameters (no scaling needed)")
        print(f"   Window: {window_size} frames, Overlap: {overlap} frames")

    stride = window_size - overlap  # Calculate stride from overlap

    print(f"\nAnalyzing video with sliding window...")
    print(f"Window size: {window_size} frames, Overlap: {overlap} frames, Stride: {stride} frames")
    print(f"Prediction alignment: '{config.get('prediction_alignment', 'center')}' (to reduce 'seeing the future' effect)")

    # Sliding window predictions
    predictions = []

    for i in range(0, num_frames - window_size, stride):
        window = landmarks[i:i+window_size]
        window = np.expand_dims(window, axis=0)  # Add batch dimension

        # Predict
        pred = model.predict(window, verbose=0)
        pred_class = np.argmax(pred[0])
        confidence = pred[0][pred_class]

        # Align prediction based on configuration
        # This addresses "seeing the future" effect where predictions appear too early
        alignment = config.get('prediction_alignment', 'center')

        if alignment == 'start':
            # Original behavior: assign to window start
            pred_start = i
            pred_end = i + window_size
        elif alignment == 'center':
            # Center prediction on window (reduces early bias by ~0.75s)
            window_center = i + window_size // 2
            half_window = window_size // 4  # Use quarter window on each side
            pred_start = window_center - half_window
            pred_end = window_center + half_window
        elif alignment == 'end':
            # Assign to window end (most conservative)
            pred_start = i + window_size // 2
            pred_end = i + window_size
        else:
            # Default to center if invalid option
            window_center = i + window_size // 2
            half_window = window_size // 4
            pred_start = window_center - half_window
            pred_end = window_center + half_window

        predictions.append({
            'frame_start': pred_start,
            'frame_end': pred_end,
            'class_idx': pred_class,
            'class_name': label_classes[pred_class],
            'confidence': float(confidence)
        })

        if (i // stride) % 100 == 0:
            print(f"Analyzed {i}/{num_frames} frames ({i/num_frames*100:.1f}%)")

    # Filter by confidence
    high_conf_predictions = [
        p for p in predictions
        if p['confidence'] >= config['confidence_threshold']
    ]

    print(f"\nFound {len(high_conf_predictions)} high-confidence predictions")

    # Merge nearby detections of the same stroke
    strokes = merge_stroke_detections(high_conf_predictions, config, fps)

    # Add timing information
    for stroke in strokes:
        stroke['time_start'] = stroke['frame_start'] / fps
        stroke['time_end'] = stroke['frame_end'] / fps
        stroke['duration'] = stroke['time_end'] - stroke['time_start']

    print(f"After merging: {len(strokes)} strokes detected")
    if strokes:
        durations = [f"{s['duration']:.2f}s" for s in strokes[:5]]
        print(f"Stroke durations (first 5): {durations}")

    return strokes, fps


def merge_stroke_detections(predictions, config, fps):
    """
    Merge nearby detections of the same stroke type

    Modified to respect max_stroke_duration_seconds to prevent
    unrealistically long stroke detections.
    """
    if not predictions:
        return []

    # Sort by frame start
    predictions = sorted(predictions, key=lambda x: x['frame_start'])

    merged_strokes = []
    current_stroke = None

    for pred in predictions:
        if current_stroke is None:
            current_stroke = {
                'class_name': pred['class_name'],
                'frame_start': pred['frame_start'],
                'frame_end': pred['frame_end'],
                'confidences': [pred['confidence']]
            }
        elif (pred['class_name'] == current_stroke['class_name'] and
              pred['frame_start'] - current_stroke['frame_end'] <= config['merge_nearby_strokes']):

            # Check if merging would create unrealistic duration
            potential_end = max(current_stroke['frame_end'], pred['frame_end'])
            potential_duration_frames = potential_end - current_stroke['frame_start']
            potential_duration_seconds = potential_duration_frames / fps

            if potential_duration_seconds <= config['max_stroke_duration_seconds']:
                # Safe to merge - within realistic duration
                current_stroke['frame_end'] = potential_end
                current_stroke['confidences'].append(pred['confidence'])
            else:
                # Merging would exceed max duration - save current and start new
                duration = current_stroke['frame_end'] - current_stroke['frame_start']
                if duration >= config['min_stroke_duration']:
                    current_stroke['avg_confidence'] = np.mean(current_stroke['confidences'])
                    merged_strokes.append(current_stroke)

                # Start new stroke with this prediction
                current_stroke = {
                    'class_name': pred['class_name'],
                    'frame_start': pred['frame_start'],
                    'frame_end': pred['frame_end'],
                    'confidences': [pred['confidence']]
                }
        else:
            # Different class or too far apart - save current stroke
            duration = current_stroke['frame_end'] - current_stroke['frame_start']
            if duration >= config['min_stroke_duration']:
                current_stroke['avg_confidence'] = np.mean(current_stroke['confidences'])
                merged_strokes.append(current_stroke)

            # Start new stroke
            current_stroke = {
                'class_name': pred['class_name'],
                'frame_start': pred['frame_start'],
                'frame_end': pred['frame_end'],
                'confidences': [pred['confidence']]
            }

    # Don't forget the last stroke
    if current_stroke and (current_stroke['frame_end'] - current_stroke['frame_start']) >= config['min_stroke_duration']:
        current_stroke['avg_confidence'] = np.mean(current_stroke['confidences'])
        merged_strokes.append(current_stroke)

    return merged_strokes

# ============================================================================
# TIMELINE VISUALIZATION
# ============================================================================

def create_timeline_visualization(strokes, fps, video_duration, output_path):
    """Create a visual timeline of detected strokes"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Color map for different stroke types
    stroke_colors = {
        'backhand': '#FF6B6B',
        'fronthand': '#4ECDC4',
        'forehand': '#4ECDC4',  # Alternative spelling
        'saque': '#FFE66D',
        'serve': '#FFE66D',  # Alternative spelling
        'slice direita': '#95E1D3',
        'slice esquerda': '#A8E6CF',
    }

    # Timeline plot
    ax1.set_xlim(0, video_duration)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_title('Stroke Detection Timeline', fontsize=14, fontweight='bold')
    ax1.set_yticks([])

    for stroke in strokes:
        color = stroke_colors.get(stroke['class_name'].lower(), '#95A5A6')
        rect = patches.Rectangle(
            (stroke['time_start'], 0.1),
            stroke['duration'],
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax1.add_patch(rect)

        # Add label if stroke is long enough
        if stroke['duration'] > 1.0:
            ax1.text(
                stroke['time_start'] + stroke['duration']/2,
                0.5,
                stroke['class_name'],
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold'
            )

    # Add grid
    ax1.grid(True, axis='x', alpha=0.3)

    # Statistics plot
    stroke_counts = {}
    for stroke in strokes:
        stroke_counts[stroke['class_name']] = stroke_counts.get(stroke['class_name'], 0) + 1

    if stroke_counts:
        labels = list(stroke_counts.keys())
        counts = list(stroke_counts.values())
        colors = [stroke_colors.get(label.lower(), '#95A5A6') for label in labels]

        ax2.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Stroke Frequency', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)

        # Add count labels on bars
        for i, (label, count) in enumerate(zip(labels, counts)):
            ax2.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nTimeline saved to: {output_path}")

    return fig

# ============================================================================
# VIDEO VISUALIZATION
# ============================================================================

def create_annotated_video(video_path, strokes, model, label_classes, output_path, config=CONFIG):
    """
    Create annotated video showing pose estimation and model predictions

    Args:
        video_path: Path to original video
        strokes: List of detected strokes
        model: Trained model
        label_classes: Class labels
        output_path: Path to save annotated video
        config: Configuration dict
    """
    print(f"\nGenerating annotated video...")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create frame-to-stroke mapping
    frame_predictions = {}
    for stroke in strokes:
        for frame_idx in range(stroke['frame_start'], stroke['frame_end'] + 1):
            frame_predictions[frame_idx] = {
                'class_name': stroke['class_name'],
                'confidence': stroke['avg_confidence']
            }

    # Color map for stroke types
    stroke_colors = {
        'backhand': (107, 107, 255),      # Red (BGR)
        'fronthand': (196, 205, 78),       # Cyan (BGR)
        'forehand': (196, 205, 78),
        'saque': (109, 230, 255),          # Yellow (BGR)
        'serve': (109, 230, 255),
    }

    frame_idx = 0
    all_landmarks = []

    print("Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy for annotation
        annotated_frame = frame.copy()

        # Extract pose
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Extract landmarks for prediction
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            all_landmarks.append(np.array(landmarks))

            # Get bounding box around pose
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in results.pose_landmarks.landmark]
            y_coords = [lm.y * h for lm in results.pose_landmarks.landmark]

            x_min = max(0, int(min(x_coords)) - 20)
            x_max = min(w, int(max(x_coords)) + 20)
            y_min = max(0, int(min(y_coords)) - 20)
            y_max = min(h, int(max(y_coords)) + 20)
        else:
            all_landmarks.append(np.zeros(33 * 4))
            x_min, y_min, x_max, y_max = 0, 0, width, height

        # Check if this frame has a prediction
        if frame_idx in frame_predictions:
            pred = frame_predictions[frame_idx]
            class_name = pred['class_name']
            confidence = pred['confidence']
            color = stroke_colors.get(class_name.lower(), (149, 165, 149))

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 3)

            # Draw label background
            label = f"{class_name.upper()} ({confidence:.0%})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )

            label_y = y_min - 10 if y_min > 30 else y_max + 30
            cv2.rectangle(
                annotated_frame,
                (x_min, label_y - text_height - 10),
                (x_min + text_width + 10, label_y + baseline),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x_min + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Add frame counter
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Add timestamp
        time_sec = frame_idx / fps
        cv2.putText(
            annotated_frame,
            f"Time: {time_sec:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Write frame
        out.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    cap.release()
    out.release()
    pose.close()

    print(f"✅ Annotated video saved to: {output_path}")
    print(f"   Total frames: {frame_idx}")
    print(f"   Duration: {frame_idx/fps:.2f}s")

    return output_path

# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def generate_stroke_report(strokes, fps, video_duration, output_path):
    """Generate detailed stroke analysis report"""

    report = []
    report.append("=" * 70)
    report.append("TENNIS STROKE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Video info
    report.append(f"Video Duration: {timedelta(seconds=int(video_duration))}")
    report.append(f"Total Strokes Detected: {len(strokes)}")
    report.append("")

    # Stroke frequency
    stroke_counts = {}
    for stroke in strokes:
        stroke_counts[stroke['class_name']] = stroke_counts.get(stroke['class_name'], 0) + 1

    report.append("Stroke Frequency:")
    for stroke_type, count in sorted(stroke_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(strokes) * 100) if strokes else 0
        report.append(f"  {stroke_type}: {count} ({percentage:.1f}%)")
    report.append("")

    # Detailed stroke list
    report.append("Detailed Stroke Timeline:")
    report.append("-" * 70)

    for i, stroke in enumerate(strokes, 1):
        start_time = timedelta(seconds=int(stroke['time_start']))
        end_time = timedelta(seconds=int(stroke['time_end']))
        report.append(
            f"{i:3d}. {stroke['class_name']:15s} | "
            f"{start_time} - {end_time} | "
            f"Duration: {stroke['duration']:.2f}s | "
            f"Confidence: {stroke['avg_confidence']:.2%}"
        )

    report.append("")
    report.append("=" * 70)

    # Save report
    report_text = '\n'.join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {output_path}")

    return report_text

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_video(video_path, output_dir='./analysis_output'):
    """
    Complete video analysis pipeline

    Args:
        video_path: Path to video file
        output_dir: Directory to save results
    """

    print("=" * 70)
    print("TENNIS VIDEO STROKE ANALYSIS")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem

    # Load model and label classes
    print("Loading trained model...")
    model = keras.models.load_model(CONFIG['model_path'])
    label_classes = np.load(CONFIG['label_classes_path'], allow_pickle=True)
    print(f"Model loaded. Classes: {', '.join(label_classes)}")
    print()

    # Detect strokes
    strokes, fps = detect_strokes_in_video(video_path, model, label_classes)

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    cap.release()

    print(f"\nDetected {len(strokes)} strokes in video")

    if len(strokes) == 0:
        print("⚠️  No strokes detected. Try:")
        print("   - Lowering confidence_threshold in CONFIG")
        print("   - Checking if video contains the trained stroke types")
        return

    # Generate timeline visualization
    timeline_path = output_dir / f"{video_name}_timeline.png"
    create_timeline_visualization(strokes, fps, video_duration, timeline_path)

    # Generate report
    report_path = output_dir / f"{video_name}_report.txt"
    generate_stroke_report(strokes, fps, video_duration, report_path)

    # Save JSON with stroke data
    json_path = output_dir / f"{video_name}_strokes.json"
    with open(json_path, 'w') as f:
        json.dump({
            'video_path': str(video_path),
            'video_duration': video_duration,
            'fps': fps,
            'strokes': strokes
        }, f, indent=2)
    print(f"Stroke data saved to: {json_path}")

    # Generate annotated video if enabled
    if CONFIG.get('visualize_video', True):
        annotated_video_path = output_dir / f"{video_name}_annotated.mp4"
        create_annotated_video(
            video_path=video_path,
            strokes=strokes,
            model=model,
            label_classes=label_classes,
            output_path=annotated_video_path
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)



# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Detect and analyze tennis strokes in video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--output-dir', default='./analysis_output',
                       help='Directory to save analysis results')
    parser.add_argument('--confidence', type=float,
                       help=f'Confidence threshold (0-1, default: {CONFIG["confidence_threshold"]})')
    parser.add_argument('--overlap', type=int,
                       help=f'Overlap between windows in frames (default: {CONFIG["overlap"]}, should match training)')
    parser.add_argument('--window-size', type=int,
                       help=f'Window size in frames (default: {CONFIG["window_size"]}, must match training)')
    parser.add_argument('--max-duration', type=float,
                       help=f'Maximum stroke duration in seconds (default: {CONFIG["max_stroke_duration_seconds"]})')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip annotated video generation if true')

    args = parser.parse_args()

    # Update CONFIG only with explicitly provided CLI arguments
    if args.confidence is not None:
        CONFIG['confidence_threshold'] = args.confidence
    if args.overlap is not None:
        CONFIG['overlap'] = args.overlap
    if args.window_size is not None:
        CONFIG['window_size'] = args.window_size
    if args.max_duration is not None:
        CONFIG['max_stroke_duration_seconds'] = args.max_duration
    if args.no_visualize:
        CONFIG['visualize_video'] = False

    # Run analysis
    analyze_video(args.video_path, args.output_dir)
