"""
Video Preprocessing Tool
Improve video quality for better pose detection:
- Brighten dark videos (night recordings)
- Crop/zoom to make player larger
- Auto-detect and track player
- Optional fisheye correction
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import mediapipe as mp

# MediaPipe for player detection
mp_pose = mp.solutions.pose

def auto_detect_player_bbox(frame, pose):
    """
    Automatically detect player's bounding box using MediaPipe
    Returns (x, y, width, height) or None if not detected
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w = frame.shape[:2]

        # Get all landmark coordinates
        x_coords = [lm.x * w for lm in results.pose_landmarks.landmark]
        y_coords = [lm.y * h for lm in results.pose_landmarks.landmark]

        # Get bounding box with padding
        x_min = max(0, int(min(x_coords)))
        x_max = min(w, int(max(x_coords)))
        y_min = max(0, int(min(y_coords)))
        y_max = min(h, int(max(y_coords)))

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    return None


def smooth_bboxes(bboxes, window_size=5):
    """
    Smooth bounding boxes over time to reduce jitter
    Uses moving average
    """
    if len(bboxes) < window_size:
        return bboxes

    smoothed = []
    for i in range(len(bboxes)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(bboxes), i + window_size // 2 + 1)

        window = bboxes[start_idx:end_idx]

        # Average coordinates
        avg_x = int(np.mean([b[0] for b in window]))
        avg_y = int(np.mean([b[1] for b in window]))
        avg_w = int(np.mean([b[2] for b in window]))
        avg_h = int(np.mean([b[3] for b in window]))

        smoothed.append((avg_x, avg_y, avg_w, avg_h))

    return smoothed


def add_padding_to_bbox(bbox, frame_shape, padding_percent=0.3):
    """
    Add padding around bounding box
    padding_percent: 0.3 = 30% extra space around player
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]

    # Calculate padding
    pad_w = int(w * padding_percent)
    pad_h = int(h * padding_percent)

    # Apply padding
    x_new = max(0, x - pad_w)
    y_new = max(0, y - pad_h)
    w_new = min(frame_w - x_new, w + 2 * pad_w)
    h_new = min(frame_h - y_new, h + 2 * pad_h)

    return (x_new, y_new, w_new, h_new)


def brighten_frame(frame, brightness_increase=50, contrast_increase=1.2):
    """
    Brighten and increase contrast of dark frames

    Args:
        frame: Input frame
        brightness_increase: Value to add to all pixels (0-100)
        contrast_increase: Contrast multiplier (1.0-2.0)
    """
    # Convert to float for processing
    frame_float = frame.astype(np.float32)

    # Increase brightness
    frame_float += brightness_increase

    # Increase contrast
    frame_float = (frame_float - 127.5) * contrast_increase + 127.5

    # Clip values
    frame_float = np.clip(frame_float, 0, 255)

    return frame_float.astype(np.uint8)


def auto_brightness_adjustment(frame):
    """
    Automatically determine if frame is too dark and adjust
    Uses histogram analysis
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate mean brightness
    mean_brightness = np.mean(gray)

    # Determine adjustment needed
    if mean_brightness < 80:  # Very dark
        brightness = 60
        contrast = 1.3
    elif mean_brightness < 120:  # Somewhat dark
        brightness = 30
        contrast = 1.15
    else:  # Acceptable brightness
        brightness = 0
        contrast = 1.0

    if brightness > 0:
        return brighten_frame(frame, brightness, contrast)
    else:
        return frame


def correct_fisheye(frame, strength=0.5):
    """
    Correct fisheye distortion
    strength: 0.0-1.0, where 1.0 is maximum correction
    """
    h, w = frame.shape[:2]

    # Camera matrix (approximate for GoPro)
    K = np.array([
        [w * 0.7, 0, w / 2],
        [0, h * 0.7, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients (approximate for GoPro medium FOV)
    # k1, k2, p1, p2, k3
    D = np.array([-0.3 * strength, 0.1 * strength, 0, 0, 0], dtype=np.float32)

    # Undistort
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, K, D, None, new_K)

    return undistorted


def preprocess_video(
    input_path,
    output_path,
    brightness='auto',
    zoom=True,
    zoom_padding=0.3,
    zoom_factor=None,
    fisheye_correction=False,
    fisheye_strength=0.5,
    output_resolution=None,
    max_frames=None
):
    """
    Preprocess video to improve pose detection

    Args:
        input_path: Input video path
        output_path: Output video path
        brightness: 'auto', 'none', or integer value (0-100)
        zoom: If True, auto-crop to player (uses player tracking)
        zoom_padding: Extra space around player (0.0-0.5)
        zoom_factor: If set, use simple center crop instead of tracking (1.5 = 1.5x zoom, 2.0 = 2x zoom)
        fisheye_correction: If True, correct GoPro fisheye
        fisheye_strength: Fisheye correction strength (0.0-1.0)
        output_resolution: Tuple (width, height) or None to keep original
        max_frames: Maximum frames to process (for testing)
    """

    print("=" * 70)
    print("VIDEO PREPROCESSING")
    print("=" * 70)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"\nSettings:")
    print(f"  Brightness: {brightness}")

    # Determine zoom mode
    if zoom_factor is not None:
        print(f"  Zoom mode: Static center crop ({zoom_factor}x)")
        zoom = False  # Disable tracking zoom
        use_static_zoom = True
    else:
        print(f"  Zoom mode: {'Player tracking' if zoom else 'None'}")
        if zoom:
            print(f"  Zoom padding: {zoom_padding * 100:.0f}%")
        use_static_zoom = False

    print(f"  Fisheye correction: {fisheye_correction}")
    if fisheye_correction:
        print(f"  Fisheye strength: {fisheye_strength}")
    print()

    # Open video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video {input_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"  Processing: First {max_frames} frames only")

    # Initialize pose detector if zoom is enabled
    pose = None
    if zoom:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("\nDetecting player position for auto-zoom...")

    # First pass: detect player positions if zoom enabled
    bboxes = []
    if zoom:
        print("Pass 1/2: Analyzing player positions...")
        frame_idx = 0

        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            bbox = auto_detect_player_bbox(frame, pose)
            if bbox:
                bboxes.append(bbox)
            else:
                # Use previous bbox if available
                if bboxes:
                    bboxes.append(bboxes[-1])
                else:
                    bboxes.append((0, 0, width, height))

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Analyzed {frame_idx}/{total_frames} frames...")

        # Smooth bounding boxes
        print("  Smoothing zoom transitions...")
        bboxes = smooth_bboxes(bboxes, window_size=15)

        # Add padding
        bboxes = [add_padding_to_bbox(b, (height, width), zoom_padding) for b in bboxes]

        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Determine output resolution
    if zoom and bboxes:
        # Use average bbox size (tracking zoom)
        avg_w = int(np.mean([b[2] for b in bboxes]))
        avg_h = int(np.mean([b[3] for b in bboxes]))
        out_width, out_height = avg_w, avg_h
        print(f"\nZoom crop size: {out_width}x{out_height}")
    elif use_static_zoom:
        # Static center crop
        out_width = int(width / zoom_factor)
        out_height = int(height / zoom_factor)
        print(f"\nStatic zoom: {width}x{height} → {out_width}x{out_height} ({zoom_factor}x)")
    elif output_resolution:
        out_width, out_height = output_resolution
    else:
        out_width, out_height = width, height

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"\nOutput video properties:")
    print(f"  Resolution: {out_width}x{out_height}")
    print(f"  FPS: {fps:.2f}")

    # Second pass: process and write frames
    print(f"\nPass {2 if zoom else 1}/{2 if zoom else 1}: Processing video...")
    frame_idx = 0

    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply fisheye correction
        if fisheye_correction:
            frame = correct_fisheye(frame, fisheye_strength)

        # Apply brightness adjustment
        if brightness == 'auto':
            frame = auto_brightness_adjustment(frame)
        elif brightness != 'none' and brightness > 0:
            frame = brighten_frame(frame, brightness_increase=brightness)

        # Apply zoom/crop
        if zoom and frame_idx < len(bboxes):
            # Tracking zoom - crop to player bbox
            x, y, w, h = bboxes[frame_idx]
            frame = frame[y:y+h, x:x+w]
            # Resize to consistent output size
            frame = cv2.resize(frame, (out_width, out_height))
        elif use_static_zoom:
            # Static center crop
            h, w = frame.shape[:2]
            # Calculate crop region (center)
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
            # Crop and resize
            frame = frame[y:y+crop_h, x:x+crop_w]
            frame = cv2.resize(frame, (out_width, out_height))

        # Write frame
        out.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)...")

    # Cleanup
    cap.release()
    out.release()
    if pose:
        pose.close()

    print(f"\n✅ Preprocessing complete!")
    print(f"   Processed {frame_idx} frames")
    print(f"   Output saved to: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess tennis videos for better pose detection'
    )

    parser.add_argument('input', help='Input video path')
    parser.add_argument('output', help='Output video path')

    # Brightness options
    brightness_group = parser.add_mutually_exclusive_group()
    brightness_group.add_argument(
        '--brighten',
        type=int,
        metavar='VALUE',
        help='Brightness increase value (0-100). Higher = brighter'
    )
    brightness_group.add_argument(
        '--auto-brighten',
        action='store_true',
        help='Automatically adjust brightness based on video darkness (recommended)'
    )
    brightness_group.add_argument(
        '--no-brighten',
        action='store_true',
        help='Do not adjust brightness'
    )

    # Zoom options
    zoom_group = parser.add_mutually_exclusive_group()
    zoom_group.add_argument(
        '--zoom',
        action='store_true',
        help='Auto-crop video to zoom in on player using tracking (slower)'
    )
    zoom_group.add_argument(
        '--static-zoom',
        type=float,
        metavar='FACTOR',
        help='Simple center crop zoom without tracking (e.g., 1.5 = 1.5x zoom, 2.0 = 2x zoom)'
    )
    parser.add_argument(
        '--zoom-padding',
        type=float,
        default=0.3,
        help='Extra space around player when using --zoom (0.0-0.5, default: 0.3 = 30%%)'
    )

    # Fisheye options
    # --fisheye: Boolean flag (on/off) that enables fisheye correction
    parser.add_argument(
        '--fisheye',
        action='store_true',
        help='Correct GoPro fisheye distortion'
    )
    # --fisheye-strength: Float parameter (0.0-1.0) that controls how much correction is applied
    # 0.3 = Mild correction (GoPro medium FOV)
    # 0.5 = Moderate correction (default)
    # 0.8 = Strong correction (GoPro wide FOV)
    # You must use --fisheye to enable it, then optionally adjust strength with --fisheye-strength.
    parser.add_argument(
        '--fisheye-strength',
        type=float,
        default=0.5,
        help='Fisheye correction strength (0.0-1.0, default: 0.5)'
    )

    # Other options
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Process only first N frames (for testing)'
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"ERROR: Input video not found: {args.input}")
        return

    # Determine brightness setting
    if args.auto_brighten:
        brightness = 'auto'
    elif args.no_brighten:
        brightness = 'none'
    elif args.brighten:
        brightness = args.brighten
    else:
        brightness = 'none'  # Default: OFF

    # Run preprocessing
    success = preprocess_video(
        input_path=args.input,
        output_path=args.output,
        brightness=brightness,
        zoom=args.zoom,
        zoom_padding=args.zoom_padding,
        zoom_factor=args.static_zoom,
        fisheye_correction=args.fisheye,
        fisheye_strength=args.fisheye_strength,
        max_frames=args.max_frames
    )

    if success:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Verify the output looks good:")
        print(f"   Open {args.output} and check:")
        print("   - Is player larger in frame?")
        print("   - Is brightness better?")
        print("   - Is video quality acceptable?")

        print("\n2. Check pose detection on preprocessed video:")
        print(f"   poetry run python src/visualize_pose.py {args.output} --max-frames 300")

        print("\n3. If satisfied, use preprocessed video for training")
        print("   - Replace original video with preprocessed version")
        print("   - Or update video paths in annotations")

        print()


if __name__ == "__main__":
    main()
