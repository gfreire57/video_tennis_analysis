"""
Visualize MediaPipe Pose Estimation on Video
Shows skeleton overlay on video to verify pose detection is working
"""

import cv2
import mediapipe as mp
import argparse
from pathlib import Path
import numpy as np

# MediaPipe drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def visualize_pose_on_video(video_path, output_path=None, show_live=True, max_frames=None):
    """
    Visualize pose estimation on video

    Args:
        video_path: Path to input video
        output_path: Optional path to save output video
        show_live: If True, show video in window (press 'q' to quit)
        max_frames: Optional maximum number of frames to process
    """

    print("=" * 70)
    print("MEDIAPIPE POSE VISUALIZATION")
    print("=" * 70)
    print(f"\nInput video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")

    if max_frames:
        print(f"  Processing: First {max_frames} frames only")

    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\nOutput will be saved to: {output_path}")

    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("\n" + "=" * 70)
    print("PROCESSING VIDEO")
    print("=" * 70)
    if show_live:
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 'p' to pause/resume")
        print("  - Press 's' to save current frame as image")
    print()

    frame_idx = 0
    frames_with_pose = 0
    frames_without_pose = 0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()

            if not ret:
                break

            # Check max_frames limit
            if max_frames and frame_idx >= max_frames:
                break

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process pose
            results = pose.process(image_rgb)

            # Create annotated image
            annotated_image = frame.copy()

            if results.pose_landmarks:
                frames_with_pose += 1

                # Draw pose landmarks on image
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Add status text
                status_text = f"Frame {frame_idx+1}/{total_frames if not max_frames else max_frames} - POSE DETECTED"
                color = (0, 255, 0)  # Green
            else:
                frames_without_pose += 1

                # Add warning text
                status_text = f"Frame {frame_idx+1}/{total_frames if not max_frames else max_frames} - NO POSE DETECTED"
                color = (0, 0, 255)  # Red

            # Add status text to frame
            cv2.putText(
                annotated_image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Add detection rate
            detection_rate = (frames_with_pose / (frame_idx + 1)) * 100
            rate_text = f"Detection rate: {detection_rate:.1f}%"
            cv2.putText(
                annotated_image,
                rate_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Write to output video if enabled
            if out:
                out.write(annotated_image)

            # Show live if enabled
            if show_live:
                cv2.imshow('MediaPipe Pose Visualization (Press q to quit, p to pause, s to save frame)', annotated_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"\n{'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    screenshot_path = f"pose_frame_{frame_idx}.png"
                    cv2.imwrite(screenshot_path, annotated_image)
                    print(f"\nSaved frame to: {screenshot_path}")

            frame_idx += 1

            # Progress update
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames... Detection rate: {detection_rate:.1f}%")

        else:
            # Paused - just check for key presses
            if show_live:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"\n{'Paused' if paused else 'Resumed'}")

    # Cleanup
    cap.release()
    if out:
        out.release()
    if show_live:
        cv2.destroyAllWindows()
    pose.close()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal frames processed: {frame_idx}")
    print(f"Frames with pose detected: {frames_with_pose} ({frames_with_pose/frame_idx*100:.1f}%)")
    print(f"Frames without pose: {frames_without_pose} ({frames_without_pose/frame_idx*100:.1f}%)")

    if frames_with_pose / frame_idx < 0.5:
        print("\n⚠️  WARNING: Pose detection rate is low (<50%)")
        print("   Possible issues:")
        print("   - Player not fully visible in frame")
        print("   - Camera angle too far or too close")
        print("   - Poor lighting")
        print("   - Player occluded by net/objects")
        print("   - Video quality too low")
    elif frames_with_pose / frame_idx < 0.8:
        print("\n⚠️  Detection rate is OK but could be better (50-80%)")
        print("   Some frames may not be usable for training")
    else:
        print("\n✅ Good detection rate (>80%)!")
        print("   Video is suitable for training")

    if output_path:
        print(f"\n✅ Output video saved to: {output_path}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize MediaPipe pose estimation on tennis video'
    )
    parser.add_argument(
        'video_path',
        help='Path to input video file'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Path to save output video with pose overlay (optional)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not show live video window (useful for headless systems)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum number of frames to process (for quick testing)'
    )

    args = parser.parse_args()

    # Validate input video exists
    if not Path(args.video_path).exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        return

    # Run visualization
    visualize_pose_on_video(
        video_path=args.video_path,
        output_path=args.output,
        show_live=not args.no_display,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
