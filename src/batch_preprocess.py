"""
Batch Video Preprocessing
Process multiple videos at once with same settings
"""

import argparse
from pathlib import Path
import subprocess
import sys

def batch_preprocess(
    input_dir,
    output_dir,
    pattern='*.MP4',
    auto_brighten=False,
    zoom=False,
    zoom_padding=0.3,
    static_zoom=None,
    fisheye=False,
    fisheye_strength=0.5,
    max_frames=None
):
    """
    Batch preprocess all videos in a directory

    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save output videos
        pattern: File pattern (e.g., '*.MP4', '*.mp4', '*.avi')
        auto_brighten: Automatically adjust brightness
        zoom: Auto-crop to player
        zoom_padding: Extra space around player
        fisheye: Correct fisheye distortion
        fisheye_strength: Fisheye correction strength
        max_frames: Limit frames per video (for testing)
    """

    print("=" * 70)
    print("BATCH VIDEO PREPROCESSING")
    print("=" * 70)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all videos
    video_files = list(input_path.glob(pattern))

    if not video_files:
        print(f"\nERROR: No videos found matching pattern '{pattern}' in {input_dir}")
        return

    print(f"\nFound {len(video_files)} videos to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nSettings:")
    print(f"  Auto-brighten: {auto_brighten}")

    if static_zoom:
        print(f"  Zoom mode: Static center crop ({static_zoom}x)")
    else:
        print(f"  Zoom mode: {'Player tracking' if zoom else 'None'}")
        if zoom:
            print(f"  Zoom padding: {zoom_padding * 100:.0f}%")

    print(f"  Fisheye correction: {fisheye}")
    if fisheye:
        print(f"  Fisheye strength: {fisheye_strength}")
    if max_frames:
        print(f"  Max frames per video: {max_frames}")
    print()

    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print("\n" + "=" * 70)
        print(f"VIDEO {i}/{len(video_files)}: {video_file.name}")
        print("=" * 70)

        # Determine output filename
        output_file = output_path / f"{video_file.stem}_preprocessed{video_file.suffix}"

        # Build command
        cmd = [
            sys.executable,  # Use current Python interpreter
            'src/preprocess_video.py',
            str(video_file),
            str(output_file)
        ]

        if auto_brighten:
            cmd.append('--auto-brighten')

        if static_zoom:
            cmd.extend(['--static-zoom', str(static_zoom)])
        elif zoom:
            cmd.extend(['--zoom', '--zoom-padding', str(zoom_padding)])

        if fisheye:
            cmd.extend(['--fisheye', '--fisheye-strength', str(fisheye_strength)])

        if max_frames:
            cmd.extend(['--max-frames', str(max_frames)])

        # Run preprocessing
        try:
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                print(f"\n✅ Successfully processed: {video_file.name}")
            else:
                print(f"\n❌ Failed to process: {video_file.name}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error processing {video_file.name}: {e}")
        except KeyboardInterrupt:
            print("\n\n⚠️  Batch processing interrupted by user")
            print(f"Processed {i-1}/{len(video_files)} videos")
            return

    # Summary
    print("\n" + "=" * 70)
    print("BATCH PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(video_files)} videos")
    print(f"Outputs saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Verify outputs look good:")
    print(f"   Check files in {output_dir}")

    print("\n2. Test pose detection on preprocessed videos:")
    print(f"   poetry run python src/visualize_pose.py {output_dir}/video_preprocessed.MP4 --max-frames 300")

    print("\n3. Compare original vs preprocessed:")
    print("   - Open both videos side-by-side")
    print("   - Check if player is larger")
    print("   - Check if brightness is better")

    print("\n4. If satisfied, use preprocessed videos for training")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Batch preprocess multiple tennis videos'
    )

    parser.add_argument(
        'input_dir',
        help='Directory containing input videos'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to save preprocessed videos'
    )
    parser.add_argument(
        '--pattern',
        default='*.MP4',
        help='File pattern to match (default: *.MP4)'
    )

    # Brightness
    brightness_group = parser.add_mutually_exclusive_group()
    brightness_group.add_argument(
        '--auto-brighten',
        action='store_true',
        help='Auto-adjust brightness'
    )
    brightness_group.add_argument(
        '--no-brighten',
        action='store_true',
        help='Do not adjust brightness'
    )

    # Zoom
    zoom_group = parser.add_mutually_exclusive_group()
    zoom_group.add_argument(
        '--zoom',
        action='store_true',
        help='Auto-crop to player with tracking'
    )
    zoom_group.add_argument(
        '--static-zoom',
        type=float,
        metavar='FACTOR',
        help='Simple center crop zoom (e.g., 1.5 = 1.5x zoom, faster than --zoom)'
    )
    zoom_group.add_argument(
        '--no-zoom',
        action='store_true',
        help='Do not zoom'
    )
    parser.add_argument(
        '--zoom-padding',
        type=float,
        default=0.3,
        help='Zoom padding for --zoom mode (default: 0.3)'
    )

    # Fisheye
    parser.add_argument(
        '--fisheye',
        action='store_true',
        help='Correct fisheye distortion'
    )
    parser.add_argument(
        '--fisheye-strength',
        type=float,
        default=0.5,
        help='Fisheye strength (default: 0.5)'
    )

    # Testing
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Process only first N frames per video (for testing)'
    )

    args = parser.parse_args()

    # Validate directories
    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return

    # Run batch preprocessing
    batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        auto_brighten=args.auto_brighten,
        zoom=args.zoom,
        zoom_padding=args.zoom_padding,
        static_zoom=args.static_zoom,
        fisheye=args.fisheye,
        fisheye_strength=args.fisheye_strength,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
