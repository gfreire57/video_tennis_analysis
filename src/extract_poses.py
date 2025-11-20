"""
Extract and Save Pose Landmarks

This script extracts pose landmarks from all videos ONCE and saves them to disk.
This avoids re-extracting poses every time you train the model.

Usage:
    poetry run python src/extract_poses.py
    poetry run python src/extract_poses.py --output-dir ./pose_data
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

# Import from train_model
import train_model


def extract_and_save_poses(video_base_path, label_studio_exports, output_dir, force_reextract=False):
    """Extract poses from all videos and save to disk

    Args:
        video_base_path: Directory containing videos
        label_studio_exports: Directory containing Label Studio JSON annotations
        output_dir: Directory to save extracted pose data
        force_reextract: If True, re-extract even if files exist
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("POSE EXTRACTION AND SAVING")
    print("="*70)
    print(f"Video directory: {video_base_path}")
    print(f"Annotations directory: {label_studio_exports}")
    print(f"Output directory: {output_dir}")
    print(f"Force re-extract: {force_reextract}")
    print("="*70)
    print()

    # Initialize pose extractor
    pose_extractor = train_model.PoseExtractor()

    # Get all annotation files
    json_files = list(Path(label_studio_exports).glob('*.json'))
    print(f"Found {len(json_files)} annotation files\n")

    extracted_count = 0
    skipped_count = 0
    error_count = 0

    for idx, json_file in enumerate(json_files, 1):
        print(f"\n[{idx}/{len(json_files)}] Processing: {json_file.name}")
        print("-" * 70)

        # Parse annotations
        video_filename, annotations = train_model.parse_label_studio_json_frames(json_file)
        video_path = Path(video_base_path) / video_filename

        print(f"Video: {video_filename}")
        print(f"Annotations: {len(annotations)} strokes")

        # Create safe filename for output
        video_stem = Path(video_filename).stem
        output_file = output_dir / f"{video_stem}_poses.npz"

        # Check if already extracted
        if output_file.exists() and not force_reextract:
            print(f"✓ Already extracted: {output_file.name}")
            print("  Use --force to re-extract")
            skipped_count += 1
            continue

        # Check if video exists
        if not video_path.exists():
            print(f"✗ Video not found: {video_path}")
            error_count += 1
            continue

        # Extract poses
        print(f"Extracting poses from: {video_path}")
        landmarks, fps = pose_extractor.extract_from_video(str(video_path))

        if landmarks is None or len(landmarks) == 0:
            print(f"✗ Failed to extract poses")
            error_count += 1
            continue

        print(f"✓ Extracted {len(landmarks)} frames of pose data")

        # Save to disk
        data = {
            'landmarks': landmarks,
            'fps': fps,
            'video_filename': video_filename,
            'annotations': annotations,
            'num_frames': len(landmarks),
            'extracted_at': datetime.now().isoformat(),
        }

        np.savez_compressed(output_file, **data)

        # Verify saved file
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved to: {output_file.name} ({file_size_mb:.2f} MB)")

        extracted_count += 1

    # Summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total videos: {len(json_files)}")
    print(f"  Extracted: {extracted_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"\nPose data saved to: {output_dir}")
    print("="*70)

    # Create metadata file
    metadata = {
        'extraction_date': datetime.now().isoformat(),
        'video_base_path': str(video_base_path),
        'label_studio_exports': str(label_studio_exports),
        'total_videos': len(json_files),
        'extracted': extracted_count,
        'skipped': skipped_count,
        'errors': error_count,
        'pose_extractor_config': {
            'model_complexity': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
        }
    }

    metadata_file = output_dir / 'extraction_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_file}")


def load_saved_poses(pose_data_dir):
    """Load all saved pose data from disk

    Args:
        pose_data_dir: Directory containing saved .npz files

    Returns:
        List of dictionaries with pose data for each video
    """
    pose_data_dir = Path(pose_data_dir)

    if not pose_data_dir.exists():
        raise FileNotFoundError(f"Pose data directory not found: {pose_data_dir}")

    pose_files = list(pose_data_dir.glob('*_poses.npz'))

    if len(pose_files) == 0:
        raise FileNotFoundError(f"No pose files found in {pose_data_dir}")

    print(f"\nLoading {len(pose_files)} saved pose files from: {pose_data_dir}")

    all_pose_data = []

    for pose_file in pose_files:
        data = np.load(pose_file, allow_pickle=True)

        pose_data = {
            'landmarks': data['landmarks'],
            'fps': float(data['fps']),
            'video_filename': str(data['video_filename']),
            'annotations': data['annotations'].tolist(),
            'num_frames': int(data['num_frames']),
        }

        all_pose_data.append(pose_data)
        print(f"  ✓ Loaded: {pose_file.name} ({pose_data['num_frames']} frames)")

    print(f"✓ Successfully loaded {len(all_pose_data)} pose datasets\n")

    return all_pose_data


def verify_pose_data(pose_data_dir):
    """Verify integrity of saved pose data

    Args:
        pose_data_dir: Directory containing saved .npz files
    """
    pose_data_dir = Path(pose_data_dir)

    print("\n" + "="*70)
    print("VERIFYING POSE DATA")
    print("="*70)

    pose_files = list(pose_data_dir.glob('*_poses.npz'))

    if len(pose_files) == 0:
        print(f"✗ No pose files found in {pose_data_dir}")
        return

    print(f"Found {len(pose_files)} pose files\n")

    total_frames = 0
    total_annotations = 0

    for pose_file in pose_files:
        try:
            data = np.load(pose_file, allow_pickle=True)

            landmarks = data['landmarks']
            fps = float(data['fps'])
            video_filename = str(data['video_filename'])
            annotations = data['annotations'].tolist()

            total_frames += len(landmarks)
            total_annotations += len(annotations)

            print(f"✓ {pose_file.name}")
            print(f"    Video: {video_filename}")
            print(f"    Frames: {len(landmarks)}, FPS: {fps}")
            print(f"    Annotations: {len(annotations)}")
            print(f"    Shape: {landmarks.shape}")

        except Exception as e:
            print(f"✗ {pose_file.name}: ERROR - {e}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total pose files: {len(pose_files)}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total annotations: {total_annotations}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract and save pose landmarks from videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--video-dir', type=str,
                       default=train_model.CONFIG['video_base_path'],
                       help='Directory containing videos')
    parser.add_argument('--annotations-dir', type=str,
                       default=train_model.CONFIG['label_studio_exports'],
                       help='Directory containing Label Studio JSON annotations')
    parser.add_argument('--output-dir', type=str,
                       default='./pose_data',
                       help='Directory to save extracted pose data')
    parser.add_argument('--force', action='store_true',
                       help='Force re-extraction even if files exist')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing pose data (do not extract)')

    args = parser.parse_args()

    if args.verify:
        verify_pose_data(args.output_dir)
    else:
        extract_and_save_poses(
            video_base_path=args.video_dir,
            label_studio_exports=args.annotations_dir,
            output_dir=args.output_dir,
            force_reextract=args.force
        )
