"""
Utility script to verify Label Studio annotations before training
Helps identify issues with annotations and video files
"""

import json
import cv2
from pathlib import Path
from collections import Counter
from datetime import datetime

def verify_annotations(json_folder=r'D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\label_studio_exports',
                       video_folder=r'D:\Mestrado\redes_neurais\dados_filtrados\videos',
                       output_file=None):
    """Verify all Label Studio JSON files and corresponding videos

    Args:
        json_folder: Path to folder containing JSON annotation files
        video_folder: Path to folder containing video files
        output_file: Optional path to save verification report (markdown or txt format)
    """

    # Build report content
    report = []

    def add_to_report(line):
        """Add line to both console and report"""
        print(line)
        report.append(line)

    add_to_report("=" * 70)
    add_to_report("LABEL STUDIO ANNOTATION VERIFICATION")
    add_to_report(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add_to_report("=" * 70)

    json_files = list(Path(json_folder).glob('*.json'))

    if len(json_files) == 0:
        add_to_report(f"\n❌ ERROR: No JSON files found in {json_folder}")
        return

    add_to_report(f"\n✅ Found {len(json_files)} annotation files\n")

    all_labels = []
    total_annotations = 0
    videos_ok = 0
    videos_missing = 0
    file_details = []

    for json_file in json_files:
        file_info = {'name': json_file.name, 'video_found': False, 'annotations': []}

        add_to_report("=" * 70)
        add_to_report(f"File: {json_file.name}")
        add_to_report("=" * 70)

        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            data = data[0]

        # Get video path - handle both old and new Label Studio formats
        try:
            video_path = data['task']['data']['video']
        except KeyError:
            video_path = data['data']['video']

        # Extract relative path from Label Studio format
        # Label Studio format: "/data/local-files/?d=videos/Bhand_1.MP4"
        # We want: "Bhand_1.MP4"
        video_filename = video_path.replace('/data/local-files/?d=', '')
        # Remove the 'videos/' prefix if present, since video_folder already points to the videos directory
        if video_filename.startswith('videos/') or video_filename.startswith('videos\\'):
            video_filename = video_filename[7:]  # Remove 'videos/' or 'videos\'

        # Construct full path, ensuring proper path separator
        full_video_path = str(Path(video_folder) / video_filename)

        file_info['video_filename'] = video_filename
        file_info['video_path'] = full_video_path

        add_to_report(f"Video: {video_filename}")

        # Check if video exists
        if Path(full_video_path).exists():
            add_to_report(f"✅ Video found")
            file_info['video_found'] = True
            videos_ok += 1

            # Get video info
            cap = cv2.VideoCapture(full_video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                file_info['video_info'] = {
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'frames': frame_count,
                    'duration': duration
                }

                add_to_report(f"  Resolution: {width}x{height}")
                add_to_report(f"  FPS: {fps:.2f}")
                add_to_report(f"  Frames: {frame_count}")
                add_to_report(f"  Duration: {duration:.2f}s")

                cap.release()
            else:
                add_to_report(f"  ⚠️  Warning: Cannot open video file")
        else:
            add_to_report(f"❌ Video NOT found: {full_video_path}")
            videos_missing += 1

        # Parse annotations - handle both old and new Label Studio formats
        result_data = None
        if 'result' in data and isinstance(data['result'], list):
            # New format: annotations directly in 'result'
            result_data = data['result']
        elif 'annotations' in data and data['annotations'] and len(data['annotations']) > 0:
            # Old format: annotations in 'annotations[0]['result']'
            result_data = data['annotations'][0]['result']

        if result_data:
            num_annotations = len(result_data)
            total_annotations += num_annotations

            add_to_report(f"\nAnnotations: {num_annotations}")

            for i, result in enumerate(result_data, 1):
                if result['type'] == 'timelinelabels':
                    for range_item in result['value']['ranges']:
                        start_frame = range_item['start']
                        end_frame = range_item['end']
                        label = result['value']['timelinelabels'][0]
                        duration_frames = end_frame - start_frame

                        all_labels.append(label.lower())
                        file_info['annotations'].append({
                            'label': label,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'duration_frames': duration_frames
                        })

                        add_to_report(f"  {i}. {label}: frames {start_frame}-{end_frame} "
                              f"({duration_frames} frames)")

                        # Check for issues
                        if duration_frames <= 0:
                            add_to_report(f"     ⚠️  WARNING: Invalid duration!")
                        elif duration_frames < 10:
                            add_to_report(f"     ⚠️  WARNING: Very short segment (< 10 frames)")
                        elif duration_frames > 180:
                            add_to_report(f"     ℹ️  Info: Long segment (> 6 seconds at 30fps)")
        else:
            add_to_report("\n⚠️  No annotations found in this file")

        file_details.append(file_info)
        add_to_report("")

    # Summary
    add_to_report("=" * 70)
    add_to_report("SUMMARY")
    add_to_report("=" * 70)

    add_to_report(f"\nFiles:")
    add_to_report(f"  Total JSON files: {len(json_files)}")
    add_to_report(f"  Videos found: {videos_ok} ✅")
    add_to_report(f"  Videos missing: {videos_missing} ❌")

    add_to_report(f"\nAnnotations:")
    add_to_report(f"  Total annotations: {total_annotations}")

    if len(all_labels) > 0:
        label_counts = Counter(all_labels)
        add_to_report(f"\nLabel distribution:")
        for label, count in label_counts.most_common():
            percentage = (count / len(all_labels)) * 100
            add_to_report(f"  {label}: {count} ({percentage:.1f}%)")

        # Check balance
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        balance_ratio = min_count / max_count if max_count > 0 else 0

        add_to_report(f"\nClass balance:")
        if balance_ratio > 0.7:
            add_to_report(f"  ✅ Good balance (ratio: {balance_ratio:.2f})")
        elif balance_ratio > 0.4:
            add_to_report(f"  ⚠️  Moderate imbalance (ratio: {balance_ratio:.2f})")
            add_to_report(f"     Consider adding more samples of underrepresented classes")
        else:
            add_to_report(f"  ❌ Significant imbalance (ratio: {balance_ratio:.2f})")
            add_to_report(f"     Training may be biased toward majority class")
            add_to_report(f"     Strongly recommend balancing your dataset")

    # Recommendations
    add_to_report(f"\n{'='*70}")
    add_to_report("RECOMMENDATIONS")
    add_to_report(f"{'='*70}")

    if videos_missing > 0:
        add_to_report("\n❌ Fix missing videos:")
        add_to_report("   - Check video file paths in CONFIG['video_base_path']")
        add_to_report("   - Ensure video filenames match Label Studio annotations")
        add_to_report("   - Video files should be in the 'videos/' folder")

    if total_annotations < 30:
        add_to_report("\n⚠️  Limited data:")
        add_to_report(f"   - You have {total_annotations} annotated segments")
        add_to_report("   - Aim for at least 50+ per class for good results")
        add_to_report("   - With limited data, expect 60-75% accuracy")
    elif total_annotations < 100:
        add_to_report("\n✅ Moderate amount of data:")
        add_to_report(f"   - You have {total_annotations} annotated segments")
        add_to_report("   - Should achieve 70-85% accuracy")
        add_to_report("   - More data will improve results")
    else:
        add_to_report("\n✅ Good amount of data:")
        add_to_report(f"   - You have {total_annotations} annotated segments")
        add_to_report("   - Should achieve 80-90%+ accuracy")

    if len(all_labels) > 0:
        unique_labels = set(all_labels)
        if len(unique_labels) == 1:
            add_to_report("\n⚠️  Only one class found:")
            add_to_report("   - You need at least 2 classes (forehand + backhand)")
            add_to_report("   - Add annotations for other stroke types")

    add_to_report("\n✅ Ready to train!" if videos_missing == 0 else "\n❌ Fix issues before training")

    # Save report to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Check if markdown format
            if output_path.suffix.lower() in ['.md', '.markdown']:
                # Write as markdown
                f.write("# Label Studio Annotation Verification Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")

                # Summary section
                f.write("## Summary\n\n")
                f.write(f"- **Total JSON files:** {len(json_files)}\n")
                f.write(f"- **Videos found:** {videos_ok} ✅\n")
                f.write(f"- **Videos missing:** {videos_missing} ❌\n")
                f.write(f"- **Total annotations:** {total_annotations}\n\n")

                # Label distribution
                if len(all_labels) > 0:
                    f.write("## Label Distribution\n\n")
                    label_counts = Counter(all_labels)
                    f.write("| Label | Count | Percentage |\n")
                    f.write("|-------|-------|------------|\n")
                    for label, count in label_counts.most_common():
                        percentage = (count / len(all_labels)) * 100
                        f.write(f"| {label} | {count} | {percentage:.1f}% |\n")
                    f.write("\n")

                # File details
                f.write("## File Details\n\n")
                for file_info in file_details:
                    f.write(f"### {file_info['name']}\n\n")
                    f.write(f"- **Video:** {file_info['video_filename']}\n")
                    f.write(f"- **Status:** {'✅ Found' if file_info['video_found'] else '❌ Not Found'}\n")

                    if 'video_info' in file_info:
                        vi = file_info['video_info']
                        f.write(f"- **Resolution:** {vi['resolution']}\n")
                        f.write(f"- **FPS:** {vi['fps']:.2f}\n")
                        f.write(f"- **Frames:** {vi['frames']}\n")
                        f.write(f"- **Duration:** {vi['duration']:.2f}s\n")

                    if file_info['annotations']:
                        f.write(f"\n**Annotations ({len(file_info['annotations'])}):**\n\n")
                        f.write("| # | Label | Start Frame | End Frame | Duration |\n")
                        f.write("|---|-------|-------------|-----------|----------|\n")
                        for i, ann in enumerate(file_info['annotations'], 1):
                            f.write(f"| {i} | {ann['label']} | {ann['start_frame']} | {ann['end_frame']} | {ann['duration_frames']} |\n")
                    else:
                        f.write("\n⚠️ No annotations\n")
                    f.write("\n")

                # Full console output
                f.write("---\n\n## Full Console Output\n\n```\n")
                f.write('\n'.join(report))
                f.write("\n```\n")
            else:
                # Write as plain text
                f.write('\n'.join(report))

        print(f"\n📄 Report saved to: {output_path}")

def visualize_annotation(json_file, video_folder='videos/', output_folder='./verification_output/'):
    """
    Visualize annotations on video frames
    Saves sample frames showing annotated segments
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        data = data[0]

    # Get video - handle both old and new Label Studio formats
    try:
        video_path = data['task']['data']['video']
    except KeyError:
        video_path = data['data']['video']

    # Extract relative path from Label Studio format
    # Label Studio format: "/data/local-files/?d=videos/Bhand_1.MP4"
    # We want: "Bhand_1.MP4"
    video_filename = video_path.replace('/data/local-files/?d=', '')
    # Remove the 'videos/' prefix if present, since video_folder already points to the videos directory
    if video_filename.startswith('videos/') or video_filename.startswith('videos\\'):
        video_filename = video_filename[7:]  # Remove 'videos/' or 'videos\'

    # Construct full path, ensuring proper path separator
    full_video_path = str(Path(video_folder) / video_filename)

    if not Path(full_video_path).exists():
        print(f"Video not found: {full_video_path}")
        return

    cap = cv2.VideoCapture(full_video_path)

    # Get annotations - handle both old and new Label Studio formats
    result_data = None
    if 'result' in data and isinstance(data['result'], list):
        # New format: annotations directly in 'result'
        result_data = data['result']
    elif 'annotations' in data and data['annotations'] and len(data['annotations']) > 0:
        # Old format: annotations in 'annotations[0]['result']'
        result_data = data['annotations'][0]['result']

    if result_data:
        for i, result in enumerate(result_data, 1):
            if result['type'] == 'timelinelabels':
                for range_item in result['value']['ranges']:
                    start_frame = range_item['start']
                    end_frame = range_item['end']
                    label = result['value']['timelinelabels'][0]

                    # Extract middle frame
                    mid_frame = (start_frame + end_frame) // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                    ret, frame = cap.read()

                    if ret:
                        # Add label text
                        cv2.putText(frame, f"{label} (Frame {mid_frame})",
                                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                   2, (0, 255, 0), 3)

                        # Save frame
                        output_name = f"{Path(json_file).stem}_annotation_{i}_frame_{mid_frame}.jpg"
                        output_path = f"{output_folder}/{output_name}"
                        cv2.imwrite(output_path, frame)
                        print(f"Saved: {output_path}")

    cap.release()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify Label Studio annotations')
    parser.add_argument('--json-folder',
        default=r'D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\label_studio_exports',
        help='Folder containing JSON files')
    parser.add_argument('--video-folder',
        default=r'D:\Mestrado\redes_neurais\dados_filtrados\videos',
        help='Folder containing videos')
    parser.add_argument('--output-file',
        default=None,
        help='Path to save verification report (e.g., report.md or report.txt)')
    parser.add_argument('--visualize',
        action='store_true',
        help='Generate visualization images')

    args = parser.parse_args()

    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = 'verification_report.md'

    # Run verification
    verify_annotations(args.json_folder, args.video_folder, args.output_file)

    # Optionally visualize
    if args.visualize:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        json_files = list(Path(args.json_folder).glob('*.json'))
        for json_file in json_files:
            visualize_annotation(json_file, args.video_folder)
