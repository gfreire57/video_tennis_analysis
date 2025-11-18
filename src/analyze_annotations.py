"""
Analyze Label Studio annotations to help calibrate window size

This script reads all annotation files and generates a report showing:
- Number of annotations per video
- Frame ranges for each stroke
- Length of each annotation
- Statistics (min, max, mean, median length)
"""

import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict


def parse_label_studio_json(json_path):
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
    # Remove the 'videos/' prefix if present
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
                        'label': label.lower(),  # Normalize to lowercase
                        'length': int(end_frame) - int(start_frame)
                    })

    return video_path, annotations


def analyze_annotations(annotation_dir):
    """
    Analyze all annotation files and generate report

    Args:
        annotation_dir: Directory containing Label Studio JSON exports

    Returns:
        dict with analysis results
    """
    annotation_dir = Path(annotation_dir)
    json_files = sorted(annotation_dir.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {annotation_dir}")
        return None

    print(f"Found {len(json_files)} annotation files\n")
    print("=" * 70)

    # Statistics per video and overall
    all_lengths = []
    all_lengths_by_class = defaultdict(list)
    video_stats = []

    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        print("=" * 70)

        try:
            video_path, annotations = parse_label_studio_json(json_file)

            if not annotations:
                print(f"Video: {video_path}")
                print("Annotations: 0")
                print("  ⚠ No annotations found")
                continue

            print(f"Video: {video_path}")
            print(f"Annotations: {len(annotations)}")

            # Sort annotations by start frame
            annotations = sorted(annotations, key=lambda x: x['start_frame'])

            # Display each annotation
            for anno in annotations:
                start = anno['start_frame']
                end = anno['end_frame']
                length = anno['length']
                label = anno['label']

                print(f"  - {label}: frames {start}-{end} ({length} frames)")

                # Collect statistics
                all_lengths.append(length)
                all_lengths_by_class[label].append(length)

            # Video-level statistics
            lengths = [a['length'] for a in annotations]
            video_stats.append({
                'video': video_path,
                'num_annotations': len(annotations),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'mean_length': np.mean(lengths),
                'median_length': np.median(lengths)
            })

        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {e}")
            continue

    # Overall statistics
    if all_lengths:
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)
        print(f"\nTotal annotations: {len(all_lengths)}")
        print(f"\nAnnotation lengths (frames):")
        print(f"  Minimum:  {min(all_lengths):>6} frames")
        print(f"  Maximum:  {max(all_lengths):>6} frames")
        print(f"  Mean:     {np.mean(all_lengths):>6.1f} frames")
        print(f"  Median:   {np.median(all_lengths):>6.1f} frames")
        print(f"  Std Dev:  {np.std(all_lengths):>6.1f} frames")

        # Percentiles
        print(f"\nPercentiles:")
        print(f"  25th:     {np.percentile(all_lengths, 25):>6.1f} frames")
        print(f"  50th:     {np.percentile(all_lengths, 50):>6.1f} frames")
        print(f"  75th:     {np.percentile(all_lengths, 75):>6.1f} frames")
        print(f"  90th:     {np.percentile(all_lengths, 90):>6.1f} frames")
        print(f"  95th:     {np.percentile(all_lengths, 95):>6.1f} frames")

        # Statistics by class
        print("\n" + "=" * 70)
        print("STATISTICS BY CLASS")
        print("=" * 70)

        for label in sorted(all_lengths_by_class.keys()):
            lengths = all_lengths_by_class[label]
            print(f"\n{label.upper()}:")
            print(f"  Count:    {len(lengths):>6}")
            print(f"  Minimum:  {min(lengths):>6} frames")
            print(f"  Maximum:  {max(lengths):>6} frames")
            print(f"  Mean:     {np.mean(lengths):>6.1f} frames")
            print(f"  Median:   {np.median(lengths):>6.1f} frames")
            print(f"  Std Dev:  {np.std(lengths):>6.1f} frames")

        # Window size recommendations
        print("\n" + "=" * 70)
        print("WINDOW SIZE RECOMMENDATIONS")
        print("=" * 70)

        min_length = min(all_lengths)
        median_length = np.median(all_lengths)
        p25_length = np.percentile(all_lengths, 25)

        print(f"\nBased on your annotations:")
        print(f"\n  Conservative (fit smallest annotations):")
        print(f"    window_size = {min_length} frames")
        print(f"    ↳ Will work with ALL annotations (100% coverage)")

        print(f"\n  Moderate (fit 75% of annotations):")
        print(f"    window_size = {int(p25_length)} frames")
        print(f"    ↳ Will work with 75% of annotations")

        print(f"\n  Relaxed (fit 50% of annotations):")
        print(f"    window_size = {int(median_length)} frames")
        print(f"    ↳ Will work with 50% of annotations")

        print(f"\n  Note: Annotations shorter than window_size will be rejected")
        print(f"        unless majority voting threshold is adjusted.")

        # Save report to file
        save_report(video_stats, all_lengths, all_lengths_by_class, annotation_dir)

    else:
        print("\n⚠ No annotations found in any file")

    return {
        'all_lengths': all_lengths,
        'by_class': all_lengths_by_class,
        'video_stats': video_stats
    }


def save_report(video_stats, all_lengths, all_lengths_by_class, annotation_dir):
    """Save detailed report to file"""
    report_path = annotation_dir.parent / 'annotation_analysis_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ANNOTATION ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Per-video statistics
        f.write("PER-VIDEO STATISTICS\n")
        f.write("=" * 70 + "\n\n")

        for stat in video_stats:
            f.write(f"Video: {stat['video']}\n")
            f.write(f"  Annotations: {stat['num_annotations']}\n")
            f.write(f"  Min length:  {stat['min_length']} frames\n")
            f.write(f"  Max length:  {stat['max_length']} frames\n")
            f.write(f"  Mean length: {stat['mean_length']:.1f} frames\n")
            f.write(f"  Median:      {stat['median_length']:.1f} frames\n\n")

        # Overall statistics
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total annotations: {len(all_lengths)}\n\n")
        f.write(f"Annotation lengths (frames):\n")
        f.write(f"  Minimum:  {min(all_lengths):>6} frames\n")
        f.write(f"  Maximum:  {max(all_lengths):>6} frames\n")
        f.write(f"  Mean:     {np.mean(all_lengths):>6.1f} frames\n")
        f.write(f"  Median:   {np.median(all_lengths):>6.1f} frames\n")
        f.write(f"  Std Dev:  {np.std(all_lengths):>6.1f} frames\n\n")

        f.write(f"Percentiles:\n")
        f.write(f"  25th:     {np.percentile(all_lengths, 25):>6.1f} frames\n")
        f.write(f"  50th:     {np.percentile(all_lengths, 50):>6.1f} frames\n")
        f.write(f"  75th:     {np.percentile(all_lengths, 75):>6.1f} frames\n")
        f.write(f"  90th:     {np.percentile(all_lengths, 90):>6.1f} frames\n")
        f.write(f"  95th:     {np.percentile(all_lengths, 95):>6.1f} frames\n\n")

        # By class
        f.write("=" * 70 + "\n")
        f.write("STATISTICS BY CLASS\n")
        f.write("=" * 70 + "\n\n")

        for label in sorted(all_lengths_by_class.keys()):
            lengths = all_lengths_by_class[label]
            f.write(f"{label.upper()}:\n")
            f.write(f"  Count:    {len(lengths):>6}\n")
            f.write(f"  Minimum:  {min(lengths):>6} frames\n")
            f.write(f"  Maximum:  {max(lengths):>6} frames\n")
            f.write(f"  Mean:     {np.mean(lengths):>6.1f} frames\n")
            f.write(f"  Median:   {np.median(lengths):>6.1f} frames\n")
            f.write(f"  Std Dev:  {np.std(lengths):>6.1f} frames\n\n")

        # Recommendations
        f.write("=" * 70 + "\n")
        f.write("WINDOW SIZE RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")

        min_length = min(all_lengths)
        median_length = np.median(all_lengths)
        p25_length = np.percentile(all_lengths, 25)

        f.write(f"Conservative (100% coverage):\n")
        f.write(f"  window_size = {min_length} frames\n\n")

        f.write(f"Moderate (75% coverage):\n")
        f.write(f"  window_size = {int(p25_length)} frames\n\n")

        f.write(f"Relaxed (50% coverage):\n")
        f.write(f"  window_size = {int(median_length)} frames\n\n")

    print(f"\n{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze Label Studio annotations to calibrate window size'
    )
    parser.add_argument(
        'annotation_dir',
        type=str,
        nargs='?',
        default=r'D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\label_studio_exports',
        help='Directory containing Label Studio JSON exports'
    )

    args = parser.parse_args()

    analyze_annotations(args.annotation_dir)
