"""
===============================================================================
PYSCENEDETECT HISTOGRAM DETECTOR — SCENE SEGMENTATION + CLIP EXPORT + METRICS
===============================================================================

Purpose
-------
This script performs scene (shot) boundary detection using the classical
HistogramDetector algorithm from PySceneDetect, exports each detected scene
as an individual video clip, and evaluates detection accuracy against
ground-truth frame annotations.

This is a full processing pipeline:

    1) Detect shot boundaries
    2) Generate scene clips
    3) Compare with ground truth
    4) Compute evaluation metrics
    5) Save summary CSV

Typical use:
    - Video dataset preparation
    - Highlight segmentation preprocessing
    - Classical baseline evaluation
    - Comparing with AdaptiveDetector / TransNetV2


===============================================================================
CORE IDEA — HISTOGRAM-BASED SCENE DETECTION
===============================================================================

Each video frame has a color distribution (histogram).

If consecutive frames have very different color distributions,
it indicates a shot change.

Conceptually:

    Similar histogram  → same camera shot
    Different histogram → new scene cut

This method detects:
    camera switches
    replay transitions
    editing cuts

But struggles with:
    flashes
    gradual fades
    rapid lighting changes


===============================================================================
INPUT / OUTPUT STRUCTURE
===============================================================================

Input directory:
    ./
        A.mp4
        B.mp4
        ...

Output directory:
    ./output_histogram_detector/
        A/
            clip_1.mp4
            clip_2.mp4
            ...
        summary_histogram_detector.csv

Each video gets its own folder of scene clips.


===============================================================================
SCENE DETECTION PROCESS
===============================================================================

PySceneDetect returns scenes as time ranges:

    [(start_time, end_time), ...]

We convert them to frame numbers:

    frame_number = scene_start.get_frames()

Each start frame represents a detected cut boundary.


===============================================================================
CLIP GENERATION
===============================================================================

For each detected scene:

    1) Seek video to start frame
    2) Read frames sequentially
    3) Write frames into a new video file

OpenCV settings:
    codec = mp4v
    fps   = fixed (25)
    resolution = original video

Each shot becomes an independent playable clip.


===============================================================================
GROUND TRUTH MATCHING (WITH TOLERANCE)
===============================================================================

Human annotations rarely match exact frame numbers.
Therefore a detection is considered correct if:

    |detected_frame − ground_truth_frame| ≤ margin

Default:
    margin = 25 frames (~1 second at 25fps)

This prevents penalizing near-correct predictions.


===============================================================================
METRICS CALCULATED
===============================================================================

True Positive (TP)
    Detected cut matches real cut

False Positive (FP)
    Detector predicted extra cut

False Negative (FN)
    Detector missed real cut


Precision
---------
    TP / (TP + FP)
    Reliability of detections

Recall
------
    TP / (TP + FN)
    Coverage of real cuts

F1 Score
--------
    Balanced accuracy measure


===============================================================================
PARAMETERS
===============================================================================

threshold
    Sensitivity of histogram difference
    lower → more cuts (higher FP)
    higher → fewer cuts (higher FN)

bins
    Histogram resolution
    larger bins = finer color comparison

min_scene_len
    Minimum frames allowed per scene
    prevents noisy rapid cuts


===============================================================================
RESULT CSV CONTENT
===============================================================================

summary_histogram_detector.csv contains:

    video name
    parameter configuration
    number of scenes
    ground truth frames
    detected frames
    TP / FP / FN
    precision / recall / F1
    false positive frames
    false negative frames
    processing time


===============================================================================
PIPELINE FLOW
===============================================================================

for each video:
    detect scenes
    export clips
    compute metrics
    store results

finally:
    save CSV summary


===============================================================================
WHY THIS SCRIPT EXISTS
===============================================================================

HistogramDetector → color-based classical baseline
AdaptiveDetector  → motion-based classical baseline
TransNetV2        → deep learning method

This script provides a reproducible baseline segmentation dataset.


===============================================================================
LIMITATIONS
===============================================================================

- Sensitive to brightness flashes
- Not robust to gradual transitions
- Fixed FPS output
- Slower due to clip exporting


===============================================================================
END OF DOCUMENTATION
===============================================================================
"""

import os
import time
import pandas as pd
import itertools
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import HistogramDetector


def calculate_results_with_margin(detected_frames_set, ground_truth_frames, margin):
    """Calculate TP, FP, and FN with a specified margin."""
    true_positives = 0
    matched_detected_frames = set()
    false_positive_frames = set()
    false_negative_frames = set()

    ground_truth_frames = sorted(ground_truth_frames)
    detected_frames_set = sorted(detected_frames_set)

    for gt_frame in ground_truth_frames:
        matched = False
        for detected_frame in detected_frames_set:
            if detected_frame not in matched_detected_frames and abs(detected_frame - gt_frame) <= margin:
                true_positives += 1
                matched_detected_frames.add(detected_frame)
                matched = True
                break
        if not matched:
            false_negative_frames.add(gt_frame)

    for detected_frame in detected_frames_set:
        if detected_frame not in matched_detected_frames:
            false_positive_frames.add(detected_frame)

    false_positives = len(false_positive_frames)
    false_negatives = len(false_negative_frames)

    return true_positives, false_positives, false_negatives, false_positive_frames, false_negative_frames


def predict_scenedetect(video_path, config):
    """Detect scenes in a video using HistogramDetector without saving output clips."""
    detected_frames_set = set()
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(HistogramDetector(
            threshold=config['threshold'],
            bins=config['bins'],
            min_scene_len=config['min_scene_len']
        ))

        video_manager.start()
        scene_manager.detect_scenes(video_manager)
        scenes = scene_manager.get_scene_list()

        for scene in scenes:
            start_frame = scene[0].get_frames()
            detected_frames_set.add(start_frame)

    finally:
        video_manager.release()

    return scenes, detected_frames_set


def create_clips_from_scenes(video_path, scenes, output_dir, fps):
    """Create individual clips for each scene using OpenCV."""
    os.makedirs(output_dir, exist_ok=True)

    video_capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for i, (start_time, end_time) in enumerate(scenes):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip_filename = os.path.join(output_dir, f"clip_{i + 1}.mp4")
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))

        print(f"Creating clip: {clip_filename} from frame {start_frame} to {end_frame}")
        for frame_num in range(start_frame, end_frame):
            ret, frame = video_capture.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"Clip saved successfully: {clip_filename}")

    video_capture.release()


def process_videos(video_directory, ground_truth, config):
    """Process all mp4 files in the video directory and return results."""
    results = []
    param_combinations = list(itertools.product(*config['param_ranges']))

    for video_file in os.listdir(video_directory):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_directory, video_file)
            gt_key = video_file[0]

            for params in param_combinations:
                process_start_time = time.time()

                scenes, detected_frames_set = predict_scenedetect(video_path, {
                    'threshold': params[0],
                    'bins': params[1],
                    'min_scene_len': params[2]
                })

                processing_time = time.time() - process_start_time

                # Calculate metrics and track FP and FN frames
                true_positives, false_positives, false_negatives, false_positive_frames, false_negative_frames = calculate_results_with_margin(
                    detected_frames_set, ground_truth[gt_key], config['margin'])

                precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0
                f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

                output_dir = os.path.join(config['output_base_dir'], os.path.splitext(video_file)[0])
                create_clips_from_scenes(video_path, scenes, output_dir, config['fps'])

                results.append({
                    'Video': video_file,
                    'Threshold': params[0],
                    'Bins': params[1],
                    'Min Scene Length': params[2],
                    'Num Scenes Generated': len(scenes),
                    'Ground Truth Frame Values': ', '.join(map(str, ground_truth[gt_key])),
                    'Detected Frame Values': ', '.join(map(str, sorted(detected_frames_set))),
                    'True Positives': true_positives,
                    'False Positives': false_positives,
                    'False Negatives': false_negatives,
                    'False Positive Frame Numbers': ', '.join(map(str, sorted(false_positive_frames))),
                    'False Negative Frame Numbers': ', '.join(map(str, sorted(false_negative_frames))),
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1_score,
                    'Processing Time (s)': f"{processing_time:.2f}"
                })

    return results


def save_results_to_csv(results, output_base_dir):
    """Save the results to a CSV file."""
    sorted_results = sorted(results, key=lambda x: x['Video'], reverse=False)
    results_df = pd.DataFrame(sorted_results)
    csv_file_path = os.path.join(output_base_dir, 'summary_histogram_detector.csv')
    results_df.to_csv(csv_file_path, index=False, float_format="%.2f")


def main():
    config = {
        'video_directory': "./",
        'output_base_dir': "./output_histogram_detector",
        'ground_truth': {
            "A": [0, 405, 451, 519, 569, 659, 702, 728, 967, 978, 1178, 1295, 1398, 1481, 1621, 1630],
            "B": [0, 306, 417, 553, 657, 733, 843, 910, 1001, 1013, 1268, 1394, 1673, 1838, 1852],
            "C": [0, 376, 922, 931, 1108, 1175, 1265, 1621, 1631, 1681],
            "D": [0, 352, 460, 545, 651, 731, 931, 942, 1145, 1398, 1542, 1720, 1732, 1780],
            "E": [0, 56, 85, 144, 413, 657, 803, 901, 947, 1016, 1028, 1212, 1347, 1563, 1710, 1799, 1811],
            "F": [0, 83, 95, 358, 515, 574, 664, 709, 798, 1036, 1049, 1222, 1413, 1604, 1738, 1750],
            "G": [0, 318, 458, 559, 608, 781, 876, 888, 1262, 1396, 1536, 1744, 1903, 1999],
            "H": [0, 35, 134, 183, 338, 542, 610, 692, 746, 814, 878, 887, 1037, 1207, 1391, 1399, 1484, 1511],
            "I": [0, 25, 48, 311, 488, 622, 686, 748, 763, 1093, 1107],
            "J": [0, 21, 34, 311, 457, 557, 615, 829, 904, 992, 1005, 1176, 1405, 1611, 1756, 1768, 1823, 1920]
        },
        'margin': 25,
        'fps': 25,
        'param_ranges': [
            [0.05],  # threshold values
            [1024],  # histogram bins
            [10]  # min scene length
        ]
    }

    results = process_videos(config['video_directory'], config['ground_truth'], config)
    save_results_to_csv(results, config['output_base_dir'])

    print("Processing completed and summary file created as 'summary_histogram_detector.csv'.")


if __name__ == '__main__':
    main()
