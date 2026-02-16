"""
===============================================================================
PYSCENEDETECT ADAPTIVE DETECTOR — SCENE SEGMENTATION + PARAMETER BENCHMARKING
===============================================================================

Purpose
-------
This script performs shot boundary detection using the classical computer-vision
algorithm AdaptiveDetector from PySceneDetect.

Unlike the TransNetV2 deep learning script, this file implements a
TRADITIONAL video segmentation pipeline and evaluates detection quality against
ground-truth annotations.

The script performs four tasks:

    1) Detect scene cuts using AdaptiveDetector
    2) Generate video clips for each detected scene
    3) Compare detected cuts with human annotations
    4) Benchmark detector parameters and save metrics

This is designed for:
    - comparing classical vs deep learning scene detection
    - finding optimal detection thresholds
    - building reproducible experiments


===============================================================================
KEY CONCEPT — WHAT IS A SCENE CUT?
===============================================================================

A video is composed of shots:

    Shot A → Cut → Shot B → Cut → Shot C

A scene cut occurs when visual content changes abruptly
(camera switch, replay, transition, etc).

AdaptiveDetector identifies cuts by measuring frame-to-frame
content change using luminance and histogram differences.


===============================================================================
INPUT / OUTPUT STRUCTURE
===============================================================================

Input:
    All .mp4 videos in working directory

Output:
    ./output_adaptive_detector/
        A/
            clip_1.mp4
            clip_2.mp4
            ...
        B/
            clip_1.mp4
            ...
        summary_adaptive_detector.csv

Each video gets its own folder of extracted scenes.


===============================================================================
HOW ADAPTIVE DETECTOR WORKS
===============================================================================

The algorithm compares consecutive frames and computes content change.

If change > threshold → new scene detected

Core parameters:

adaptive_threshold
    Sensitivity to change
    lower = more cuts
    higher = fewer cuts

window_width
    Number of frames used to smooth detection

min_content_val
    Minimum difference needed to consider cut

luma_only
    Use brightness only instead of full color

min_scene_len
    Minimum allowed length of a scene in frames


===============================================================================
PARAMETER SWEEP (EXPERIMENT MODE)
===============================================================================

This script does NOT just run once.
It tests combinations of detector parameters:

param_ranges → multiple configurations
itertools.product → generates combinations

For each video:
    For each parameter configuration:
        run detection
        compute metrics
        save results

This allows hyperparameter optimization.


===============================================================================
DETECTED FRAMES REPRESENTATION
===============================================================================

PySceneDetect returns scenes as timecodes:

    [(start_time, end_time), ...]

We convert to frame numbers:

    frame_number = scene_start.get_frames()

Detected frames represent shot boundaries.


===============================================================================
GROUND TRUTH MATCHING (WITH TOLERANCE)
===============================================================================

Human annotations and detectors rarely match exactly.

Therefore we use a margin window:

    |detected_frame − ground_truth_frame| ≤ margin

Default:
    margin = 25 frames

Meaning:
    prediction within ±1 second (at 25 fps) counts as correct


===============================================================================
METRICS CALCULATED
===============================================================================

TP — True Positive
    Detected cut matches real cut

FP — False Positive
    Detector found a cut where none exists

FN — False Negative
    Detector missed a real cut


Precision
---------
    TP / (TP + FP)
    Reliability of predictions

Recall
------
    TP / (TP + FN)
    Completeness of detection

F1 Score
--------
    Harmonic mean of precision and recall


===============================================================================
CLIP GENERATION
===============================================================================

Each detected scene is exported as a standalone video.

Process:
    seek video → start_frame
    read frames → until end_frame
    write to new file

OpenCV VideoWriter settings:
    codec = mp4v
    fps   = fixed (25)
    resolution = original video


===============================================================================
RESULT CSV CONTENT
===============================================================================

summary_adaptive_detector.csv contains:

    video name
    parameter configuration
    number of scenes detected
    ground truth frames
    detected frames
    TP / FP / FN
    precision / recall / F1
    false positive frames
    false negative frames
    processing time

This file allows quantitative comparison across configurations.


===============================================================================
PIPELINE FLOW
===============================================================================

for each video:
    for each parameter combination:
        detect scenes
        extract clips
        compute metrics
        store results

finally:
    save all experiment results to CSV


===============================================================================
WHY THIS SCRIPT EXISTS
===============================================================================

TransNetV2 → modern deep learning detector
AdaptiveDetector → classical baseline

This script allows direct comparison:

    classical vs neural network segmentation performance


===============================================================================
LIMITATIONS
===============================================================================

- Sensitive to flashes and camera motion
- Parameter tuning required per dataset
- Slower due to repeated runs (benchmark mode)
- Uses fixed FPS when writing clips


===============================================================================
END OF DOCUMENTATION
===============================================================================
"""


import os
import time
import pandas as pd
import itertools
import cv2
from scenedetect import VideoManager, SceneManager, AdaptiveDetector


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
    """Detect scenes in a video without saving output clips."""
    detected_frames_set = set()
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(**config))

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    for i, (start_time, end_time) in enumerate(scenes):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Move to the start frame
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

        out.release()  # Release the writer for the current clip
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
                    **config['default_params'],
                    **dict(zip(config['param_keys'], params))
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

                # Save clips using OpenCV
                output_dir = os.path.join(config['output_base_dir'], os.path.splitext(video_file)[0])
                create_clips_from_scenes(video_path, scenes, output_dir, config['fps'])

                results.append({
                    'Video': video_file,
                    'Adaptive Threshold': params[0],
                    'Window Width': params[1],
                    'Min Content Val': params[2],
                    'Luma Only': params[3],
                    'Min Scene Len': params[4],
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
    csv_file_path = os.path.join(output_base_dir, 'summary_adaptive_detector.csv')
    results_df.to_csv(csv_file_path, index=False, float_format="%.2f")


def main():
    config = {
        'video_directory': "./",
        'output_base_dir': "./output_adaptive_detector",
        'ground_truth': {
            "A": [0, 16, 314, 390, 510, 525, 658, 795, 893, 908, 1146, 1235, 1633, 1693],
            "B": [0, 124, 162, 293, 483, 559, 669, 706, 830, 832, 1070, 1266, 1596, 1597],
            "C": [0, 115, 165, 301, 438, 499, 706, 782, 850, 927, 936, 1168, 1342, 1472, 1483, 1928, 1982],
            "D": [0, 310, 786, 807, 1139, 1229, 1341, 1445, 1521, 1544],
            "E": [0, 348, 409, 534, 571, 616, 682, 711, 813, 870, 918, 1020, 1029, 1284, 1411, 1566, 1575, 1804, 1956]
        },
        'margin': 25,
        'fps': 25,  # Frame rate for saving clips
        'default_params': {
            'adaptive_threshold': 3.0,
            'window_width': 2,
            'luma_only': False,
            'min_scene_len': 15
        },
        'param_ranges': [
            [1.7],  # adaptive thresholds 1.8
            [8],  # window widths 6
            [15],  # min content values 12.0
            [False],  # luma only options False
            [15]  # min scene lengths 10
        ],
        'param_keys': ['adaptive_threshold', 'window_width', 'min_content_val', 'luma_only', 'min_scene_len']
    }

    # Process videos and save results
    results = process_videos(config['video_directory'], config['ground_truth'], config)
    save_results_to_csv(results, config['output_base_dir'])

    print("Processing completed and summary file created as 'summary_adaptive_detector.csv'.")


if __name__ == '__main__':
    main()
