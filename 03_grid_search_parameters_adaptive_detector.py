"""
===============================================================================
PYSCENEDETECT ADAPTIVE DETECTOR — PARALLEL GRID SEARCH EVALUATION FRAMEWORK
===============================================================================

Purpose
-------
This script benchmarks the AdaptiveDetector scene cut algorithm from
PySceneDetect by running a multi-parameter grid search and comparing
detected shot boundaries against manually labeled ground truth frames.

Unlike a simple detection script, this file is designed as a
RESEARCH EXPERIMENT PIPELINE.

The pipeline performs:

    1) Scene cut detection using AdaptiveDetector
    2) Hyperparameter tuning via grid search
    3) Parallel processing across videos
    4) Accuracy evaluation (Precision / Recall / F1 Score)
    5) CSV export for analysis and comparison

Typical usage:
    - Classical vs deep learning comparison
    - Selecting optimal AdaptiveDetector parameters
    - Producing quantitative experiment tables


===============================================================================
WHAT ADAPTIVE DETECTOR DOES
===============================================================================

AdaptiveDetector identifies scene changes using frame-to-frame content change.

Instead of comparing only colors (HistogramDetector),
this method analyzes intensity variation over time
and adapts sensitivity dynamically.

Conceptually:

    Small variation → same shot
    Large variation → new shot boundary

Better at:
    motion transitions
    camera pans
    broadcast cuts

Weaker at:
    very gradual fades
    extreme lighting flashes


===============================================================================
INPUT / OUTPUT STRUCTURE
===============================================================================

Input:
    ./A.mp4
    ./B.mp4
    ./C.mp4
    ...

Output:
    ./output_adaptive_detector/
        grid_search_results_adaptive_detector.csv

NOTE:
No video clips are created.
This script only evaluates detection accuracy.


===============================================================================
GROUND TRUTH MATCHING
===============================================================================

Each video has manually labeled cut frames:

    "A": [0, 405, 451, 519, ...]

Since detection rarely matches exact frame numbers,
we allow tolerance:

    |detected_frame − ground_truth_frame| ≤ margin

Default:
    margin = 25 frames (~1 second @25fps)

This avoids penalizing near-correct detections.


===============================================================================
GRID SEARCH (PARAMETER OPTIMIZATION)
===============================================================================

We test combinations of AdaptiveDetector parameters:

adaptive_threshold
    sensitivity to change magnitude

window_width
    smoothing window across frames

min_content_val
    minimum change needed to trigger detection

luma_only
    brightness-only comparison vs color comparison

min_scene_len
    minimum allowed shot duration

All combinations generated using:

    itertools.product(...)

For EACH video:
    For EACH parameter combination:
        run detection
        compute metrics
        store results


===============================================================================
MULTIPROCESSING
===============================================================================

Videos are processed in parallel using:

    multiprocessing.Pool()

Each CPU core processes a separate video simultaneously.

This significantly reduces experiment runtime during grid search.


===============================================================================
METRICS COMPUTED
===============================================================================

True Positive (TP)
    Correctly detected scene cut

False Positive (FP)
    Detector predicted non-existent cut

False Negative (FN)
    Detector missed real cut


Precision
---------
    TP / (TP + FP)
    Prediction reliability

Recall
------
    TP / (TP + FN)
    Detection completeness

F1 Score
--------
    Harmonic balance between precision and recall


===============================================================================
RESULT CSV CONTENT
===============================================================================

grid_search_results_adaptive_detector.csv contains:

    Video name
    Parameter configuration
    Number of scenes detected
    Ground truth frames
    Detected frames
    TP / FP / FN
    Precision / Recall / F1
    False positive frames
    False negative frames
    Processing time

Rows sorted by:
    Video name → Highest F1 score first

This allows selecting the best parameter configuration.


===============================================================================
PIPELINE FLOW
===============================================================================

collect videos
    ↓
parallel processing
    ↓
for each parameter set
    ↓
detect scene boundaries
    ↓
compare with ground truth
    ↓
compute metrics
    ↓
save CSV report


===============================================================================
WHY THIS SCRIPT EXISTS
===============================================================================

HistogramDetector → color-based detection
AdaptiveDetector → motion-based detection
TransNetV2 → deep learning detection

This script provides a quantitative baseline to compare
classical algorithms against neural networks.


===============================================================================
LIMITATIONS
===============================================================================

- Requires ground truth annotations
- Sensitive to parameter selection
- No clip export (evaluation only)
- Performance varies across video styles


===============================================================================
END OF DOCUMENTATION
===============================================================================
"""

import os
import time
import pandas as pd
import itertools
from scenedetect import VideoManager, SceneManager, AdaptiveDetector
from multiprocessing import Pool

# Configuration dictionary to centralize inputs
config = {
    "video_directory": "./",  # Directory containing the video files
    "output_base_dir": "./output_adaptive_detector",  # Directory to save results
    "ground_truth": {  # Ground truth frame values per video
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
    "margin": 25,  # Margin for evaluation metrics
    "adaptive_thresholds": [1.5, 2.0],  # List of adaptive threshold values
    "min_scene_lens": [15, 16],  # List of minimum scene lengths
    "window_widths": [6],  # List of window width values
    "min_content_vals": [14.0],  # List of minimum content values
    "luma_only_options": [True]  # Options for luma_only parameter
}


def predict_scenedetect(video_path, adaptive_threshold=3.0, window_width=2,
                        min_content_val=15.0, luma_only=False, min_scene_len=15):
    """Detect scenes in a video without saving output clips."""
    detected_frames_set = set()
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            window_width=window_width,
            luma_only=luma_only,
            min_scene_len=min_scene_len
        ))
        video_manager.start()
        scene_manager.detect_scenes(video_manager)
        scenes = scene_manager.get_scene_list()
        for scene in scenes:
            start_frame = scene[0].get_frames()
            detected_frames_set.add(start_frame)
    finally:
        video_manager.release()
    return len(scenes), detected_frames_set


def calculate_results_with_margin(detected_frames_set, ground_truth_frames, margin):
    """Calculate TP, FP, FN, and the FP/FN frames with a specified margin."""
    true_positives = 0
    matched_detected_frames = set()
    false_positives_frames = set(detected_frames_set)
    false_negatives_frames = set(ground_truth_frames)

    ground_truth_frames = sorted(ground_truth_frames)
    detected_frames_set = sorted(detected_frames_set)

    for gt_frame in ground_truth_frames:
        for detected_frame in detected_frames_set:
            if detected_frame not in matched_detected_frames and abs(detected_frame - gt_frame) <= margin:
                true_positives += 1
                matched_detected_frames.add(detected_frame)
                false_positives_frames.discard(detected_frame)
                false_negatives_frames.discard(gt_frame)
                break

    false_positives = len(false_positives_frames)
    false_negatives = len(false_negatives_frames)

    return true_positives, false_positives, false_negatives, sorted(false_positives_frames), sorted(
        false_negatives_frames)


# Generate all combinations of parameters
param_combinations = list(
    itertools.product(config["adaptive_thresholds"], config["window_widths"],
                      config["min_content_vals"], config["luma_only_options"],
                      config["min_scene_lens"])
)

# This function handles the processing for each video
def process_video(video_file):
    video_path = os.path.join(config["video_directory"], video_file)
    gt_key = video_file[0]  # Assumes video files are labeled "A.mp4", "B.mp4", etc.
    ground_truth_frames = config["ground_truth"].get(gt_key, [])

    video_results = []

    for params in param_combinations:
        adaptive_threshold, window_width, min_content_val, luma_only, min_scene_len = params
        process_start_time = time.time()

        num_scenes, detected_frames_set = predict_scenedetect(
            video_path,
            adaptive_threshold=adaptive_threshold,
            window_width=window_width,
            min_content_val=min_content_val,
            luma_only=luma_only,
            min_scene_len=min_scene_len
        )

        process_end_time = time.time()
        processing_time = process_end_time - process_start_time

        true_positives, false_positives, false_negatives, fp_frames, fn_frames = calculate_results_with_margin(
            detected_frames_set, ground_truth_frames, margin=config["margin"]
        )

        precision = true_positives / (true_positives + false_positives) if (
                                                                                       true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (
                                                                                    true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        video_results.append({
            'Video': video_file,
            'Adaptive Threshold': adaptive_threshold,
            'Window Width': window_width,
            'Min Content Val': min_content_val,
            'Luma Only': luma_only,
            'Min Scene Len': min_scene_len,
            'Num Scenes Generated': num_scenes,
            'Ground Truth Frame Values': ', '.join(map(str, ground_truth_frames)),
            'Detected Frame Values': ', '.join(map(str, sorted(detected_frames_set))),
            'True Positives': true_positives,
            'False Positives': false_positives,
            'False Negatives': false_negatives,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'FP Frames': ', '.join(map(str, fp_frames)),
            'FN Frames': ', '.join(map(str, fn_frames)),
            'Processing Time (s)': f"{processing_time:.2f}"
        })

    return video_results


def main():
    # Get all video files to process
    video_files = [f for f in os.listdir(config["video_directory"]) if f.endswith('.mp4')]

    # Use multiprocessing to process videos in parallel
    with Pool() as pool:
        all_results = pool.map(process_video, video_files)

    # Flatten the list of results
    results = [item for sublist in all_results for item in sublist]

    # Sort results by Video and F1 Score
    sorted_results = sorted(results, key=lambda x: (x['Video'], -x['F1 Score']))

    # Save results to CSV
    results_df = pd.DataFrame(sorted_results)
    csv_file_path = os.path.join(config["output_base_dir"], 'grid_search_results_adaptive_detector.csv')
    results_df.to_csv(csv_file_path, index=False)

    print("Grid search completed and summary file created as 'grid_search_results_adaptive_detector.csv'.")


if __name__ == "__main__":
    main()
