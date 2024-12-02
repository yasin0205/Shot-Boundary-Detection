import os
import time
import pandas as pd
import itertools
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import HistogramDetector
from multiprocessing import Pool

# Configuration dictionary to centralize inputs
config = {
    "video_directory": "./",  # Directory containing the video files
    "output_base_dir": "./output_histogram_detector",  # Directory to save results
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
    "thresholds": [0.05, 0.06],  # Threshold values for histogram differences
    "bins": [128, 256],  # Number of bins to use in histogram calculation
    "min_scene_lens": [10]  # Minimum scene length values
}


def predict_scenedetect(video_path, threshold=0.05, bins=256, min_scene_len=15):
    """Detect scenes in a video without saving output clips."""
    detected_frames_set = set()
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(HistogramDetector(
            threshold=threshold,
            bins=bins,
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
    itertools.product(config["thresholds"], config["bins"], config["min_scene_lens"])
)


# This function handles the processing for each video
def process_video(video_file):
    video_path = os.path.join(config["video_directory"], video_file)
    gt_key = video_file[0]  # Assumes video files are labeled "A.mp4", "B.mp4", etc.
    ground_truth_frames = config["ground_truth"].get(gt_key, [])

    video_results = []

    for params in param_combinations:
        threshold, bins, min_scene_len = params
        process_start_time = time.time()

        num_scenes, detected_frames_set = predict_scenedetect(
            video_path,
            threshold=threshold,
            bins=bins,
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
            'Threshold': threshold,
            'Bins': bins,
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
    csv_file_path = os.path.join(config["output_base_dir"], 'grid_search_results_histogram.csv')
    results_df.to_csv(csv_file_path, index=False)

    print("Grid search completed and summary file created as 'grid_search_results_histogram.csv'.")


if __name__ == "__main__":
    main()
