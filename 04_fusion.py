"""
===============================================================================
DETECTION FUSION PIPELINE — TRANSNETV2 + ADAPTIVE DETECTOR ENSEMBLE
===============================================================================

Purpose
-------
This script combines (fuses) shot boundary detections produced by:

    1) TransNetV2 (deep learning detector)
    2) PySceneDetect AdaptiveDetector (classical motion-based detector)

The goal is to improve scene boundary accuracy by merging detections from
two independent methods and re-evaluating performance against ground truth.

The pipeline performs:

    1) Load detection results from CSV outputs of both methods
    2) Merge detections using a tolerance-based union strategy
    3) Compute evaluation metrics (Precision / Recall / F1)
    4) Generate fused video clips
    5) Save final summary CSV

This implements a DECISION-LEVEL ENSEMBLE approach.


===============================================================================
WHY FUSION WORKS
===============================================================================

Different detectors fail in different situations:

TransNetV2 (Deep Learning)
    + Good at semantic changes
    + Handles motion and broadcast transitions
    - Sometimes misses subtle edits

AdaptiveDetector (Classical)
    + Sensitive to sudden motion/brightness changes
    + Good at sharp edits
    - False positives during camera motion

Combining them reduces individual weaknesses.


===============================================================================
INPUT FILES
===============================================================================

From previous scripts:

./output_adaptive_detector/summary_adaptive_detector.csv
./output_transnet_v2/summary_transnet_v2.csv

Each CSV contains detected frame indices for each video.

The script merges both detections per video.


===============================================================================
FRAME UNION LOGIC
===============================================================================

We combine detected frames from both detectors.

Problem:
    Two detectors may detect the SAME cut but at slightly different frames.

Solution:
    If two frames differ by ≤ FRAME_DIFF_THRESHOLD (default = 2 frames)
    → treat them as the same boundary

Algorithm:
    1) Combine both frame lists
    2) Sort frames
    3) Add frame only if no existing frame is within tolerance

Result:
    A cleaned union set of boundaries.


===============================================================================
GROUND TRUTH MATCHING
===============================================================================

We compare fused detections to annotated ground truth.

A detection is correct if:

    |detected_frame − ground_truth_frame| ≤ MARGIN

Default:
    MARGIN = 25 frames (~1 second @25fps)

This accounts for annotation ambiguity.


===============================================================================
METRICS COMPUTED
===============================================================================

True Positive (TP)
    Union detection matches real cut

False Positive (FP)
    Union detection not present in ground truth

False Negative (FN)
    Ground truth cut not detected

Precision
---------
    TP / (TP + FP)

Recall
------
    TP / (TP + FN)

F1 Score
--------
    Harmonic balance between precision and recall


===============================================================================
VIDEO CLIP GENERATION
===============================================================================

After fusion, clips are generated using union frame boundaries.

For each video:

    boundary[i] → boundary[i+1]
        becomes a new clip

Last boundary → end of video
        becomes final clip

OpenCV VideoWriter settings:
    codec = mp4v
    fps   = original video fps
    resolution = original video resolution


===============================================================================
OUTPUT STRUCTURE
===============================================================================

./output_fusion/
    00.csv                → raw merged detection table
    summary_fusion.csv    → final evaluation results

    A/
        A_clip_1.mp4
        A_clip_2.mp4
        ...
    B/
        B_clip_1.mp4
        ...


===============================================================================
SUMMARY CSV CONTENT
===============================================================================

summary_fusion.csv contains:

    Video
    TP / FP / FN
    Precision / Recall / F1 Score
    Frames detected by each method
    Fused union frames
    False positive frames
    False negative frames
    Processing time

This allows comparison:

    Adaptive vs TransNetV2 vs Fusion


===============================================================================
PIPELINE FLOW
===============================================================================

load CSV detections
        ↓
merge frame detections (union with tolerance)
        ↓
evaluate accuracy vs ground truth
        ↓
generate fused clips
        ↓
save summary CSV


===============================================================================
WHY THIS SCRIPT EXISTS
===============================================================================

Individual detectors:
    have complementary strengths

Fusion:
    increases recall without severely harming precision

This approximates an ensemble model commonly used in
computer vision research.


===============================================================================
LIMITATIONS
===============================================================================

- Requires outputs from previous scripts
- Union strategy may increase false positives
- No weighting between detectors (simple union)
- Processing time increases due to clip export


===============================================================================
END OF DOCUMENTATION
===============================================================================
"""


import os
import pandas as pd
import cv2
import time

# Configuration section for adjustable parameters
FRAME_DIFF_THRESHOLD = 2  # Frame difference threshold for union frames
MARGIN = 25  # Margin for true positives and false negatives

# Set paths
pyscene_detect_path = "./output_adaptive_detector/summary_adaptive_detector.csv"
transnetV2_detect_path = "./output_transnet_v2/summary_transnet_v2.csv"
output_base_path = "./output_fusion"
video_directory = "./"

# Import the CSV files into DataFrames
df_pyscene_detect = pd.read_csv(pyscene_detect_path)
df_transnetV2_detect = pd.read_csv(transnetV2_detect_path)

# Clean and prepare data: rename columns to make them more readable
df_pyscene_detect.rename(columns={
    'Ground Truth Frame Values': 'Ground Truth Frames',
    'Num Scenes Generated': 'Total Scene Detect by PySceneDetect',
    'Detected Frame Values': 'Detected Frames by PySceneDetect'
}, inplace=True)

df_transnetV2_detect.rename(columns={
    'Video File Name': 'Video',
    'Total Detected Frames': 'Total Scene Detect by TransNetV2',
    'Detected Frames': 'Detected Frames by TransNetV2'
}, inplace=True)

# Merge DataFrames on the 'Video' column
df = pd.merge(
    df_pyscene_detect[['Video', 'Detected Frames by PySceneDetect', 'Ground Truth Frames']],
    df_transnetV2_detect[['Video', 'Detected Frames by TransNetV2']],
    on='Video', how='inner'
)

# Clean the 'Video' column by removing '.mp4' extensions for consistency
df['Video'] = df['Video'].str.replace('.mp4', '', regex=False)
df = df.sort_values(by='Video', ascending=True)


# Function to calculate the union of detected frames from both methods
def calculate_union(row):
    transnet_frames = list(map(int, row['Detected Frames by TransNetV2'].split(', ')))
    pyscene_frames = list(map(int, row['Detected Frames by PySceneDetect'].split(', ')))

    # Combine frames with a tolerance of ±FRAME_DIFF_THRESHOLD
    union_frames = set()
    all_frames = sorted(transnet_frames + pyscene_frames)

    for frame in all_frames:
        if not any(abs(frame - u_frame) <= FRAME_DIFF_THRESHOLD for u_frame in union_frames):
            union_frames.add(frame)

    return sorted(union_frames)



# Apply the union calculation to create a new 'Union' column
df['Union'] = df.apply(calculate_union, axis=1)


# Function to calculate TP, FP, FN, Precision, Recall, F1-Score, and FP/FN frames
def calculate_metrics(row):
    ground_truth = sorted(list(map(int, row['Ground Truth Frames'].split(','))))
    union_frames = sorted(row['Union'])

    TP = 0  # True Positives
    FP_frames = []  # False Positives
    FN_frames = []  # False Negatives
    matched_union = set()  # To track matched union frames

    # Match ground truth frames with union frames within ±MARGIN tolerance
    for gt_frame in ground_truth:
        match_found = False
        for union_frame in union_frames:
            if union_frame in matched_union:
                continue  # Skip already matched frames
            if abs(gt_frame - union_frame) <= MARGIN:
                TP += 1
                matched_union.add(union_frame)
                match_found = True
                break
        if not match_found:
            FN_frames.append(gt_frame)  # No match found

    # Remaining union frames are false positives
    FP_frames = [frame for frame in union_frames if frame not in matched_union]

    FP = len(FP_frames)
    FN = len(FN_frames)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return pd.Series([TP, FP, FN, precision, recall, f1_score, FP_frames, FN_frames],
                     index=['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1 Score', 'FP Frames', 'FN Frames'])



# Apply the metrics calculation for each row
df[['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1 Score', 'FP Frames', 'FN Frames']] = df.apply(calculate_metrics,
                                                                                               axis=1)

# Save the merged dataframe to CSV
merged_df_csv_path = os.path.join(output_base_path, "00.csv")
df.to_csv(merged_df_csv_path, index=False)

# Video processing to create clips based on union frames
processing_times = []
for video_name in df['Video'].unique():
    start_time = time.time()
    video_path = os.path.join(video_directory, f"{video_name}.mp4")

    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        continue

    video_output_dir = os.path.join(output_base_path, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    union_frames = list(df[df['Video'] == video_name]['Union'].values[0])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create clips for each detected scene
    for i in range(len(union_frames) - 1):
        start_frame = union_frames[i]
        end_frame = union_frames[i + 1]
        clip_output_path = os.path.join(video_output_dir, f"{video_name}_clip_{i + 1}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

    # Handle the last segment
    last_start_frame = union_frames[-1]
    last_clip_output_path = os.path.join(video_output_dir, f"{video_name}_clip_{len(union_frames)}.mp4")
    out_last = cv2.VideoWriter(last_clip_output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_last.write(frame)

    out_last.release()
    cap.release()

    end_time = time.time()
    processing_time = end_time - start_time
    processing_times.append(processing_time)

# Summary DataFrame
summary_df = df[['Video', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1 Score',
                 'Detected Frames by PySceneDetect', 'Ground Truth Frames',
                 'Detected Frames by TransNetV2', 'Union', 'FP Frames', 'FN Frames']].copy()

summary_df.loc[:, 'Processing Time (seconds)'] = processing_times

# Save summary to CSV with comma-separated strings for FP and FN frames
summary_df['Union'] = summary_df['Union'].apply(lambda x: ', '.join(map(str, x)) if x else '0')
summary_df['FP Frames'] = summary_df['FP Frames'].apply(lambda x: ', '.join(map(str, x)) if x else '0')
summary_df['FN Frames'] = summary_df['FN Frames'].apply(lambda x: ', '.join(map(str, x)) if x else '0')

summary_csv_path = os.path.join(output_base_path, 'summary_fusion.csv')
summary_df.to_csv(summary_csv_path, index=False, float_format="%.2f")

print("Processing complete. Check output directory for video clips and summary saved in summary_fusion.csv")
