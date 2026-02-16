"""
===============================================================================
TRANSNETV2 SCENE DETECTION + CLIP GENERATION + METRIC EVALUATION PIPELINE
===============================================================================

Purpose
-------
This script performs AUTOMATIC SHOT BOUNDARY DETECTION on videos using the
deep learning model TransNetV2, splits the video into scene clips, and evaluates
detection accuracy against manually labeled ground-truth frames.

The pipeline does 3 major tasks:

    1) Detect scene cuts using a neural network (TransNetV2)
    2) Export each detected scene as a separate video clip
    3) Compare detected cuts with ground truth → compute Precision/Recall/F1

This is typically used in:
    - sports broadcast segmentation
    - highlight detection pipelines
    - video indexing
    - dataset preprocessing for action recognition
    - benchmarking shot boundary detectors


===============================================================================
WHAT TRANSNETV2 DOES (CORE IDEA)
===============================================================================

A video is not continuous visually — it consists of SHOTS:

    Shot A | Cut | Shot B | Cut | Shot C | ...

A "scene cut" (shot boundary) occurs when the camera view changes abruptly.

TransNetV2 is a convolutional neural network trained to predict:
    probability(frame i is a scene boundary)

The model outputs a probability for EACH frame:
    0 → not a cut
    1 → strong cut confidence

We convert these predictions into actual scene intervals.


===============================================================================
INPUT / OUTPUT STRUCTURE
===============================================================================

Input directory:
    ./
        A.mp4
        B.mp4
        C.mp4
        ...

Output directory:
    ./output_transnet_v2/
        A/
            clip_1.mp4
            clip_2.mp4
            clip_3.mp4
        B/
            clip_1.mp4
            clip_2.mp4
        summary_transnet_v2.csv

Each video gets its own folder containing separated scenes.


===============================================================================
MODEL OUTPUT EXPLANATION
===============================================================================

model.predict_video(video_path) returns:

    video_frames                → decoded frames (unused)
    single_frame_predictions    → probability of cut per frame
    all_frame_predictions       → internal smoothing predictions

We convert predictions into scenes using:

    scenes = model.predictions_to_scenes(predictions)

Output format:
    [(start_frame, end_frame), (start_frame, end_frame), ...]

Example:
    [(0, 150), (150, 340), (340, 512)]

Meaning:
    scene 1 = frames 0–149
    scene 2 = frames 150–339
    scene 3 = frames 340–511


===============================================================================
CLIP GENERATION LOGIC
===============================================================================

For each detected scene:

    1) Seek video to start_frame
    2) Read frames sequentially
    3) Write frames into new video file

OpenCV VideoWriter parameters:

    codec  = mp4v
    fps    = original video fps
    size   = original resolution

Result:
    Each scene becomes an independent playable video clip.


===============================================================================
GROUND TRUTH ANNOTATIONS
===============================================================================

ground_truth_frames dictionary contains manually labeled cut frames.

Example:
    "A": [0, 405, 451, 519, 569]

Meaning:
    Human labeled shot boundaries at these frame indices.

Video filename (without extension) is used as key:
    A.mp4 → key "A"


===============================================================================
HOW DETECTION IS EVALUATED
===============================================================================

A detected cut is considered CORRECT if it occurs close to a ground truth cut.

Tolerance window:
    ± margin frames (default = 25 frames)

Reason:
    Humans and models rarely agree on exact frame due to motion blur,
    encoding differences, and gradual transitions.

So instead of exact equality:
    abs(detected_frame - ground_truth_frame) <= margin


===============================================================================
METRICS COMPUTED
===============================================================================

Definitions:

TP (True Positive)
    Detected cut matches a real cut within margin

FP (False Positive)
    Model predicted a cut where none exists

FN (False Negative)
    Model missed a real cut


Precision
---------
    How many detected cuts are correct

        precision = TP / (TP + FP)


Recall
------
    How many real cuts were found

        recall = TP / (TP + FN)


F1 Score
--------
    Balanced performance metric

        F1 = 2 * (precision * recall) / (precision + recall)


===============================================================================
WHAT GETS SAVED TO CSV
===============================================================================

summary_transnet_v2.csv contains per-video report:

    Video name
    Number of detected scenes
    Detected frame indices
    Ground truth indices
    TP / FP / FN
    Precision / Recall / F1
    Incorrect detections (FP frames)
    Missed detections (FN frames)
    Processing time

This allows benchmarking multiple videos and comparing models.


===============================================================================
PROCESS FLOW
===============================================================================

for each video:
    load video
    run TransNetV2 inference
    convert predictions → scenes
    export scene clips
    compare with ground truth
    compute metrics
    store report

finally:
    save summary CSV


===============================================================================
IMPORTANT DESIGN DECISIONS
===============================================================================

Why deep learning instead of color histogram cuts?
    Sports videos have flashes, replays, camera motion → traditional methods fail

Why frame tolerance margin?
    Real shot boundary is not a single exact frame in practice

Why re-encode clips?
    Ensures compatibility and exact frame boundaries


===============================================================================
LIMITATIONS
===============================================================================

- Requires GPU for fast inference (CPU is slow)
- Gradual transitions may be partially inaccurate
- Needs correct ground truth naming (A, B, C...)
- Large videos consume RAM during prediction


===============================================================================
END OF DOCUMENTATION
===============================================================================
"""

import os
import time
import csv
import cv2
from transnetv2 import TransNetV2

# Load the TransNetV2 model
model = TransNetV2(model_dir="./transnetv2-weights/")

# Configuration
config = {
    "video_directory": "./",
    "output_directory": "./output_transnet_v2",
    "fps": 25,
    "margin": 25,
    "ground_truth_frames": {
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
    }
}


def calculate_metrics(detected_frames, ground_truth):
    margin = config['margin']
    TP = 0
    matched_detected_frames = set()
    fp_frames = set(detected_frames)
    fn_frames = set(ground_truth)

    for gt in ground_truth:
        for detected in detected_frames:
            if detected not in matched_detected_frames and abs(detected - gt) <= margin:
                TP += 1
                matched_detected_frames.add(detected)
                fp_frames.discard(detected)
                fn_frames.discard(gt)
                break

    FP = len(fp_frames)
    FN = len(fn_frames)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return TP, FP, FN, precision, recall, f1_score, sorted(fp_frames), sorted(fn_frames)


def process_videos():
    summary_report = []
    for video_file in os.listdir(config['video_directory']):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(config['video_directory'], video_file)
            print(f"Processing video: {video_file}")
            start_time = time.time()

            video_output_dir = os.path.join(config['output_directory'], os.path.splitext(video_file)[0])
            os.makedirs(video_output_dir, exist_ok=True)

            # Detect scenes
            video_frames, single_frame_predictions, _ = model.predict_video(video_path)
            scene_list = model.predictions_to_scenes(single_frame_predictions)

            # Create clips
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec

            for idx, (start_frame, end_frame) in enumerate(scene_list):
                clip_path = os.path.join(video_output_dir, f"clip_{idx + 1}.mp4")
                out = cv2.VideoWriter(clip_path, fourcc, fps, (frame_width, frame_height))

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for frame_num in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()
            cap.release()

            # Detected frames and metrics
            detected_frames = [scene[0] for scene in scene_list]
            video_name_key = os.path.splitext(video_file)[0]
            ground_truth = config['ground_truth_frames'].get(video_name_key, [])

            metrics = calculate_metrics(detected_frames, ground_truth)

            summary_report.append({
                'video_file': video_file,
                'total_detected_frames': len(scene_list),
                'detected_frames': detected_frames,
                'ground_truth_frames': ground_truth,
                'TP': metrics[0],
                'FP': metrics[1],
                'FN': metrics[2],
                'precision': metrics[3],
                'recall': metrics[4],
                'f1_score': metrics[5],
                'fp_frames': metrics[6],
                'fn_frames': metrics[7],
                'processing_time': time.time() - start_time
            })

    return summary_report


def save_summary_to_csv(summary_report):
    summary_file_path = os.path.join(config['output_directory'], 'summary_transnet_v2.csv')
    with open(summary_file_path, 'w', newline='', encoding='utf-8') as summary_file:
        csv_writer = csv.writer(summary_file)
        csv_writer.writerow(['Video File Name', 'Total Detected Frames', 'Detected Frames',
                             'Ground Truth Frames', 'TP', 'FP', 'FN',
                             'Precision', 'Recall', 'F1 Score', 'FP Frames', 'FN Frames', 'Processing Time'])
        for report in summary_report:
            csv_writer.writerow([
                report['video_file'],
                report['total_detected_frames'],
                ', '.join(map(str, report['detected_frames'])),
                ', '.join(map(str, report['ground_truth_frames'])),
                report['TP'],
                report['FP'],
                report['FN'],
                f"{report['precision']:.2f}",
                f"{report['recall']:.2f}",
                f"{report['f1_score']:.2f}",
                ', '.join(map(str, report['fp_frames'])),
                ', '.join(map(str, report['fn_frames'])),
                f"{report['processing_time']:.2f}"
            ])
    print(f"Summary report saved to: {summary_file_path}")


# Run the processing
if __name__ == "__main__":
    os.makedirs(config['output_directory'], exist_ok=True)
    summary_report = process_videos()
    save_summary_to_csv(summary_report)
