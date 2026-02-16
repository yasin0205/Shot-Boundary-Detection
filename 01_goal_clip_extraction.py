"""
===============================================================================
GOAL CLIP EXTRACTION PIPELINE — ARCHITECTURE & LOGIC DOCUMENTATION
===============================================================================

Purpose
-------
This script scans a football (soccer) dataset where each match folder contains:
    - 2 match videos (first half + second half)
    - 1 annotation JSON file

The script extracts short video clips around every "Goal" event using FFmpeg.

For each detected goal:
    Clip = 10 seconds BEFORE the goal + 70 seconds AFTER the goal

The final clips are saved into a centralized folder:
    ./00_goalcut/

This script is typically used for:
    - highlight generation
    - action recognition datasets
    - sports event detection ML training
    - broadcast indexing


===============================================================================
EXPECTED DATASET STRUCTURE
===============================================================================

Root dataset folder:
    ./dataset/
        Match_001/
            1_720p.mp4
            2_720p.mp4
            Labels-v2.json

        Match_002/
            1.mkv
            2.mkv
            Labels-v2.json

IMPORTANT ASSUMPTIONS:
----------------------
Each match folder MUST contain:
    exactly 2 videos → half 1 and half 2
    exactly 1 JSON annotation file

Video naming rule:
    filename contains "1"  → first half
    filename contains "2"  → second half

If structure is invalid → folder is skipped


===============================================================================
OUTPUT STRUCTURE
===============================================================================

All extracted clips go into a single folder:

    ./00_goalcut/
        Match_001_Goal_1_345.mp4
        Match_001_Goal_2_1890.mp4
        Match_002_Goal_1_120.mp4

Filename format:
    {MatchFolder}_Goal_{Half}_{TimestampInSeconds}.mp4

This ensures:
    - unique naming
    - searchable metadata
    - no overwriting between matches


===============================================================================
ANNOTATION FORMAT (JSON)
===============================================================================

Annotations contain events in this format:

{
    "annotations": [
        {
            "gameTime": "1 - 12:35",
            "label": "Goal"
        },
        {
            "gameTime": "2 - 44:10",
            "label": "Foul"
        }
    ]
}

gameTime meaning:
    "HalfNumber - MM:SS"

Example:
    "1 - 12:35"
        half = 1
        minute = 12
        second = 35


===============================================================================
TIME CONVERSION LOGIC
===============================================================================

We convert gameTime → seconds because FFmpeg requires seconds.

Formula:
    total_seconds = minutes * 60 + seconds

Example:
    "1 - 12:35"
    = 12*60 + 35
    = 755 seconds

This timestamp represents:
    position INSIDE THE HALF VIDEO (not full match time)


===============================================================================
CLIP WINDOW CALCULATION
===============================================================================

We want context around the goal event.

Chosen window:
    start_time = goal_time - 10 seconds
    end_time   = goal_time + 70 seconds

Why:
    - replay buildup before goal
    - celebration after goal

Edge protection:
    start_time = max(0, start_time)
    (prevents negative timestamps near video start)

Duration:
    duration = end_time - start_time


===============================================================================
VIDEO SELECTION LOGIC
===============================================================================

Each goal belongs to a specific half.

Example:
    "1 - 12:35" → use first half video
    "2 - 05:10" → use second half video

The script maps:
    half → correct video file

gameTime_videos = {
    1: first_half_video,
    2: second_half_video
}


===============================================================================
HOW CLIPS ARE EXTRACTED (FFMPEG)
===============================================================================

Command executed:

ffmpeg -ss START -i input.mp4 -t DURATION -c:v libx264 -c:a aac -y output.mp4

Parameter meaning:
------------------
-ss START
    Seek to start timestamp in seconds

-i input.mp4
    Input video file

-t DURATION
    Length of clip in seconds

-c:v libx264
    Re-encode video using H.264 (compatibility)

-c:a aac
    Re-encode audio using AAC

-y
    Overwrite existing files automatically


Why re-encoding instead of copying?
-----------------------------------
Ensures:
    - accurate seeking
    - frame alignment
    - consistent output format
    - avoids keyframe cut issues


===============================================================================
PROCESSING FLOW
===============================================================================

For each match folder:

1) Verify folder structure
2) Identify half 1 and half 2 videos
3) Load annotation JSON
4) Iterate annotations
5) If label == "Goal":
        convert time → seconds
        choose correct half video
        compute clip window
        run ffmpeg
6) Save clip


===============================================================================
ERROR HANDLING
===============================================================================

Handled cases:
    - invalid JSON format
    - wrong gameTime string
    - missing video
    - ffmpeg failure

Skipped safely without stopping full dataset processing.


===============================================================================
LIMITATIONS
===============================================================================

- Requires FFmpeg installed and available in PATH
- Relies on filename containing "1" and "2"
- Only processes "Goal" events
- Always re-encodes (slower but accurate)
- Single-threaded (no parallel processing)


===============================================================================
END OF DOCUMENTATION
===============================================================================
"""

import os
import json
import subprocess
# make sure you have label and video inside a folder tother
# Paths
dataset_path = r".\dataset"
output_path = r".\00_goalcut"

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)


# Helper function to convert "1 - MM:SS" to seconds
def game_time_to_seconds(game_time_str):
    try:
        half, time_str = game_time_str.split(" - ")
        minutes, seconds = map(int, time_str.split(":"))
        total_seconds = minutes * 60 + seconds
        return total_seconds
    except ValueError:
        print(f"Invalid gameTime format: {game_time_str}")
        return None


# Function to create goal clips using ffmpeg
def create_goal_clips(folder_path):
    # Skip the output folder if it gets processed by mistake
    if folder_path == output_path:
        return

    print(f"Processing folder: {folder_path}")

    # Get video files and JSON file
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv'))]
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    print(f"Found video files: {video_files}")
    print(f"Found JSON files: {json_files}")

    # Only proceed if we have two videos and one JSON file
    if len(video_files) != 2 or len(json_files) != 1:
        print(f"Skipping {folder_path} due to unexpected file structure.")
        return

    # Map gameTime values to their respective videos
    gameTime_videos = {}
    for video_file in video_files:
        if "1" in video_file:
            gameTime_videos[1] = os.path.join(folder_path, video_file)
        elif "2" in video_file:
            gameTime_videos[2] = os.path.join(folder_path, video_file)

    print(f"Mapped videos: {gameTime_videos}")

    # Read the JSON file
    json_path = os.path.join(folder_path, json_files[0])
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded annotations: {annotations}")

    # Iterate through annotations and create clips for "Goal" labels
    for annotation in annotations.get("annotations", []):
        print(f"Processing annotation: {annotation}")
        if annotation.get("label") == "Goal":
            game_time_str = annotation.get("gameTime")
            game_time = game_time_str.split(" - ")[0]  # Extract the half (1 or 2)
            timestamp = game_time_to_seconds(game_time_str)

            if timestamp is None:
                continue

            # Convert game_time to an integer
            try:
                game_time = int(game_time)
            except ValueError:
                print(f"Invalid gameTime value: {game_time}")
                continue

            if game_time in gameTime_videos:
                video_path = gameTime_videos[game_time]
                start_time = max(0, timestamp - 10)  # 10 seconds before the goal, but not less than 0
                end_time = timestamp + 70  # 70 seconds after the goal
                duration = end_time - start_time

                output_file = os.path.join(output_path,
                                           f"{os.path.basename(folder_path)}_Goal_{game_time}_{timestamp}.mp4")

                print(f"Creating clip from {start_time} to {end_time} for {video_path}")

                # ffmpeg command for cutting the clip
                command = [
                    "ffmpeg",
                    "-ss", str(start_time),
                    "-i", video_path,
                    "-t", str(duration),
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-y",  # Overwrite output files without asking
                    output_file
                ]

                # Run the command
                try:
                    subprocess.run(command, check=True)
                    print(f"Created clip: {output_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing video {video_path}: {e}")


# Iterate through each subfolder in the dataset path
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        create_goal_clips(folder_path)
