"""
===============================================================================
VIDEO FRAME EXTRACTION PIPELINE — ARCHITECTURE & LOGIC DOCUMENTATION
===============================================================================

Purpose
-------
This script scans a directory for .mp4 videos, extracts EVERY frame from each
video using OpenCV, saves the frames as image files, and generates a CSV file
mapping each frame number to its timestamp in the video timeline.

The script is intended for preprocessing tasks such as:
    - computer vision datasets
    - annotation pipelines
    - motion analysis
    - video indexing
    - ML training frame generation


Overall Workflow
----------------
main()
  └── process_videos()
        └── extract_frames()
              ├── read video metadata (fps, frame count)
              ├── iterate frame-by-frame
              ├── compute timestamp from frame index
              ├── save image
              └── store frame info → CSV summary

The program processes ALL .mp4 files located in the working directory.


===============================================================================
DIRECTORY STRUCTURE (AUTO-CREATED)
===============================================================================

Input folder:
    ./
      video1.mp4
      video2.mp4

Output folder created:
    ./output_ffmpeg/

Inside output_ffmpeg each video gets its own subfolder:

    ./output_ffmpeg/
        video1/
            0_000000.jpg
            1_000000.jpg
            2_000000.jpg
            ...
        video1_summary.csv

        video2/
            0_000000.jpg
            1_000000.jpg
            ...
        video2_summary.csv

Key idea:
Each video is isolated into its own folder so frame names never collide.


===============================================================================
HOW VIDEO FRAMES ARE READ
===============================================================================

OpenCV reads video as a continuous stream of images.

    video = cv2.VideoCapture(video_path)

Internally, a video is simply:
    a sequence of images shown at a fixed speed (FPS)

So a video timeline is:

frame 0 -> frame 1 -> frame 2 -> ... -> frame N

Each call to:
    success, frame = video.read()

returns:
    success = True if frame exists
    frame = numpy array (the image pixels)

Loop continues until success becomes False (end of video).


===============================================================================
WHAT IS FPS AND WHY IT MATTERS
===============================================================================

fps = Frames Per Second

Example:
    fps = 30
    → video shows 30 images every second

Meaning:
    frame 0   = 0.000 seconds
    frame 30  = 1.000 seconds
    frame 60  = 2.000 seconds

So time is not stored in the video per frame.
We COMPUTE time mathematically.


===============================================================================
HOW TIMESTAMP IS CALCULATED
===============================================================================

We compute time using:

    timestamp_seconds = frame_number / fps

Example:
    fps = 25
    frame_number = 50

    timestamp = 50 / 25 = 2 seconds

Then converted to human readable format:

    HH:MM:SS

using:
    timedelta(seconds=timestamp)

We then remove milliseconds for cleaner labeling.


===============================================================================
HOW FRAME FILENAMES ARE BUILT
===============================================================================

Filename format:
    {frame_number}_{timestamp}.jpg

Example:
    150_000006.jpg

Meaning:
    frame index = 150
    video time = 00:00:06

Colons (:) are removed because Windows/Linux filenames cannot safely contain them.


===============================================================================
HOW CSV SUMMARY IS CREATED
===============================================================================

We store per-frame metadata:

    Frame Number | Timestamp
    -------------------------
    0            | 00:00:00
    1            | 00:00:00
    2            | 00:00:00
    ...
    30           | 00:00:01

This allows:
    - mapping annotations back to original video
    - event reconstruction
    - temporal analysis


===============================================================================
KEY VARIABLES EXPLAINED
===============================================================================

video_path
    Full path to the video file being processed

fps
    Frames per second — determines time progression

total_frames
    Total number of frames in video (metadata only, not used for loop)

frame_num
    Current frame index counter (starts at 0, increments per read)

timestamp
    Computed time in seconds based on frame_num / fps

frame
    Actual image array (pixel matrix)

frame_data
    Python list collecting all frame metadata before writing CSV


===============================================================================
IMPORTANT DESIGN DECISIONS
===============================================================================

1) We iterate using video.read() instead of frame index seeking
   → ensures compatibility with variable codecs

2) We compute timestamps instead of reading them
   → many codecs do not store reliable per-frame timestamps

3) Every frame is saved
   → dataset completeness over storage efficiency

4) CSV summary stored outside frame folder
   → easier dataset scanning without image loading


===============================================================================
LIMITATIONS
===============================================================================

- Very large videos → large storage usage
- No frame skipping (extracts ALL frames)
- Only supports .mp4 files
- No parallel processing (single-threaded)


===============================================================================
END OF DOCUMENTATION
===============================================================================
"""

import os
import cv2
import pandas as pd
from datetime import timedelta


def create_output_directory(output_path):
    """
    Create an output directory if it doesn't already exist.

    Parameters:
    - output_path (str): The path to the output directory.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def extract_frames(video_path, output_folder):
    """
    Extract frames from a video file and save them as images.
    A CSV summary of frame numbers and timestamps will also be created.

    Parameters:
    - video_path (str): The path to the input video file.
    - output_folder (str): The folder where frames and summary will be saved.
    """
    # Open video using OpenCV
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Get frames per second
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count

    # Create a list to store frame data
    frame_data = []

    # Extract video filename and prepare save path
    video_filename = os.path.basename(video_path)
    video_basename, _ = os.path.splitext(video_filename)

    # Create a folder to save frames
    save_folder = os.path.join(output_folder, video_basename)
    create_output_directory(save_folder)

    frame_num = 0
    success, frame = video.read()

    while success:
        # Calculate timestamp in seconds and minutes:seconds format
        timestamp = frame_num / fps
        time_min_sec = str(timedelta(seconds=timestamp)).split('.')[0]  # mm:ss format

        # Construct frame filename using frame number and timestamp
        frame_filename = f"{frame_num}_{time_min_sec.replace(':', '')}.jpg"
        frame_path = os.path.join(save_folder, frame_filename)

        # Save the frame as an image
        cv2.imwrite(frame_path, frame)

        # Add frame data (frame number, timestamp)
        frame_data.append([frame_num, time_min_sec])

        # Read the next frame
        success, frame = video.read()
        frame_num += 1

    # Release the video capture object
    video.release()

    # Create a CSV file with the frame summary
    summary_csv = pd.DataFrame(frame_data, columns=["Frame Number", "Timestamp"])
    summary_csv_path = os.path.join(output_folder, f"{video_basename}_summary.csv")
    summary_csv.to_csv(summary_csv_path, index=False)


def process_videos(input_directory, output_directory):
    """
    Process all video files in the input directory by extracting frames and creating summaries.

    Parameters:
    - input_directory (str): The directory containing video files.
    - output_directory (str): The directory where frames and summaries will be saved.
    """
    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_directory, filename)
            print(f"Processing video: {video_path}")
            extract_frames(video_path, output_directory)


def main():
    """
    Main function to set input and output directories and process videos.
    """
    # Define input and output directories
    input_dir = r"./"
    output_dir = os.path.join(input_dir, "output_ffmpeg")

    # Create output directory if it doesn't exist
    create_output_directory(output_dir)

    # Process videos in the input directory
    process_videos(input_dir, output_dir)

    print("Frame extraction and CSV summary creation completed.")


# Run the script
if __name__ == "__main__":
    main()
