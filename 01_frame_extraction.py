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
