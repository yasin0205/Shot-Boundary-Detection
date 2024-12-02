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
