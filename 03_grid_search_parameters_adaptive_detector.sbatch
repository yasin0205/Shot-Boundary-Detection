#!/bin/bash
#SBATCH --job-name=pyscene_detect_parallel  # Job name
#SBATCH --partition=defq                    # Partition
#SBATCH --time=00:00                     # Runtime of 10 hours
#SBATCH --nodes=1                           # Use one node
#SBATCH --ntasks=1                          # Single task, we will manage parallelism in the script
#SBATCH --cpus-per-task=20                  # Request up to 20 CPUs (adjust based on your cluster)
#SBATCH --output=./Output_PySceneDetect/%j-%x-stdout.txt  # Standard output log
#SBATCH --error=./Output_PySceneDetect/%j-%x-stderr.txt   # Standard error log

# Activate the virtual environment
source /global/D1/homes/yasin/my-py310-venv-/bin/activate

# Define the array of video files
video_files=( ./A.mp4 ./B.mp4 ./C.mp4 ./D.mp4 ./E.mp4 ./F.mp4 ./G.mp4 ./H.mp4 ./I.mp4 ./J.mp4 )

# Process the videos with Python script
python ./03_grid_search_parameters_adaptive_detector.py

