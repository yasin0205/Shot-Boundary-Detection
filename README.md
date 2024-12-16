
# Shot Boundary Detection in Soccer

## Abstract
The production of multimedia videos has surged thanks to affordable video equipment, large storage devices, and user-friendly editing tools. As video content proliferates, efficient analysis becomes increasingly critical to reducing production costs and manual labor. A fundamental step in video analysis is shot boundary detection (SBD), which segments videos into discrete units called shots. However, detecting shot boundaries poses significant challenges due to the diversity of transition types and frequent domain-specific variations. This study compares popular SBD algorithms within the soccer domain, revealing important insights. It was found that pixel-based approaches outperform the histogram-based approach in terms of precision, while deep learning models, such as TransNetV2, struggle with certain types of transitions, particularly logo transitions. Building on these findings, we developed a hybrid algorithm that combines pixel and deep learning approaches to improve performance. The proposed algorithm achieved significant performance improvements, with an average F1 score of 0.94 in the La Liga tournament and 0.88 in five other European tournaments. Key contributions of this work include the creation of a specialized soccer dataset, a comprehensive comparison of widely used SBD algorithms, and the development of an improved SBD algorithm tailored specifically for soccer videos.

---

## Pipeline Execution Steps

To execute the pipeline successfully, follow the steps below:

### 1. Clone the Repository
Clone the GitHub repository to your local machine.

### 2. Install Dependencies
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Extract Goal Clips
Run the script `01_goal_clip_extraction.py` to extract goal-related clips from the full-length video using the JSON file annotations:
```bash
python 01_goal_clip_extraction.py
```
- **Output:** Clips will be saved in the `dataset/00_goal_cut` folder.

### 4. Extract Frames
Decompose each goal clip into individual frames by running `01_frame_extraction.py`:
```bash
python 01_frame_extraction.py
```
- **Output:** Frames will be saved in the `output_ffmpeg` folder.

### 5. Shot Boundary Detection with TransNetV2
Perform initial shot boundary detection using `02_transnet_v2.py`:
```bash
python 02_transnet_v2.py
```
- **Output:** Detected boundaries, clips, and a CSV file will be saved in the `output_transnet_v2` folder.

### 6. Detect Shot Boundaries with Fine-Tuned AdaptiveDetector
Run the fine-tuned AdaptiveDetector algorithm:
```bash
python 03_adaptive_detector.py
```
- **To experiment with parameters:** Use `03_grid_search_parameters_adaptive_detector.py`.
- **For statistical analysis:** Use `03_statistical_analysis_adaptive_detector.ipynb`.

If running the grid search algorithm with SLURM, use the `03_grid_search_parameters_adaptive_detector.sbatch` file.
- **Output:** Clips and CSV files will be saved in the `output_pyscene` folder.
- Follow the same steps for HistogramDetector.

### 7. Fusion of Detection Results
Merge the outputs from TransNetV2 and AdaptiveDetector using `04_fusion.py`:
```bash
python 04_fusion.py
```
- **Output:** Final results, including clips and a CSV file, will be saved in the `output_fusion` folder.

---

## Contact
For further questions or inquiries, please contact:
**Mohammad Azizul Islam Yasin**  
Email: azizulislamyasin@gmail.com
```
