
---

# Shot Boundary Detection in Soccer

## Abstract

The production of multimedia videos has surged due to affordable video equipment, large-capacity storage devices, and user-friendly editing tools. As video content proliferates, efficient video analysis has become increasingly critical for reducing production costs and manual labor. A fundamental step in video analysis is shot boundary detection (SBD), which segments videos into discrete units called *shots*. However, detecting shot boundaries remains challenging due to the diversity of transition types and domain-specific variations.

This study compares popular SBD algorithms within the soccer domain and reveals several important insights. Pixel-based approaches outperform histogram-based methods in terms of precision, while deep learning models such as TransNetV2 struggle with certain transitions, particularly logo transitions. Based on these findings, we propose a hybrid algorithm that combines pixel-based and deep learning approaches to improve performance. The proposed method achieves significant improvements, with an average F1 score of **0.94** on the La Liga tournament and **0.88** across five other European tournaments.

**Key contributions include:**

* A specialized soccer SBD dataset
* A comprehensive comparison of widely used SBD algorithms
* A hybrid SBD algorithm tailored for soccer broadcast videos

---

## Dataset Description

### `dataset/`

* Contains **full-length soccer match broadcasts**
* One folder per match
* Each match folder includes:

  * `1.mp4` → First half
  * `2.mp4` → Second half
  * `labels.json` → SoccerNet-style event annotations (goals, cards, etc.)

These annotations are used to automatically extract **goal-centered clips**.

### `Ground Truth/annotation.csv`

* A single CSV file containing **manually annotated shot boundaries**
* Used for **parameter optimization and evaluation**

---

## Repository Structure

```
Shot-Boundary-Detection/
├── dataset/                        # Full match videos + event annotations from SoccerNet
│   ├── match_1/
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   └── labels.json
│   └── ...
│
│   └── 00_goalcut/                 # Extracted goal-centered clips (MP4)
│
├── output_ffmpeg/                  # Extracted frames + timestamp CSVs
├── output_transnet_v2/             # TransNetV2 SBD outputs
├── output_adaptive_detector/       # Adaptive detector outputs
├── output_histogram_detector/      # Histogram detector outputs
├── output_fusion/                  # Final fused SBD results
│
├── Ground Truth/
│   └── annotation.csv              # Manual annotations
│
├── 01_goal_clip_extraction.py      # Event-based goal clip extraction
├── 01_frame_extraction.py          # Frame extraction
├── 02_transnet_v2.py               # SBD using TransNetV2
├── 03_adaptive_detector.py         # Pixel-based adaptive detector
├── 03_histogram_detector.py        # Histogram-based detector
├── 03_grid_search_parameters_adaptive_detector.py
│                                   # Parameter tuning for AdaptiveDetector
├── 03_grid_search_parameters_histogram_detector.py
│                                   # Parameter tuning for HistogramDetector
├── 03_statistical_analysis_adaptive_detector.ipynb
│                                   # Quantitative evaluation and plots
├── 03_statistical_analysis_histogram_detector.ipynb
│                                   # Quantitative evaluation and plots
├── 04_fusion.py                    # Fusion of TransNetV2 and AdaptiveDetector
│
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Pipeline Execution Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yasin0205/Shot-Boundary-Detection.git
cd Shot-Boundary-Detection
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Notes:**

* Python **3.7** is required.

---

### 3. Extract Goal Clips

```bash
python 01_goal_clip_extraction.py
```

---

### 4. Extract Frames

```bash
python 01_frame_extraction.py
```

---

### 5. Shot Boundary Detection Using TransNetV2

```bash
python 02_transnet_v2.py
```

**If you encounter inference issues:**

Download the pretrained weights from:
[https://github.com/soCzech/TransNetV2/tree/master/inference/transnetv2-weights](https://github.com/soCzech/TransNetV2/tree/master/inference/transnetv2-weights)

Place them in the `transnetv2-weights/` directory.

---

### 6. Pixel-Based Shot Boundary Detection

#### AdaptiveDetector

```bash
python 03_adaptive_detector.py
```

**Optional parameter tuning:**

```bash
python 03_grid_search_parameters_adaptive_detector.py
```

If running the grid search with **SLURM**, use:

```
03_grid_search_parameters_adaptive_detector.sbatch
```

* Statistical analysis is provided in:

  ```
  03_statistical_analysis_adaptive_detector.ipynb
  ```

---

#### HistogramDetector

```bash
python 03_histogram_detector.py
```

**Optional parameter tuning:**

```bash
python 03_grid_search_parameters_histogram_detector.py
```

If running the grid search with **SLURM**, use:

```
03_grid_search_parameters_histogram_detector.sbatch
```

* Statistical analysis is provided in:

  ```
  03_statistical_analysis_histogram_detector.ipynb
  ```

---

### 7. Fusion of Detection Results

```bash
python 04_fusion.py
```

**This step:**

* Combines TransNetV2 and AdaptiveDetector outputs into a final fused prediction set

---

## Contact

**Mohammad Azizul Islam Yasin**
📧 Email: [azizulislamyasin@gmail.com](mailto:azizulislamyasin@gmail.com)

---
