> **If you’re viewing this file via Google Drive, it’s strongly recommended to open the repository directly on GitHub for better display and navigation for this README file:**
> **[https://github.com/BrunoScholles98/BrunoScholles_InfiniteFoundry_Challenge](https://github.com/BrunoScholles98/BrunoScholles_InfiniteFoundry_Challenge)**

## Contents

1. [Repo & Environment Installation](#inst)
2. [Hand Detection Training (Optional)](#train)
3. [Inference Code (Main Code)](#inf)

# Bruno Scholles | Infinite Foundry Technical Challenge — Computer Vision & Robotics Engineer

This repository contains the full solution developed for the **Infinite Foundry Technical Challenge**. The code was executed on a **remote Linux (Ubuntu) server**, and the **inference pipeline outputs an annotated `.mp4` video** showing the detected hand operations and metrics dashboard. 

**All the commands shown here assume that you are using the Linux terminal in the root folder `/BrunoScholles_InfiniteFoundry_Challenge`**.

The repository is organized as follows:

* **`dataset_challenge_hands/`** – Contains the provided dataset and metadata for training.
* **`src/training/`** – Includes the training pipeline (`train_hand_detector.py`) and baseline YOLOv12 weights.
* **`src/inference/`** – Main inference logic. The core script `main.py` loads the trained model and processes the input video, producing the output video with detections and overlays.
* **`trained_models/`** – Stores the trained YOLOv12 model (`best.pt`) and configuration files.
* **`results/`** – Contains final processed videos (e.g., `output_video_detect_RTX3090.mp4`).

---

<a name="inst"></a>
## 1. Repo & Environment Installation

### 📦 Before cloning the repository (Git LFS)

> **If you downloaded this repository via Google Drive, you can skip the Git LFS step and go straight to Step 1.**

**This repository contains videos larger than 100 MB and therefore requires Git LFS**. If you don’t have Git LFS installed on your machine, run the following command in your terminal:

```bash
sudo apt install git-lfs
```

After that, enable Git LFS in your machine and clone the repository with the following two commands:

```bash
git lfs install
```

```bash
git clone https://github.com/BrunoScholles98/BrunoScholles_InfiniteFoundry_Challenge.git
```

### Step 1: Create and activate the conda environment

```bash
cd BrunoScholles_InfiniteFoundry_Challenge
```

```bash
conda env create -f environment.yml -n infinite
```

```bash
conda activate infinite
```

### Step 2: Verify CUDA installation (optional)

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA detected')"
```

Expected output (depending on your hardware): `True   NVIDIA GeForce RTX 4090`

Otherwise, if CUDA is not available: `False   No CUDA detected`

### Step 3: YOLOv12 Dependencies Installation

```bash
chmod +x install_yolov12.sh
```

```bash
./install_yolov12.sh
```

> ⚠️ **Important:**
> When you run this script (**`install_yolov12.sh`**), a folder named **`yolov12`** will be automatically created in the project directory.
> This folder contains essential dependencies and files required for the model to work properly. **Do not delete or move it under any circumstances**, as doing so will cause errors during training and inference.

---

<a name="train"></a>
## 2. Hand Detection Training (Optional)

The `train_hand_detector.py` script (located in the `training/` folder) automates the entire process. It automatically prepares the dataset (splitting it into train/validation/test sets) and starts the YOLOv12 training.

However, this step isn’t necessary, since I’ve already provided a trained model trained with the data provided (located in the /trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt).

Run it by passing the paths to the dataset, the pretrained weights, and the output directory.

**Arguments:**
- \--base\_dir | path to the dataset training folder (Default: `/challenge_hands/train`)
- \--weights | path to the pre-trained YOLOv12 weights (Default: `/training/YOLOv12_Baseline_Weights/yolov12n.pt`)
- \--output\_dir | path to your output trained model folder (Default: `trained_models/yolov12n_hands_new`)

**Usage Example:**

First, activate the environment:

```bash
conda activate infinite
```

To run training using the default paths (the script finds them automatically, as long as the files are in the expected locations):

```bash
python src/training/train_hand_detector.py
```

To specify the paths manually:

```bash
python src/training/train_hand_detector.py --base_dir /path/to/dataset/train --weights /path/to/yolov12n.pt --output_dir /path/to/trained_models/run_name
```
The best model will be saved in your output directory under `/weights/best.pt`.

> If no arguments are provided, the script will use the default directories. However, this is only recommended if you have not modified the original folder structure of the repository.
> If, for any reason, you encounter an error when running the scripts without arguments, it is strongly recommended to explicitly pass the required paths as arguments.

---

<a name="inf"></a>
## 3. Inference Code (Main Code)

This script loads the trained YOLOv12 model (provided in `trained_models/`) to detect hands in an input video. Based on the position and interaction of the detected hands, it applies specific detection logic (located in `src/detectors/`) to count four main operations:

1.  **Pick Piece:** When a hand crosses a horizontal reference line.
2.  **Place in Box:** Based on a heuristic for when only one hand is visible for a short period.
3.  **Pen-mark:** When both hands are close and stable for a time.
4.  **Probe-pass:** A state machine that tracks the specific interaction of the hand with the probe.

The script generates an output video (`.mp4`) with the detections drawn (bounding boxes, reference lines) and a real-time metrics panel.

**Arguments:**
- \--model\_path | Path to the trained .pt model file. (Default: `trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt`)
- \--input\_video | Path to the input video to be processed. (Default: `tarefas_cima.mp4`)
- \--output\_video | Path to save the resulting output video. (Default: `results/output_video_detections.mp4`)

**Usage Example:**

First, activate the environment:

```bash
conda activate infinite
```

To run inference using the default paths (the script finds them automatically, as long as the files are in the expected locations):

```bash
python src/inference/main.py
```

To specify the paths manually:

```bash
python src/inference/main.py --model_path path/to/hand/trained/model.pt --input_video path/to/trained/inference/video.mp4 --output_video path/to/output/video.mp4
```

The processed video will be saved at the specified output path (by default, in `results/output_video_detections.mp4`).

> If no arguments are provided, the script will use the default directories. However, this is only recommended if you have not modified the original folder structure of the repository.
> If, for any reason, you encounter an error when running the scripts without arguments, it is strongly recommended to explicitly pass the required paths as arguments.
