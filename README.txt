# Bruno Scholles | Infinite Foundry Technical Challenge — Computer Vision & Robotics Engineer

----------------------------------------------------------------

## 1. Environment Installation

### Step 1: Create and activate the conda environment

```bash
conda env create -f environment.yml -n infinite
conda activate infinite
```

### Step 2: Verify CUDA installation (optional)

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA detected')"
```

    Expected output (depending on your hardware):
    True
    NVIDIA GeForce RTX 4090

    If CUDA is not available:
    False
    No CUDA detected

### Step 3: YOLOv12 Dependencies Installation

```bash
chmod +x install_yolov12.sh
./install_yolov12.sh
```
----------------------------------------------------------------

## 2. Hand Detection Training (Optional)

The `train_hand_detector.py` script (located in the `training/` folder) automates the entire process.
It automatically prepares the dataset (splitting it into train/validation/test sets) and starts the YOLOv12 training.

However, this step isn’t necessary, since I’ve already provided a trained model trained with
the data provided (located in the /trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt).

Run it by passing the paths to the dataset, the pretrained weights, and the output directory.

Arguments:
--base_dir | path to the dataset training folder (already provided in the folder /training/YOLOv12_Baseline_Weights/yolov12n.pt)
--weights | path to the pre-trained YOLOv12 weights (already provided in the folder /challenge_hands/train)
--output_dir /path/to/trained_models/run_name | path to your trained model folder

Usage Example:
```bash
conda activate infinite
python training/train_hand_detector.py --base_dir /path/to/dataset/train --weights /path/to/yolov12n.pt --output_dir /path/to/trained_models/run_name
```

If no arguments are provided, the script will attempt to use default paths, however, this is not recommended.
Please provide the paths explicitly as arguments. The best model will be at your output directory, in /weights/best.pt

----------------------------------------------------------------

## 3. Inference Code (Main Code)