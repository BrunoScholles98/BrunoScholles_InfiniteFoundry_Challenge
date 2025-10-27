"""
Usage example (in my case):
python train_hand_detector.py \
    --base_dir /mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/challenge_hands/train \
    --weights /mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/YOLOv12_Baseline_Weights/yolov12n.pt \
    --output_dir /mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands_new
"""

import os
import shutil
import random
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def discover_dirs(base_dir: Path):
    # Automatically detect YOLO-style image and label directories
    if (base_dir / "images").exists():
        images_dir = base_dir / "images"
    else:
        images_dir = base_dir

    candidates = [
        base_dir / "labels",
        base_dir.parent / "labels",
        base_dir.with_name("labels"),
    ]
    labels_dir = next((c for c in candidates if c.exists()), None)
    if labels_dir is None:
        raise FileNotFoundError(f"Could not locate 'labels' directory near {base_dir}")
    return images_dir, labels_dir


def list_pairs(images_dir: Path, labels_dir: Path):
    # Return all valid (image, label) pairs
    exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])
    pairs = []
    for img in img_files:
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
    if not pairs:
        raise RuntimeError("No matching image–label pairs were found.")
    return pairs


def split_pairs(pairs, ratios, seed=42):
    # Randomly split dataset into train, val, and test subsets
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def safe_link_or_copy(src: Path, dst: Path, use_symlinks=True):
    # Link files if possible; otherwise copy them safely
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlinks:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def materialize_split(split, dst_images: Path, dst_labels: Path, use_symlinks=True):
    # Physically create or link split directories for YOLO training
    for img, lab in split:
        safe_link_or_copy(img, dst_images / img.name, use_symlinks)
        safe_link_or_copy(lab, dst_labels / lab.name, use_symlinks)


def write_data_yaml(path: Path, train_dir: Path, val_dir: Path, test_dir: Path):
    # Generate a YOLOv12-compatible dataset YAML file
    data = {
        "train": str(train_dir),
        "val": str(val_dir),
        "test": str(test_dir),
        "names": {0: "hand"},
        "nc": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train a YOLOv12 hand detector.")
    parser.add_argument("--base_dir", type=str, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--weights", type=str, default=None, help="Path to the pretrained YOLOv12 weights.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory where the model and outputs will be saved.")
    args = parser.parse_args()

    # Determine script path to build fallback paths dynamically
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    # Default fallback dataset: one level above, under /challenge_hands/train
    default_dataset = script_dir.parent / "challenge_hands" / "train"

    # Default fallback weights: located under the script directory
    default_weights = script_dir / "YOLOv12_Baseline_Weights" / "yolov12n.pt"

    # Default fallback output: one level above this script
    default_output = script_dir.parent / "trained_models" / "yolov12n_hands_new"

    # Resolve paths safely with fallbacks
    base_dir = Path(args.base_dir) if args.base_dir and Path(args.base_dir).exists() else default_dataset
    weights_path = Path(args.weights) if args.weights and Path(args.weights).exists() else default_weights
    output_dir = Path(args.output_dir) if args.output_dir else default_output

    # Training hyperparameters
    seed = 42
    img_size = 640
    epochs = 50
    batch = 16
    split_ratios = (0.75, 0.125, 0.125)
    use_symlinks = True

    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA device detected: {gpu_name}")
    else:
        device = "cpu"
        print("[WARNING] No GPU detected. Using CPU for training (this will be slower).")

    # Print resolved paths
    print(f"[INFO] Using dataset base directory: {base_dir}")
    print(f"[INFO] Using pretrained weights: {weights_path}")
    print(f"[INFO] Model outputs will be saved to: {output_dir}")

    # Detect YOLO-style directories
    images_dir, labels_dir = discover_dirs(base_dir)
    print(f"[OK] Found images at: {images_dir}")
    print(f"[OK] Found labels at: {labels_dir}")

    # Pair images and labels
    pairs = list_pairs(images_dir, labels_dir)
    print(f"[OK] Total valid pairs: {len(pairs)}")

    # Dataset split
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, split_ratios, seed)
    print(f"[OK] Split summary: train={len(train_pairs)} | val={len(val_pairs)} | test={len(test_pairs)}")

    # Create dataset folder structure
    dataset_dir = output_dir / "dataset"
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    # Populate splits with symlinks or copies
    materialize_split(train_pairs, dataset_dir / "images/train", dataset_dir / "labels/train", use_symlinks)
    materialize_split(val_pairs,   dataset_dir / "images/val",   dataset_dir / "labels/val", use_symlinks)
    materialize_split(test_pairs,  dataset_dir / "images/test",  dataset_dir / "labels/test", use_symlinks)

    # Write the dataset YAML configuration
    data_yaml = output_dir / "hands_data.yaml"
    write_data_yaml(
        data_yaml,
        dataset_dir / "images/train",
        dataset_dir / "images/val",
        dataset_dir / "images/test"
    )
    print(f"[OK] Data configuration written to: {data_yaml}")

    # Load pretrained YOLO model (fallback if not provided)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    print(f"[OK] Loading pretrained YOLOv12 weights from {weights_path}")
    model = YOLO(str(weights_path))

    # Start training
    print(f"[OK] Starting training on device: {device}")
    results = model.train(
        data=str(data_yaml),
        imgsz=img_size,
        epochs=epochs,
        batch=batch,
        project=str(output_dir),
        name="yolov12_hands_run",
        device=device,    # Automatically chosen device (GPU or CPU)
        pretrained=True,
        exist_ok=True,
        amp=False,        # Disable AMP to avoid mixed precision issues
        workers=0,        # Reduce dataloader subprocesses for stability
    )

    print("\n[OK] Training completed successfully.")
    print(f"Weights saved at: {output_dir / 'yolov12_hands_run' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()