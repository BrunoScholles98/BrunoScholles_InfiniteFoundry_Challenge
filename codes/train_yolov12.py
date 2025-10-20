# ============================================================
# Treino YOLOv12n - Detecção de Mãos
# ============================================================

from pathlib import Path
import shutil
import random
import yaml
import os
from ultralytics import YOLO

# ============================================================
# VARIÁVEIS GLOBAIS
# ============================================================
BASE_DIR = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/challenge_hands/train")
PRETRAINED_WEIGHTS = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/YOLOv12_Baseline_Weights/yolov12n.pt")
OUTPUT_DIR = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands")

SEED = 42
IMG_SIZE = 640
EPOCHS = 50
BATCH = 16
CLASS_NAMES = ["hand"]
SPLIT_RATIOS = (0.75, 0.125, 0.125)
USE_SYMLINKS = True

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def discover_dirs(base_dir: Path):
    """Detecta automaticamente pastas de imagens e labels YOLO."""
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
        raise FileNotFoundError(f"Não foi possível localizar pasta 'labels' próxima de {base_dir}")
    return images_dir, labels_dir


def list_pairs(images_dir: Path, labels_dir: Path):
    """Retorna lista de (imagem, label) válidos."""
    exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])
    pairs = []
    for img in img_files:
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
    if not pairs:
        raise RuntimeError("Nenhuma imagem com anotação correspondente foi encontrada.")
    return pairs


def split_pairs(pairs, ratios, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    return pairs[:n_train], pairs[n_train:n_train+n_val], pairs[n_train+n_val:]


def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if USE_SYMLINKS:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def materialize_split(split, dst_images: Path, dst_labels: Path):
    for img, lab in split:
        safe_link_or_copy(img, dst_images / img.name)
        safe_link_or_copy(lab, dst_labels / lab.name)


def write_data_yaml(path: Path, train_dir: Path, val_dir: Path, test_dir: Path):
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


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def main():
    images_dir, labels_dir = discover_dirs(BASE_DIR)
    print(f"[OK] Imagens: {images_dir}")
    print(f"[OK] Labels : {labels_dir}")

    pairs = list_pairs(images_dir, labels_dir)
    print(f"[OK] Total de pares: {len(pairs)}")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, SPLIT_RATIOS, SEED)
    print(f"Split -> train={len(train_pairs)} | val={len(val_pairs)} | test={len(test_pairs)}")

    dataset_dir = OUTPUT_DIR / "dataset"
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    materialize_split(train_pairs, dataset_dir / "images/train", dataset_dir / "labels/train")
    materialize_split(val_pairs,   dataset_dir / "images/val",   dataset_dir / "labels/val")
    materialize_split(test_pairs,  dataset_dir / "images/test",  dataset_dir / "labels/test")

    data_yaml = OUTPUT_DIR / "hands_data.yaml"
    write_data_yaml(data_yaml,
                    dataset_dir / "images/train",
                    dataset_dir / "images/val",
                    dataset_dir / "images/test")
    print(f"[OK] YAML gerado em {data_yaml}")

    print(f"[OK] Carregando modelo YOLOv12 pré-treinado de {PRETRAINED_WEIGHTS}")
    model = YOLO(str(PRETRAINED_WEIGHTS))

    print(f"[OK] Iniciando treino...")
    results = model.train(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        project=str(OUTPUT_DIR),
        name="yolov12_hands_run",
        device=0,        # muda para 'cpu' se necessário
        pretrained=True,
        exist_ok=True,
        amp=False,  # 👈 desativa o check_amp problemático
        workers=0,     # 👈 diminui os subprocessos do DataLoader
    )

    print(f"\nTreino finalizado!")
    print(f"Pesos salvos em: {OUTPUT_DIR / 'yolov12_hands_run' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()