from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import torch
import random # Adicionado para seleção aleatória
from typing import Dict, Optional

# -------------------------
# CONSTANTS
# -------------------------
MODEL_PATH = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt")
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/tarefas_cima_Trim.mp4")

# Diretório de saída para as IMAGENS
OUTPUT_DIR = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/outupt_video/detected_hand_images_random")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5
NUM_RANDOM_FRAMES = 5 # Número de prints aleatórios a serem gerados


# -------------------------
# HELPERS
# -------------------------
def identify_hands(boxes: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """
    Identifica as mãos esquerda e direita com base em suas posições.
    """
    if boxes is None or len(boxes) == 0:
        return {"left": None, "right": None}
    
    boxes = np.asarray(boxes)
    
    if len(boxes) == 1:
        return {"left": None, "right": boxes[0].astype(int)}

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    idx = np.argsort(cx)[:2]
    left_box, right_box = boxes[idx[0]].astype(int), boxes[idx[1]].astype(int)
    return {"left": left_box, "right": right_box}

# -------------------------
# MAIN
# -------------------------
def main():
    print(f"[OK] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model.to(DEVICE)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total de frames no vídeo: {total_frames}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Diretório de saída criado/pronto em: {OUTPUT_DIR}")

    frames_to_process = []
    if total_frames > 0:
        # Seleciona N frames aleatórios, garantindo que não ultrapasse o número total de frames
        # e que não selecione o mesmo frame mais de uma vez.
        frames_to_process = random.sample(range(total_frames), min(NUM_RANDOM_FRAMES, total_frames))
        frames_to_process.sort() # Opcional: processar em ordem para melhor visualização do progresso
    
    print(f"[INFO] Processando {len(frames_to_process)} frames aleatórios: {frames_to_process}")

    processed_count = 0
    for frame_num in frames_to_process:
        # Pula para o frame aleatório
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        ok, frame = cap.read()
        if not ok:
            print(f"[WARNING] Não foi possível ler o frame {frame_num}. Pulando para o próximo.")
            continue

        # 1. Detectar mãos
        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
        hands = identify_hands(boxes)

        annotated = frame
        drew_hands = False # Flag para saber se desenhamos mãos neste frame

        # 2. Desenhar apenas as mãos
        if hands["left"] is not None:
            x1, y1, x2, y2 = hands["left"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(annotated, "hand_Left", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
            drew_hands = True

        if hands["right"] is not None:
            x1, y1, x2, y2 = hands["right"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 3)
            cv2.putText(annotated, "hand_Right", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3, cv2.LINE_AA)
            drew_hands = True

        # 3. Salvar o frame como imagem .jpg SE uma mão foi detectada
        if drew_hands:
            output_filename = OUTPUT_DIR / f"random_frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(output_filename), annotated)
            print(f"[INFO] Imagem salva: {output_filename}")
        else:
            print(f"[INFO] Nenhuma mão detectada no frame {frame_num}. Imagem não salva.")

        processed_count += 1
        print(f"[{processed_count}/{len(frames_to_process)}] Processado frame {frame_num}")


    cap.release()
    print(f"[✅] Processamento concluído. Imagens salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    main()