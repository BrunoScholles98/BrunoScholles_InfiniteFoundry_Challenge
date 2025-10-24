from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import deque
from typing import Dict, Optional, Tuple

# -------------------------
# CONSTANTS
# -------------------------
MODEL_PATH = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt")
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/tarefas_cima_Trim.mp4")
# Saída será um vídeo
OUTPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/outupt_video/pick_tracker_video.mp4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5
PICK_LINE_Y = 715

# -------------------------
# HELPERS
# -------------------------

def identify_hands(boxes: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """
    Identifica as mãos esquerda e direita com base na posição X.
    (Lógica do seu código original)
    """
    if boxes is None or len(boxes) == 0:
        return {"left": None, "right": None}
    boxes = np.asarray(boxes)
    if len(boxes) == 1:
        # Se só há uma mão, é a direita (e a esquerda é None)
        return {"left": None, "right": boxes[0].astype(int)}

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    idx = np.argsort(cx)[:2]
    left_box, right_box = boxes[idx[0]].astype(int), boxes[idx[1]].astype(int)
    return {"left": left_box, "right": right_box}

def update_pick_state(hands: Dict[str, Optional[np.ndarray]], 
                        picked_count: int, 
                        above_prev: bool) -> Tuple[int, bool, Tuple[int, int, int]]:
    """
    Verifica se a mão ESQUERDA cruzou a linha e atualiza o contador e a cor.
    (Baseado na sua função count_pieces_picked)
    """
    above_now = above_prev
    line_color = (255, 105, 180) # Cor rosa (abaixo) por defeito
    
    hand_to_track = hands.get("left") # Rastreia a mão ESQUERDA

    if hand_to_track is not None:
        x1, y1, x2, y2 = hand_to_track
        y_centroid = (y1 + y2) // 2
        above_now = y_centroid < PICK_LINE_Y
        
        line_color = (170, 255, 200) if above_now else (255, 105, 180) # Verde / Rosa
        
        # Evento de contagem: Estava acima (True) e agora NÃO está acima (False)
        if above_prev and not above_now:
            picked_count += 1
            
    # Se a mão esquerda não for detectada, 'above_now' mantém o valor anterior
    # e 'line_color' é definida com base nesse estado
    else:
        line_color = (170, 255, 200) if above_now else (255, 105, 180)

    return picked_count, above_now, line_color

def draw_annotations(frame: np.ndarray, 
                     hands: Dict[str, Optional[np.ndarray]], 
                     line_y: int, 
                     line_color: Tuple[int, int, int], 
                     picked_count: int) -> np.ndarray:
    """
    Desenha todas as anotações no frame (mãos, linha, contador).
    """
    annotated = frame.copy()
    width = frame.shape[1]

    # 1. Desenha Mãos
    if hands.get("left") is not None:
        x1, y1, x2, y2 = hands["left"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(annotated, "hand_Left", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
    
    if hands.get("right") is not None:
        x1, y1, x2, y2 = hands["right"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 3)
        cv2.putText(annotated, "hand_Right", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3, cv2.LINE_AA)

    # 2. Desenha Linha
    cv2.line(annotated, (0, line_y), (width, line_y), line_color, 3)
    cv2.putText(annotated, "pick_line", (50, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_color, 3, cv2.LINE_AA)

    # 3. Desenha Contador
    text_counter = f"Pieces Picked - {picked_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 3
    text_color = (255, 105, 180) # Cor rosa original
    bg_color = (255, 255, 255) # Branco

    (text_width_c, text_height_c), baseline_c = cv2.getTextSize(text_counter, font, font_scale, font_thickness)
    text_x_c = 70
    text_y_c = 100
    
    cv2.rectangle(annotated, (text_x_c - 10, text_y_c - text_height_c - baseline_c - 5), 
                  (text_x_c + text_width_c + 10, text_y_c + baseline_c + 5), bg_color, -1)
    cv2.putText(annotated, text_counter, (text_x_c, text_y_c),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return annotated

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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configura o VideoWriter para salvar o vídeo
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
    
    print(f"Gerando vídeo em: {OUTPUT_VIDEO}")

    # Estado inicial apenas para o "pick tracker"
    picked_count = 0
    above_prev = False # Assume que começa abaixo da linha (ou fora)
    line_color = (255, 105, 180) # Cor rosa (abaixo)

    progress = tqdm(total=total, desc="Processing video", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1. Detecção
        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
        hands = identify_hands(boxes)

        # 2. Filtra a mão direita (como no seu código original)
        if hands["right"] is not None:
            x1r, _, x2r, _ = hands["right"]
            if x2r < 850:
                hands["right"] = None
        
        # 3. Atualiza o estado (contador e cor da linha)
        picked_count, above_prev, line_color = update_pick_state(hands, picked_count, above_prev)

        # 4. Desenha as anotações
        annotated = draw_annotations(frame, hands, PICK_LINE_Y, line_color, picked_count)

        # 5. Escreve o frame no vídeo de saída
        out.write(annotated)
        progress.update(1)

    progress.close()
    cap.release()
    out.release()
    print(f"\n[✅] Vídeo salvo em: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()