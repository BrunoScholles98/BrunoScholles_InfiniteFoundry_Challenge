from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import torch
import time
from typing import Dict, Optional, Tuple
from collections import deque # Importa o deque para o buffer

# -------------------------
# CONSTANTS
# -------------------------
MODEL_PATH = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt")
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/tarefas_cima_Trim.mp4")
OUTPUT_DIR = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/outupt_video/")
BASE_OUTPUT_NAME = "piece_in_box_frame"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5

# Número de frames de movimento para capturar (ANTES do alerta)
FRAMES_TO_CAPTURE_MOVEMENT = 3
# Espaço entre os frames de movimento (14 frames de espaço = pulo de 15)
FRAME_GAP = 14
BUFFER_SIZE = 60 # Buffer ajustado para guardar 3 * 15 = 45+ frames

# -------------------------
# HELPERS
# -------------------------
def count_piece_in_box(num_hands: int,
                       only_one_hand_frames: int,
                       cooldown_frames: int,
                       counter: int,
                       fps: float,
                       alert: Dict) -> Tuple[int, int, int, Dict]:
    """
    Detecta quando uma peça é colocada na caixa (transição de 2 mãos para 1 mão).
    """
    cooldown_duration = int(2 * fps)
    only_one_hand_frames = only_one_hand_frames + 1 if num_hands == 1 else 0
    cooldown_frames = max(0, cooldown_frames - 1)

    if only_one_hand_frames >= 14 and cooldown_frames == 0:
        counter += 1
        cooldown_frames = cooldown_duration
        only_one_hand_frames = 0
        alert["active"] = True
        alert["start_time"] = time.time()

    return counter, only_one_hand_frames, cooldown_frames, alert


def draw_piece_in_box_alert(frame: np.ndarray, alert: Dict) -> np.ndarray:
    """
    Desenha a mensagem "PIECE IN THE BOX" com fundo quando o alerta está ativo.
    """
    if not alert["active"]:
        return frame
    
    elapsed = time.time() - alert["start_time"]
    if elapsed > alert["duration"]:
        alert["active"] = False
        return frame

    h, _ = frame.shape[:2]
    arrow_h = 160
    color = (255, 0, 0) # Cor da seta e texto
    thickness = 25

    # Desenha a seta
    tip_y = h - 50
    base_y = tip_y - arrow_h
    base_x = 200
    tip = (base_x, tip_y)
    cv2.line(frame, (base_x, base_y), tip, color, thickness)
    pts = np.array([(base_x - 50, tip_y - 70), (base_x + 50, tip_y - 70), tip], np.int32)
    cv2.fillPoly(frame, [pts], color)
    
    # Adiciona fundo branco para o texto "PIECE IN THE BOX"
    text = "PIECE IN THE BOX"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    font_thickness = 5
    text_color = color
    bg_color = (255, 255, 255) # Branco

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = base_x - 100
    text_y = base_y - 40
    
    # Desenha o retângulo de fundo
    cv2.rectangle(frame, (text_x, text_y - text_height - baseline), 
                  (text_x + text_width, text_y + baseline), bg_color, -1)
    
    # Desenha o texto
    cv2.putText(frame, text, (text_x, text_y),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return frame


def identify_hands(boxes: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """
    Identifica as mãos esquerda e direita com base na posição X.
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

    fps = cap.get(cv2.CAP_PROP_FPS)

    in_box_count = 0
    only_one_hand_frames = 0
    cooldown_frames = 0
    alert = {"active": False, "start_time": 0.0, "duration": 1.0}

    # Variáveis para capturar frames de movimento
    alert_triggered = False
    
    # NOVO: Buffer para armazenar (frame, hands_dict)
    frame_buffer = deque(maxlen=BUFFER_SIZE)

    print("Processing video... looking for 'piece in box' event and preceding movement.")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Cria o diretório de saída se não existir

    # Calcula o tamanho mínimo necessário do buffer
    gap_indices = FRAME_GAP + 1
    required_buffer_size = gap_indices * FRAMES_TO_CAPTURE_MOVEMENT

    while True:
        ok, frame = cap.read()
        if not ok:
            if not alert_triggered:
                print("Video finished without finding the event.")
            else:
                # Este caso não deve mais acontecer, pois saímos após salvar
                print("Video finished.")
            break

        # Executa a detecção IMEDIATAMENTE
        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
        hands = identify_hands(boxes)

        # Filtra a mão direita
        if hands["right"] is not None:
            x1r, _, x2r, _ = hands["right"]
            if x2r < 850:
                hands["right"] = None

        # Adiciona o frame e as mãos detectadas ao buffer
        frame_buffer.append((frame.copy(), hands))

        num_hands = (hands["left"] is not None) + (hands["right"] is not None)

        if not alert_triggered: # Só detecta o alerta se ainda não foi disparado
            in_box_count, only_one_hand_frames, cooldown_frames, alert = count_piece_in_box(
                num_hands, only_one_hand_frames, cooldown_frames, in_box_count, fps, alert
            )

        # --- LÓGICA DE CAPTURA DE FRAMES ---
        if alert["active"] and not alert_triggered: # Se o alerta acabou de ser disparado
            
            # --- NOVO: VERIFICAÇÃO DO BUFFER ---
            # O buffer precisa ter frames suficientes para buscar T-45.
            if len(frame_buffer) <= required_buffer_size:
                print(f"[INFO] Event detected ({in_box_count})! Buffer too small ({len(frame_buffer)}/{required_buffer_size+1}). Waiting for buffer to fill...")
                # Ignora este evento e tenta no próximo frame
                alert["active"] = False # Desativa o alerta
                # A contagem (in_box_count) é mantida!
                cooldown_frames = 0     # Remove o cooldown para permitir detecção imediata
                only_one_hand_frames = 13 # Mantém o estado 'pronto para disparar' (13+1=14)
                continue # Pula para o próximo frame
            
            # --- Se o buffer estiver OK, continua para salvar ---
            alert_triggered = True
            
            # --- 1. Salva o Frame Principal (T=0) ---
            annotated_main_frame = frame.copy() # 'frame' é T=0

            # Desenha as mãos (de T=0)
            if hands["left"] is not None:
                x1, y1, x2, y2 = hands["left"]
                cv2.rectangle(annotated_main_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(annotated_main_frame, "hand_Left", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
            if hands["right"] is not None:
                x1, y1, x2, y2 = hands["right"]
                cv2.rectangle(annotated_main_frame, (x1, y1), (x2, y2), (255, 200, 0), 3)
                cv2.putText(annotated_main_frame, "hand_Right", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3, cv2.LINE_AA)

            # Adiciona o fundo branco para o texto do contador
            text_counter = f"Pieces in the box - {in_box_count}" # <-- VALOR NOVO
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 3
            text_color = (255, 0, 0) 
            bg_color = (255, 255, 255) 

            (text_width_c, text_height_c), baseline_c = cv2.getTextSize(text_counter, font, font_scale, font_thickness)
            text_x_c = 70
            text_y_c = 100
            
            cv2.rectangle(annotated_main_frame, (text_x_c, text_y_c - text_height_c - baseline_c), 
                          (text_x_c + text_width_c, text_y_c + baseline_c), bg_color, -1)
            cv2.putText(annotated_main_frame, text_counter, (text_x_c, text_y_c),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Desenha o alerta "PIECE IN THE BOX"
            annotated_main_frame = draw_piece_in_box_alert(annotated_main_frame, alert)

            # Salva o primeiro frame (com o alerta)
            output_path_main = OUTPUT_DIR / f"{BASE_OUTPUT_NAME}_main.jpg"
            cv2.imwrite(str(output_path_main), annotated_main_frame)
            print(f"\n[✅] Event detected! Main frame ({in_box_count}) saved at: {output_path_main}")
            
            # --- 2. Salva os Frames de Movimento (Anteriores) ---
            
            # Pulo de 15 = 14 frames de espaço (FRAME_GAP + 1)
            gap_indices = FRAME_GAP + 1 
            
            for i in range(FRAMES_TO_CAPTURE_MOVEMENT):
                # O índice do frame no buffer. 
                # T=0 é -1. T-15 é -16. T-30 é -31. T-45 é -46.
                mov_frame_idx = -1 - (gap_indices * (i + 1))
                
                # Pega os dados (frame, mãos) do buffer
                raw_movement_frame, hands_mov = frame_buffer[mov_frame_idx]
                
                # Anota o frame de movimento (APENAS as mãos)
                annotated_mov_frame = raw_movement_frame.copy()
                if hands_mov["left"] is not None:
                    x1, y1, x2, y2 = hands_mov["left"]
                    cv2.rectangle(annotated_mov_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(annotated_mov_frame, "hand_Left", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
                if hands_mov["right"] is not None:
                    x1, y1, x2, y2 = hands_mov["right"]
                    cv2.rectangle(annotated_mov_frame, (x1, y1), (x2, y2), (255, 200, 0), 3)
                    cv2.putText(annotated_mov_frame, "hand_Right", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3, cv2.LINE_AA)

                # --- INÍCIO DA CORREÇÃO ---
                # Adiciona o fundo branco para o texto do contador (nos frames de movimento)
                # Usa o 'in_box_count - 1' (o valor ANTES do evento)
                text_counter_mov = f"Pieces in the box - {in_box_count - 1}" # <-- VALOR ANTIGO
                font_mov = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_mov = 0.9
                font_thickness_mov = 3
                text_color_mov = (255, 0, 0) 
                bg_color_mov = (255, 255, 255) 

                (text_width_mov, text_height_mov), baseline_mov = cv2.getTextSize(text_counter_mov, font_mov, font_scale_mov, font_thickness_mov)
                text_x_mov = 70
                text_y_mov = 100
                
                cv2.rectangle(annotated_mov_frame, (text_x_mov, text_y_mov - text_height_mov - baseline_mov), 
                              (text_x_mov + text_width_mov, text_y_mov + baseline_mov), bg_color_mov, -1)
                cv2.putText(annotated_mov_frame, text_counter_mov, (text_x_mov, text_y_mov),
                            font_mov, font_scale_mov, text_color_mov, font_thickness_mov, cv2.LINE_AA)
                # --- FIM DA CORREÇÃO ---

                # Salva o frame de movimento (1 é o mais antigo, 3 é o mais próximo do evento)
                mov_frame_num = FRAMES_TO_CAPTURE_MOVEMENT - i
                output_path_movement = OUTPUT_DIR / f"{BASE_OUTPUT_NAME}_movement_before_{mov_frame_num}.jpg"
                cv2.imwrite(str(output_path_movement), annotated_mov_frame)
                print(f"[✅] Movement frame (before) {mov_frame_num} saved at: {output_path_movement}")

            # --- 3. Sai do Loop ---
            print("[INFO] All required frames saved. Exiting.")
            break # Sai do loop principal

    cap.release()
    print("[INFO] Process finished.")


if __name__ == "__main__":
    main()