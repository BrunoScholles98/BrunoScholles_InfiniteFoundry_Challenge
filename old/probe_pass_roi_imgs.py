from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import deque
from typing import Dict, Optional, Tuple, List, Any

# -------------------------
# CONSTANTS
# -------------------------
MODEL_PATH = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt")
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/tarefas_cima_Trim.mp4")
OUTPUT_FOLDER = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/outupt_video/output_probe_images")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5


# -------------------------
# HELPERS
# -------------------------
def identify_hands(boxes: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    """
    Identifica as mãos esquerda e direita com base na sua posição horizontal.
    """
    if boxes is None or len(boxes) == 0:
        return {"left": None, "right": None}
    
    boxes = np.asarray(boxes)
    
    if len(boxes) == 1:
        # Assume ser a mão direita se apenas uma estiver presente (baseado no fluxo da tarefa)
        return {"left": None, "right": boxes[0].astype(int)}

    # Ordena as caixas pela coordenada X do centro
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    idx = np.argsort(cx)[:2]  # Pega as duas mãos mais à esquerda
    left_box, right_box = boxes[idx[0]].astype(int), boxes[idx[1]].astype(int)
    
    return {"left": left_box, "right": right_box}


# -------------------------
# PROBE PASS DETECTOR (MODIFICADO para ROI RAW)
# -------------------------
def detect_probe_pass(hands: Dict[str, Optional[np.ndarray]], 
                      img_annotated: np.ndarray, # Frame para anotação visual
                      img_raw: np.ndarray,       # Frame ORIGINAL e CRU para ROI
                      state: Dict[str, Any],
                      output_folder: Path,
                      current_det_count: int
                     ) -> Tuple[Dict[str, Any], np.ndarray, int]:
    """
    Detecta a operação "probe pass" e salva os frames do ROI (original CRU e movimento)
    em uma pasta específica no momento da detecção.
    """
    now = time.time()
    left, right = hands["left"], hands["right"]

    # Inicializa chaves de estado se não existirem
    state.setdefault("delay_frames", 0)
    state.setdefault("prev_raw_crop", None) # <-- MODIFICADO: prev_raw_crop agora
    state.setdefault("phase0_green_frames", 0)
    state.setdefault("roi_green_until", 0.0)
    state.setdefault("hand_phase1_green_until", 0.0)
    state.setdefault("roi_buffer", []) # Buffer para frames do ROI (raw e motion)

    if now < state.get("cooldown_until", 0):
        return state, img_annotated, current_det_count

    # --- Lógica de Reset ---
    if left is None or right is None:
        state["phase"] = 0
        state["monitoring"] = False
        state["delay_frames"] = 0
        state["prev_raw_crop"] = None # <-- MODIFICADO
        state["phase0_green_frames"] = 0
        state["roi_green_until"] = 0.0
        state["roi_buffer"] = []
        return state, img_annotated, current_det_count

    # Coordenadas e distâncias
    x1l, y1l, x2l, y2l = left
    x1r, y1r, x2r, y2r = right
    c_left = np.array([(x1l + x2l)//2, (y1l + y2l)//2])
    c_right = np.array([(x1r + x2r)//2, (y1r + y2r)//2])
    v_tl_right = np.array([x1r, y1r]) 
    centers_dist = float(np.linalg.norm(c_left - c_right))

    # Inicia o monitoramento
    if centers_dist <= 500 and not state.get("monitoring", False):
        state["monitoring"] = True
        state["phase"] = 0
        state["delay_frames"] = 0
        state["prev_raw_crop"] = None # <-- MODIFICADO
        state["roi_buffer"] = []

    # Aborta se as mãos se afastarem durante a Fase 0
    if centers_dist > 500 and state.get("phase", 0) == 0:
        state["monitoring"] = False
        state["phase"] = 0
        state["delay_frames"] = 0
        state["prev_raw_crop"] = None # <-- MODIFICADO
        state["phase0_green_frames"] = 0
        state["roi_green_until"] = 0.0
        state["roi_buffer"] = []
        return state, img_annotated, current_det_count

    # Desenha linhas de "conexão" no frame ANOTADO
    if state.get("monitoring", False) and state.get("phase", 0) in (0, 1):
        edge_points = [
            np.array([x2l, int(y1l + frac * (y2l - y1l))])
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for pt in edge_points:
            cv2.line(img_annotated, tuple(v_tl_right), tuple(pt), (0, 255, 100), 2)

    # ----- FASE 0: Mãos se alinhando -----
    if state["phase"] == 0 and state.get("monitoring", False):
        phase0_hand_should_be_red = True
        dx = max(0, x1l - v_tl_right[0], v_tl_right[0] - x2l)
        dy = max(0, y1l - v_tl_right[1], v_tl_right[1] - y2l)
        dist_to_left_box = float(np.hypot(dx, dy))

        if dist_to_left_box <= 15:
            state["count"] += 1
            state["phase"] = 1
            state["delay_frames"] = 0
            state["prev_raw_crop"] = None # <-- MODIFICADO
            state["last_detection_time"] = now
            state["phase0_green_frames"] = 14
            state["roi_buffer"] = [] 
            cv2.putText(img_annotated, "Probe Pass +1 (phase0)", (700, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

        if state["phase"] == 0 and phase0_hand_should_be_red and right is not None:
            cv2.rectangle(img_annotated, (x1r, y1r), (x2r, y2r), (0, 0, 255), 5) # Mão vermelha no frame anotado

    # ----- FASE 1: Monitorando o ROI para inserção -----
    if state["phase"] == 1 and state.get("monitoring", False):

        # --- Condição de Aborto (Fase 1) ---
        dist_x = x1r - x2l
        if dist_x > 150:
            state["phase"] = 0
            state["monitoring"] = False 
            state["delay_frames"] = 0
            state["prev_raw_crop"] = None # <-- MODIFICADO
            state["phase0_green_frames"] = 0
            state["roi_green_until"] = 0.0
            state["roi_buffer"] = []
            return state, img_annotated, current_det_count

        state["delay_frames"] += 1

        # Destaque visual verde pós-Fase 0 no frame ANOTADO
        if state["phase0_green_frames"] > 0 and right is not None:
            cv2.rectangle(img_annotated, (x1r, y1r), (x2r, y2r), (0, 255, 0), 5)
            state["phase0_green_frames"] -= 1
        elif right is not None:
            cv2.rectangle(img_annotated, (x1r, y1r), (x2r, y2r), (0, 0, 255), 5)

        # --- Monitoramento do ROI (só começa após 14 frames de delay) ---
        if state["delay_frames"] > 14:
            roi_size = 80
            cx, cy = int(v_tl_right[0]), int(v_tl_right[1])
            x1c = max(cx - roi_size//2, 0)
            y1c = max(cy - roi_size//2, 0)
            x2c = min(cx + roi_size//2, img_annotated.shape[1])
            y2c = min(cy + roi_size//2, img_annotated.shape[0])
            
            # Garante que o crop não está vazio
            if y1c < y2c and x1c < x2c:
                # --- MODIFICADO: Crop do ROI é feito no img_raw ---
                current_raw_crop = img_raw[y1c:y2c, x1c:x2c].copy()
            else:
                current_raw_crop = None

            # Destaque visual do ROI no frame ANOTADO
            roi_is_green = now <= state["roi_green_until"]
            roi_color = (0, 255, 0) if roi_is_green else (0, 0, 255)
            cv2.rectangle(img_annotated, (x1c, y1c), (x2c, y2c), roi_color, 2)
            cv2.putText(img_annotated, "roi", (x1c, max(0, y1c - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2, cv2.LINE_AA)

            # Calcula movimento usando os crops RAW
            if state["prev_raw_crop"] is not None and current_raw_crop is not None and \
               current_raw_crop.size > 0 and state["prev_raw_crop"].shape == current_raw_crop.shape:
                
                diff = cv2.absdiff(current_raw_crop, state["prev_raw_crop"])
                gray_motion = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                motion_value = np.sum(gray_motion) / (gray_motion.shape[0] * gray_motion.shape[1])

                # Adiciona ao buffer ANTES da detecção
                # Salva o frame original do ROI (RAW) e o frame de movimento (subtração)
                state["roi_buffer"].append((current_raw_crop.copy(), gray_motion.copy()))

                # --- Detecção de Movimento (Inserção) ---
                if motion_value > 25:
                    
                    # --- Lógica de Salvamento ---
                    current_det_count += 1
                    save_path = output_folder / f"det_{current_det_count:03d}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    tqdm.write(f"\n[PROBE PASS DETECTED #{current_det_count}] Saving {len(state['roi_buffer'])} frames to {save_path}")
                    try:
                        for i, (orig_img, motion_img) in enumerate(state["roi_buffer"]):
                            orig_filename = save_path / f"orig_{i:04d}.png"
                            motion_filename = save_path / f"motion_{i:04d}.png"
                            cv2.imwrite(str(orig_filename), orig_img)
                            cv2.imwrite(str(motion_filename), motion_img)
                    except Exception as e:
                        tqdm.write(f"\n[ERROR] Failed to save images for det_{current_det_count:03d}: {e}")
                    
                    # --- Fim da Lógica de Salvamento ---

                    # Reseta o estado para a próxima detecção
                    state["count"] += 1
                    state["phase"] = 2 # Entra em cooldown
                    state["cooldown_until"] = now + 2.0
                    state["monitoring"] = False
                    state["delay_frames"] = 0
                    state["prev_raw_crop"] = None # <-- MODIFICADO
                    state["last_detection_time"] = now
                    state["hand_phase1_green_until"] = now + 1.0
                    state["roi_green_until"] = now + 1.0
                    state["roi_buffer"] = []
                    
                    cv2.putText(img_annotated, "Probe Pass +1 (phase1 - insertion)", (700, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

            if current_raw_crop is not None:
                state["prev_raw_crop"] = current_raw_crop.copy() # Armazena o frame RAW atual
            else:
                state["prev_raw_crop"] = None

    # ----- Destaque Verde Pós-Detecção (no frame ANOTADO) -----
    if (now - state.get("last_detection_time", 0)) <= 1.0 and right is not None:
        x1r, y1r, x2r, y2r = right
        cv2.rectangle(img_annotated, (x1r, y1r), (x2r, y2r), (0, 255, 0), 5)
        cv2.putText(img_annotated, "hand_Right_probe_pass", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

    return state, img_annotated, current_det_count


# -------------------------
# MAIN (MODIFICADO)
# -------------------------
def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model.to(DEVICE)

    print(f"[INFO] Opening video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving detection images to: {OUTPUT_FOLDER}")
    
    global_detection_count = 0

    probe_state = {
        "count": 0,
        "monitoring": False,
        "phase": 0,
        "followup_active": False,
        "last_detection_time": 0.0,
        "cooldown_until": 0.0,
        "phase0_green_frames": 0,
        "roi_green_until": 0.0,
        "hand_phase1_green_until": 0.0,
        "delay_frames": 0,
        "prev_raw_crop": None, # <-- MODIFICADO
        "roi_buffer": [],
    }

    progress = tqdm(total=total, desc="Processing video", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Cria uma cópia do frame original para anotação, mantendo o 'frame' puro para o ROI
        annotated_frame = frame.copy() 

        # Roda a inferência do modelo
        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
        hands = identify_hands(boxes)

        if hands["right"] is not None:
            x1r, _, x2r, _ = hands["right"]
            if x2r < 850:
                hands["right"] = None

        # Desenha caixas base das mãos no frame ANOTADO
        if hands["left"] is not None:
            x1, y1, x2, y2 = hands["left"]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(annotated_frame, "hand_Left", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)

        if hands["right"] is not None:
            x1, y1, x2, y2 = hands["right"]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 200, 0), 3)
            cv2.putText(annotated_frame, "hand_Right", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3, cv2.LINE_AA)

        # Roda a detecção do Probe Pass
        # Agora passa tanto o frame anotado (para desenho) quanto o frame original (para ROI)
        probe_state, annotated_frame, global_detection_count = detect_probe_pass(
            hands, annotated_frame, frame, probe_state, OUTPUT_FOLDER, global_detection_count
        )

        # Painel de Contagem (HUD) no frame ANOTADO
        cv2.rectangle(annotated_frame, (50, 50), (500, 130), (255, 255, 255), -1)
        cv2.putText(annotated_frame, f"Probe Passes - {probe_state['count']}", (70, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 3, cv2.LINE_AA)
        
        progress.update(1)

    progress.close()
    cap.release()
    
    print(f"\n[✅] Processamento concluído.")
    print(f"[INFO] Total de {global_detection_count} detecções de 'Probe Pass' salvas em: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()