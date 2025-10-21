from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import torch
import time
from tqdm import tqdm

# -------------------------
# VARIÁVEIS GLOBAIS
# -------------------------
MODEL_PATH = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt")
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/tarefas_cima.mp4")
OUTPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/outupt_video/output_video_detected.mp4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5


# -------------------------
# FUNÇÕES AUXILIARES
# -------------------------
def drop_piece_inside_box(num_maos, frames_apenas_uma_mao, cooldown_frames, contador, fps):
    cooldown_duracao = int(2 * fps)

    if num_maos == 1:
        frames_apenas_uma_mao += 1
    else:
        frames_apenas_uma_mao = 0

    if cooldown_frames > 0:
        cooldown_frames -= 1

    if frames_apenas_uma_mao >= 14 and cooldown_frames == 0:
        contador += 1
        cooldown_frames = cooldown_duracao
        frames_apenas_uma_mao = 0

    return contador, frames_apenas_uma_mao, cooldown_frames


def identificar_maos(boxes):
    if len(boxes) == 0:
        return {"left": None, "right": None}
    elif len(boxes) == 1:
        return {"left": None, "right": boxes[0]}
    else:
        boxes_sorted = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
        return {"left": boxes_sorted[0], "right": boxes_sorted[1]}


def count_pieces_picked(maos, contador_picked, acima_da_linha_anterior, linha_cor):
    """
    Controla o contador 'Pieces Picked' e a cor da linha 'pile_line'.
    """
    pile_line_y = 715
    acima_da_linha_atual = False
    rosa = (255, 105, 180)
    verde_menta = (170, 255, 200)

    if maos["left"] is not None:
        x1, y1, x2, y2 = map(int, maos["left"])
        y_centroid = (y1 + y2) // 2

        acima_da_linha_atual = y_centroid < pile_line_y

        if acima_da_linha_atual:
            linha_cor = verde_menta
        else:
            linha_cor = rosa

        if acima_da_linha_anterior and not acima_da_linha_atual:
            contador_picked += 1
    else:
        acima_da_linha_atual = acima_da_linha_anterior

    return contador_picked, acima_da_linha_atual, linha_cor


# -------------------------
# DETECÇÃO DE PEN MARK
# -------------------------
def detect_pen_mark(maos, annotated, penmark_data):
    """
    Detecta o gesto de marcação com a caneta (Pen Mark).
    Quando detectado, muda a cor da bounding da mão direita por 1s
    e exibe o texto 'hand_Right_pen_scratch'.
    """

    # --- parâmetros fixos ---
    DIST_THRESHOLD = 500
    WINDOW_SIZE = 60
    MIN_SIGN_CHANGES = 2
    MAX_SIGN_CHANGES = 6
    AMP_THRESHOLD = 120
    LEFT_STABLE_STD = 10
    COOLDOWN_TIME = 1.0
    HIGHLIGHT_TIME = 1.0  # tempo da cor laranja após detecção

    # --- helpers internos ---
    def centroid(box):
        x1, y1, x2, y2 = map(int, box)
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def mao_esquerda_parada(hist_esq):
        if len(hist_esq) < 5:
            return False
        hist_esq = np.array(hist_esq)
        desvios = np.std(hist_esq, axis=0)
        return np.mean(desvios) < LEFT_STABLE_STD

    def detectar_osc_ilatoria(dist_history):
        if len(dist_history) < 10:
            return False
        diffs = np.diff(dist_history)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        amplitude = np.ptp(dist_history)
        if MIN_SIGN_CHANGES <= sign_changes <= MAX_SIGN_CHANGES and amplitude <= AMP_THRESHOLD:
            return True
        return False

    # --- lógica principal ---
    current_time = time.time()

    if maos["left"] is not None and maos["right"] is not None:
        # centroides
        x1l, y1l, x2l, y2l = map(int, maos["left"])
        x1r, y1r, x2r, y2r = map(int, maos["right"])
        centro_esq = ((x1l + x2l) // 2, (y1l + y2l) // 2)
        centro_dir = ((x1r + x2r) // 2, (y1r + y2r) // 2)

        # distância
        dist = np.linalg.norm(np.array(centro_esq) - np.array(centro_dir))
        penmark_data["dist_history"].append(dist)
        penmark_data["left_history"].append(centro_esq)

        if len(penmark_data["dist_history"]) > WINDOW_SIZE:
            penmark_data["dist_history"].pop(0)
        if len(penmark_data["left_history"]) > WINDOW_SIZE:
            penmark_data["left_history"].pop(0)

        # proximidade
        if dist <= DIST_THRESHOLD:
            cv2.line(annotated, centro_esq, centro_dir, (0, 140, 255), 3)

            if mao_esquerda_parada(penmark_data["left_history"]):
                if current_time - penmark_data["last_detection_time"] >= COOLDOWN_TIME:
                    if detectar_osc_ilatoria(penmark_data["dist_history"]):
                        penmark_data["contador"] += 2
                        penmark_data["last_detection_time"] = current_time
                        penmark_data["last_highlight_time"] = current_time
                        penmark_data["dist_history"].clear()
                        cv2.putText(annotated, "Pen Mark Detected!", (700, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
        else:
            penmark_data["dist_history"].clear()
            penmark_data["left_history"].clear()
    else:
        penmark_data["dist_history"].clear()
        penmark_data["left_history"].clear()

    # --- destaque visual temporário ---
    if (current_time - penmark_data["last_highlight_time"]) <= HIGHLIGHT_TIME and maos["right"] is not None:
        x1r, y1r, x2r, y2r = map(int, maos["right"])
        cv2.rectangle(annotated, (x1r, y1r), (x2r, y2r), (0, 140, 255), 5)
        cv2.putText(annotated, "hand_Right_pen_scratch", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 3, cv2.LINE_AA)

    return penmark_data, annotated


# -------------------------
# FUNÇÃO PRINCIPAL
# -------------------------
def main():
    print(f"[OK] Carregando modelo: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model.to(DEVICE)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    contador = 0
    frames_apenas_uma_mao = 0
    cooldown_frames = 0
    contador_picked = 0
    acima_da_linha_anterior = False
    linha_cor = (255, 105, 180)

    penmark_data = {
        "contador": 0,
        "dist_history": [],
        "left_history": [],
        "last_detection_time": 0,
        "last_highlight_time": -10  # começa inativo
    }

    progress_bar = tqdm(total=total, desc="Processando vídeo", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        maos = identificar_maos(boxes)
        num_maos = len([m for m in maos.values() if m is not None])

        contador, frames_apenas_uma_mao, cooldown_frames = drop_piece_inside_box(
            num_maos, frames_apenas_uma_mao, cooldown_frames, contador, fps
        )

        contador_picked, acima_da_linha_anterior, linha_cor = count_pieces_picked(
            maos, contador_picked, acima_da_linha_anterior, linha_cor
        )

        annotated = frame.copy()

        # --- desenha mãos básicas ---
        if maos["left"] is not None:
            x1, y1, x2, y2 = map(int, maos["left"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(annotated, "hand_Left", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)

        if maos["right"] is not None:
            x1, y1, x2, y2 = map(int, maos["right"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 3)
            cv2.putText(annotated, "hand_Right", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3, cv2.LINE_AA)

        # --- detecção Pen Mark ---
        penmark_data, annotated = detect_pen_mark(maos, annotated, penmark_data)

        # --- linha pile_line ---
        pile_line_y = 715
        cv2.line(annotated, (0, pile_line_y), (width, pile_line_y), linha_cor, 3)
        cv2.putText(annotated, "pile_line", (50, pile_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, linha_cor, 3, cv2.LINE_AA)

        # --- contadores ---
        cv2.rectangle(annotated, (50, 50), (550, 120), (255, 255, 255), -1)
        cv2.putText(annotated, f"Pieces in the box - {contador}",
                    (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.rectangle(annotated, (50, 130), (550, 200), (255, 255, 255), -1)
        cv2.putText(annotated, f"Pieces Picked - {contador_picked}",
                    (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 105, 180), 3, cv2.LINE_AA)

        cv2.rectangle(annotated, (50, 210), (550, 280), (255, 255, 255), -1)
        cv2.putText(annotated, f"Pen Mark - {penmark_data['contador']}",
                    (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 255), 3, cv2.LINE_AA)

        out.write(annotated)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()
    print(f"[✅] Vídeo salvo em: {OUTPUT_VIDEO}")
    print(f"[ℹ️] Pen Marks detectados: {penmark_data['contador']}")

if __name__ == "__main__":
    main()
