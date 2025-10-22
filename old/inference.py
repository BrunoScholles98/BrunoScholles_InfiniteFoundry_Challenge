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
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/tarefas_cima_Trim.mp4")
OUTPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/Dataset_Infinite/outupt_video/output_video_detected.mp4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5


# -------------------------
# FUNÇÕES AUXILIARES
# -------------------------
def drop_piece_inside_box(num_maos, frames_apenas_uma_mao, cooldown_frames, contador, fps, alert_data):
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

        # Ativa alerta visual
        alert_data["active"] = True
        alert_data["start_time"] = time.time()

    return contador, frames_apenas_uma_mao, cooldown_frames, alert_data


def desenhar_alerta_piece_in_box(frame, alert_data):
    """Desenha seta + texto azul quando uma peça é detectada na caixa."""
    if not alert_data["active"]:
        return frame

    elapsed = time.time() - alert_data["start_time"]
    if elapsed > alert_data["duration"]:
        alert_data["active"] = False
        return frame

    h, w, _ = frame.shape

    # ---- CONFIGURAÇÃO AUTOMÁTICA ----
    arrow_height = 160            # comprimento da seta
    cor_seta = (255, 0, 0)        # azul (BGR)
    thickness = 25                # grossura da linha

    # ponta inferior da seta a 50px do fundo
    ponta_inf_y = h - 50
    base_y = ponta_inf_y - arrow_height
    base_x = 200                  # posição horizontal
    ponta_inf = (base_x, ponta_inf_y)

    # corpo da seta
    p1 = (base_x, base_y)
    p2 = ponta_inf
    cv2.line(frame, p1, p2, cor_seta, thickness)

    # cabeça da seta (triângulo)
    ponta_esq = (base_x - 50, ponta_inf_y - 70)
    ponta_dir = (base_x + 50, ponta_inf_y - 70)
    pts = np.array([ponta_esq, ponta_dir, ponta_inf], np.int32)
    cv2.fillPoly(frame, [pts], cor_seta)

    # texto acima da seta
    cv2.putText(frame, "PIECE IN THE BOX", (base_x - 100, base_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, cor_seta, 5, cv2.LINE_AA)

    return frame



def identificar_maos(boxes):
    if len(boxes) == 0:
        return {"left": None, "right": None}
    elif len(boxes) == 1:
        return {"left": None, "right": boxes[0]}
    else:
        boxes_sorted = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
        return {"left": boxes_sorted[0], "right": boxes_sorted[1]}


def count_pieces_picked(maos, contador_picked, acima_da_linha_anterior, linha_cor):
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
    DIST_THRESHOLD = 500
    WINDOW_SIZE = 60
    MIN_SIGN_CHANGES = 2
    MAX_SIGN_CHANGES = 6
    AMP_THRESHOLD = 120
    LEFT_STABLE_STD = 10
    COOLDOWN_TIME = 1.0
    HIGHLIGHT_TIME = 1.0

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

    current_time = time.time()

    if maos["left"] is not None and maos["right"] is not None:
        x1l, y1l, x2l, y2l = map(int, maos["left"])
        x1r, y1r, x2r, y2r = map(int, maos["right"])
        centro_esq = ((x1l + x2l) // 2, (y1l + y2l) // 2)
        centro_dir = ((x1r + x2r) // 2, (y1r + y2r) // 2)

        dist = np.linalg.norm(np.array(centro_esq) - np.array(centro_dir))
        penmark_data["dist_history"].append(dist)
        penmark_data["left_history"].append(centro_esq)

        if len(penmark_data["dist_history"]) > WINDOW_SIZE:
            penmark_data["dist_history"].pop(0)
        if len(penmark_data["left_history"]) > WINDOW_SIZE:
            penmark_data["left_history"].pop(0)

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

    if (current_time - penmark_data["last_highlight_time"]) <= HIGHLIGHT_TIME and maos["right"] is not None:
        x1r, y1r, x2r, y2r = map(int, maos["right"])
        cv2.rectangle(annotated, (x1r, y1r), (x2r, y2r), (0, 140, 255), 5)
        cv2.putText(annotated, "hand_Right_pen_scratch", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 3, cv2.LINE_AA)

    return penmark_data, annotated


# -------------------------
# DETECTOR DE PROBE PASS (visual aprimorado)
# -------------------------
def detect_probe_pass(maos, annotated, probe_data):
    current_time = time.time()

    if maos["left"] is not None and maos["right"] is not None:
        x1l, y1l, x2l, y2l = map(int, maos["left"])
        x1r, y1r, x2r, y2r = map(int, maos["right"])

        # Cálculos de referência
        centro_esq = np.array([(x1l + x2l)//2, (y1l + y2l)//2])
        centro_dir = np.array([(x1r + x2r)//2, (y1r + y2r)//2])
        dist_centros = np.linalg.norm(centro_esq - centro_dir)

        # Pontos de interesse
        vertice_sup_esq_dir = np.array([x1r, y1r])              # vértice superior esquerdo da mão direita
        centro_aresta_direita_esq = np.array([x2l, (y1l + y2l)//2])  # meio da aresta direita da mão esquerda

        # Inicialização
        if "ultimo_vertice" not in probe_data:
            probe_data["ultimo_vertice"] = vertice_sup_esq_dir.copy()
            probe_data["dist_anterior"] = dist_centros
            probe_data["trajetoria"] = []

        # Se as mãos estão próximas, começa a monitorar
        if dist_centros <= 500 and not probe_data["monitorando"]:
            probe_data["monitorando"] = True
            probe_data["fase"] = 0
            probe_data["trajetoria"].clear()

        # Se se afastaram muito, reseta
        if dist_centros > 500:
            probe_data["monitorando"] = False
            probe_data["fase"] = 0
            probe_data["followup_active"] = False
            probe_data["trajetoria"].clear()
            return probe_data, annotated

        if probe_data["monitorando"]:
            dx = max(0, x1l - vertice_sup_esq_dir[0], vertice_sup_esq_dir[0] - x2l)
            dy = max(0, y1l - vertice_sup_esq_dir[1], vertice_sup_esq_dir[1] - y2l)
            dist = np.hypot(dx, dy)

            # ---- FASE 0 ----
            if probe_data["fase"] == 0:
                if dist <= 15:
                    probe_data["contador"] += 1
                    probe_data["fase"] = 1
                    probe_data["last_detection_time"] = current_time
                    probe_data["dist_apos_primeira"] = dist_centros
                    probe_data["followup_active"] = True
                    cv2.putText(annotated, "Probe Pass +1 (fase1)", (700, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)
                else:
                    # Linha verde CORRIGIDA: do vértice sup. esquerdo da mão direita -> aresta dir. da mão esquerda
                    cv2.line(annotated, tuple(vertice_sup_esq_dir), tuple(centro_aresta_direita_esq), (0, 255, 100), 2)

            # ---- FASE 1 ----
            elif probe_data["fase"] == 1 and probe_data["followup_active"]:
                afastou = dist_centros - probe_data["dist_apos_primeira"] >= 40
                movimento_vertice = np.linalg.norm(vertice_sup_esq_dir - probe_data["ultimo_vertice"]) >= 10

                probe_data["trajetoria"].append(tuple(vertice_sup_esq_dir))
                if len(probe_data["trajetoria"]) > 2:
                    for i in range(1, len(probe_data["trajetoria"])):
                        cv2.line(annotated, probe_data["trajetoria"][i-1],
                                 probe_data["trajetoria"][i], (0, 255, 100), 3)

                if afastou and movimento_vertice and dist_centros <= 500:
                    probe_data["contador"] += 1
                    probe_data["fase"] = 2
                    probe_data["cooldown_until"] = current_time + 1.5
                    probe_data["followup_active"] = False
                    probe_data["trajetoria"].clear()
                    cv2.putText(annotated, "Probe Pass +1 (fase2)", (700, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

            # Timeout
            if probe_data["followup_active"] and (current_time - probe_data["last_detection_time"]) > 1.5:
                probe_data["followup_active"] = False
                probe_data["fase"] = 0
                probe_data["monitorando"] = False
                probe_data["trajetoria"].clear()

        # Cooldown
        if probe_data["fase"] == 2 and current_time > probe_data["cooldown_until"]:
            probe_data["fase"] = 0
            probe_data["monitorando"] = False

        probe_data["ultimo_vertice"] = vertice_sup_esq_dir.copy()
        probe_data["dist_anterior"] = dist_centros

    # Realce da mão direita
    if (current_time - probe_data["last_detection_time"]) <= 1.0 and maos["right"] is not None:
        x1r, y1r, x2r, y2r = map(int, maos["right"])
        cv2.rectangle(annotated, (x1r, y1r), (x2r, y2r), (0, 255, 100), 5)
        cv2.putText(annotated, "hand_Right_probe_pass", (x1r, y1r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 3, cv2.LINE_AA)

    return probe_data, annotated

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

    # Contadores e variáveis de estado
    contador = 0
    frames_apenas_uma_mao = 0
    cooldown_frames = 0
    contador_picked = 0
    acima_da_linha_anterior = False
    linha_cor = (255, 105, 180)

    # Estruturas auxiliares
    alert_data = {"active": False, "start_time": 0, "duration": 1.5}
    penmark_data = {"contador": 0, "dist_history": [], "left_history": [], 
                    "last_detection_time": 0, "last_highlight_time": -10}
    probe_data = {
        "contador": 0,
        "monitorando": False,
        "fase": 0,
        "followup_active": False,
        "last_detection_time": 0,
        "cooldown_until": 0
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

        # Contagem de peças na caixa
        contador, frames_apenas_uma_mao, cooldown_frames, alert_data = drop_piece_inside_box(
            num_maos, frames_apenas_uma_mao, cooldown_frames, contador, fps, alert_data
        )

        # Contagem de peças pegadas
        contador_picked, acima_da_linha_anterior, linha_cor = count_pieces_picked(
            maos, contador_picked, acima_da_linha_anterior, linha_cor
        )

        annotated = frame.copy()

        # Desenha mãos
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

        # Detectores
        penmark_data, annotated = detect_pen_mark(maos, annotated, penmark_data)
        probe_data, annotated = detect_probe_pass(maos, annotated, probe_data)

        # Linha de referência
        pile_line_y = 715
        cv2.line(annotated, (0, pile_line_y), (width, pile_line_y), linha_cor, 3)

        # Desenhar alerta de "Piece in the Box"
        annotated = desenhar_alerta_piece_in_box(annotated, alert_data)

        # Painel de contadores
        cv2.rectangle(annotated, (50, 50), (550, 360), (255, 255, 255), -1)
        cv2.putText(annotated, f"Pieces in the box - {contador}", (70, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(annotated, f"Pieces Picked - {contador_picked}", (70, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 105, 180), 3)
        cv2.putText(annotated, f"Pen Mark - {penmark_data['contador']}", (70, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 255), 3)
        cv2.putText(annotated, f"Probe Passes - {probe_data['contador']}", (70, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

        # Salvar frame
        out.write(annotated)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()
    print(f"[✅] Vídeo salvo em: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
