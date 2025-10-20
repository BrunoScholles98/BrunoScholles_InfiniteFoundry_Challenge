from ultralytics import YOLO
import cv2
from pathlib import Path
import torch
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
    """
    Atualiza o contador com base no número de mãos detectadas e estado atual.

    Regras:
    - Só conta quando há exatamente UMA mão visível por >= 14 frames consecutivos.
    - Cooldown de 2 segundos (baseado no FPS) impede contagens duplicadas.
    """
    cooldown_duracao = int(2 * fps)

    # Atualiza frames com base no número de mãos
    if num_maos == 1:
        frames_apenas_uma_mao += 1
    else:
        frames_apenas_uma_mao = 0

    # Reduz cooldown, se ativo
    if cooldown_frames > 0:
        cooldown_frames -= 1

    # Condição de incremento
    if frames_apenas_uma_mao >= 14 and cooldown_frames == 0:
        contador += 1
        cooldown_frames = cooldown_duracao
        frames_apenas_uma_mao = 0  # reset

    return contador, frames_apenas_uma_mao, cooldown_frames


def identificar_maos(boxes):
    """
    Dado um conjunto de boxes, retorna um dicionário com:
    {
        "left": (x1, y1, x2, y2) ou None,
        "right": (x1, y1, x2, y2) ou None
    }

    Regras:
    - Se há 2 mãos, atribui pela posição X (esquerda/direita).
    - Se há apenas 1 mão, assume que é a direita (a esquerda sumiu).
    - Se nenhuma, ambas são None.
    """
    if len(boxes) == 0:
        return {"left": None, "right": None}
    elif len(boxes) == 1:
        # Só uma mão — é sempre a direita
        return {"left": None, "right": boxes[0]}
    else:
        # Duas mãos — ordena pelos centros em X
        boxes_sorted = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
        return {"left": boxes_sorted[0], "right": boxes_sorted[1]}


def count_pieces_picked(maos, contador_picked, acima_da_linha_anterior):
    """
    Controla o contador 'Pieces Picked' com base no movimento da mão esquerda
    em relação à linha rosa (pile_line).

    - A linha está em y = 715 px.
    - Se a centroide da mão esquerda sobe acima da linha (y_centroid < 715)
      e depois volta para baixo (y_centroid >= 715), soma 1.
    """
    pile_line_y = 715
    acima_da_linha_atual = False

    if maos["left"] is not None:
        x1, y1, x2, y2 = map(int, maos["left"])
        y_centroid = (y1 + y2) // 2

        # Está acima da linha?
        acima_da_linha_atual = y_centroid < pile_line_y

        # Detecta transição: estava acima e agora desceu
        if acima_da_linha_anterior and not acima_da_linha_atual:
            contador_picked += 1

    else:
        # Sem mão esquerda -> mantém estado anterior
        acima_da_linha_atual = acima_da_linha_anterior

    return contador_picked, acima_da_linha_atual


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

    print(f"[INFO] Resolução: {width}x{height} @ {fps:.2f}fps ({total} frames)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    # -------------------------
    # Controle de estado
    # -------------------------
    contador = 0
    frames_apenas_uma_mao = 0
    cooldown_frames = 0

    contador_picked = 0
    acima_da_linha_anterior = False

    progress_bar = tqdm(total=total, desc="Processando vídeo", unit="frame")

    # Loop principal
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        # Identificar mãos (esquerda/direita)
        maos = identificar_maos(boxes)
        num_maos = len([m for m in maos.values() if m is not None])

        # Atualiza contador via função externa (peças colocadas)
        contador, frames_apenas_uma_mao, cooldown_frames = drop_piece_inside_box(
            num_maos,
            frames_apenas_uma_mao,
            cooldown_frames,
            contador,
            fps
        )

        # Atualiza contador Pieces Picked (mão esquerda cruzando linha)
        contador_picked, acima_da_linha_anterior = count_pieces_picked(
            maos, contador_picked, acima_da_linha_anterior
        )

        # Desenhar bounding boxes personalizadas
        annotated = frame.copy()

        if maos["left"] is not None:
            x1, y1, x2, y2 = map(int, maos["left"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(
                annotated,
                "hand_Left",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )

        if maos["right"] is not None:
            x1, y1, x2, y2 = map(int, maos["right"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 3)
            cv2.putText(
                annotated,
                "hand_Right",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 200, 0),
                3,
                cv2.LINE_AA,
            )

        # --- Linha rosa (pile_line) ---
        cv2.line(annotated, (0, 715), (width, 715), (255, 105, 180), 3)
        cv2.putText(
            annotated,
            "pile_line",
            (50, 705),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 105, 180),
            3,
            cv2.LINE_AA,
        )

        # --- Caixa branca com contador azul (peças na caixa) ---
        cv2.rectangle(annotated, (50, 50), (550, 120), (255, 255, 255), -1)
        cv2.putText(
            annotated,
            f"Pieces in the box - {contador}",
            (70, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )

        # --- Novo contador "Pieces Picked" ---
        cv2.rectangle(annotated, (50, 130), (550, 200), (255, 255, 255), -1)
        cv2.putText(
            annotated,
            f"Pieces Picked - {contador_picked}",
            (70, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 105, 180),
            3,
            cv2.LINE_AA,
        )

        out.write(annotated)
        progress_bar.update(1)

    # Finaliza
    progress_bar.close()
    cap.release()
    out.release()
    print(f"[✅] Vídeo salvo em: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()