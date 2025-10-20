from ultralytics import YOLO
import cv2
from pathlib import Path
import torch
from tqdm import tqdm

# -------------------------
# VARIÁVEIS GLOBAIS
# -------------------------
MODEL_PATH = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/trained_models/yolov12n_hands/yolov12_hands_run/weights/best.pt")
INPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/InfiniteFoundry_Challenge/tarefas_cima.mp4")
OUTPUT_VIDEO = Path("/mnt/nas/BrunoScholles/PersonalLearning/outupt_video/output_video_detected.mp4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5


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

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Resolução: {width}x{height} @ {fps:.2f}fps ({total} frames)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    # -------------------------
    # Controle de estado
    # -------------------------
    contador = 0
    frames_apenas_uma_mao = 0
    cooldown_frames = 0

    progress_bar = tqdm(total=total, desc="Processando vídeo", unit="frame")

    # Loop principal
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, device=DEVICE, conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        # Contagem de mãos no frame
        maos_detectadas = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            centro_x = (x1 + x2) / 2
            maos_detectadas.append(centro_x)
        num_maos = len(maos_detectadas)

        # Atualiza contador via função externa
        contador, frames_apenas_uma_mao, cooldown_frames = drop_piece_inside_box(
            num_maos,
            frames_apenas_uma_mao,
            cooldown_frames,
            contador,
            fps
        )

        # Desenha resultados YOLO
        annotated = results[0].plot()

        # Caixa branca opaca com texto azul do contador
        cv2.rectangle(annotated, (50, 50), (550, 120), (255, 255, 255), -1)
        cv2.putText(
            annotated,
            f"Pieces in the box - {contador}",
            (70, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
            cv2.LINE_AA
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
