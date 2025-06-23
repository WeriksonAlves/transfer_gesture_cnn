import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Diretórios
INPUT_DIR = Path("/home/werikson/GitHub/SIBGRAPI2025_classifier/datasets/JHMDB/Frames")   # Onde estão os .avi
OUTPUT_DIR = Path("/home/werikson/GitHub/SIBGRAPI2025_classifier/datasets/JHMDB/ExtractedFrames")  # Para salvar os frames como PNG

# Percorre as pastas por classe
for class_folder in tqdm(os.listdir(INPUT_DIR), desc="Classes"):
    class_path = INPUT_DIR / class_folder
    if not class_path.is_dir():
        continue

    # Percorre os vídeos dentro de cada classe
    for video_file in class_path.glob("*.png"):
        video_name = video_file.stem  # Nome do vídeo sem extensão
        cap = cv2.VideoCapture(str(video_file))

        # Cria a pasta de saída para os frames
        output_video_dir = OUTPUT_DIR / class_folder / video_name
        output_video_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{frame_idx:05d}.png"
            cv2.imwrite(str(output_video_dir / frame_filename), frame)
            frame_idx += 1

        cap.release()

print("✅ Extração finalizada! Frames salvos em:", OUTPUT_DIR)
