import pyrealsense2 as rs
import cv2
import time
import os
import numpy as np


def capture_gesture_images(output_dir, class_name, image_width=640,
                           image_height=480, camera_fps=30, save_fps=10,
                           total_images=300):
    # Criação da pasta da classe
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Configuração da pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, image_width, image_height,
                         rs.format.bgr8, camera_fps)

    # Inicializar a câmera
    pipeline.start(config)

    print("[INFO] Iniciando captura em 5 segundos...")
    time.sleep(5)
    print("[INFO] Captura iniciada!")

    images_saved = 0
    frame_count = 0
    save_interval = int(camera_fps / save_fps)

    try:
        while images_saved < total_images:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Preview', color_image)

            # Salvar a cada "save_interval" frames
            if frame_count % save_interval == 0:
                filename = os.path.join(
                    class_dir, f"{class_name}_{images_saved:04d}.png")
                cv2.imwrite(filename, color_image)
                images_saved += 1

                if images_saved % 50 == 0:
                    print(f"[INFO] {images_saved} imagens salvas...")

            frame_count += 1

            key = cv2.waitKey(1)
            if key == 27:  # Tecla ESC
                print("[INFO] Captura interrompida pelo usuário.")
                break

        print(f"[INFO] Captura finalizada. {images_saved} imagens salvas na pasta '{class_dir}'.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# Exemplo de uso
if __name__ == "__main__":
    capture_gesture_images(
        output_dir="/home/werikson/GitHub/SIBGRAPI2025_classifier/datasets/my_class",
        class_name="P"  # Troque o nome da classe conforme necessário
    )
