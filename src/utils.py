import numpy as np
import matplotlib.pyplot as plt


import pyrealsense2 as rs
import cv2
import time
import os
import torch


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

        print(f"[INFO] Captura finalizada. {images_saved} imagens"
              f"salvas na pasta '{class_dir}'.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def print_device_info(device: str):
    """
    Exibe informações sobre o dispositivo utilizado.
    """
    print(f"Dispositivo selecionado: {device}")
    if device == 'cuda':
        print("CUDA disponível:", torch.cuda.is_available())
        print("Total de GPUs:", torch.cuda.device_count())
        print("GPU atual:", torch.cuda.current_device())
        print("Nome da GPU:", torch.cuda.get_device_name(
            torch.cuda.current_device()))
    else:
        print("Nenhuma GPU CUDA disponível.")


def show_tensor_image(tensor_img, label=None, mean=None, std=None):
    """Displays a normalized tensor image."""
    img = tensor_img.numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    if label is not None:
        plt.title(f'Label: {label}')
    plt.show()
