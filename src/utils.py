# src/utils.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
# import pyrealsense2 as rs


def capture_gesture_images(
    output_dir: str,
    class_name: str,
    image_width: int = 640,
    image_height: int = 480,
    camera_fps: int = 30,
    save_fps: int = 10,
    total_images: int = 300
) -> None:
    """
    Captures gesture images using RealSense and saves them by class.

    Args:
        output_dir (str): Base directory for dataset.
        class_name (str): Name of the gesture class.
        image_width (int): Image width (default: 640).
        image_height (int): Image height (default: 480).
        camera_fps (int): RealSense stream FPS (default: 30).
        save_fps (int): Rate of image saving (default: 10).
        total_images (int): Number of images to save (default: 300).
    """
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # RealSense pipeline setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        image_width,
        image_height,
        rs.format.bgr8,
        camera_fps
    )

    pipeline.start(config)

    print("[INFO] Capture will start in 5 seconds...")
    time.sleep(5)
    print("[INFO] Capture started!")

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
            cv2.imshow("RealSense Preview", color_image)

            if frame_count % save_interval == 0:
                filename = os.path.join(
                    class_dir, f"{class_name}_{images_saved:04d}.png"
                )
                cv2.imwrite(filename, color_image)
                images_saved += 1

                if images_saved % 50 == 0:
                    print(f"[INFO] {images_saved} images saved...")

            frame_count += 1

            if cv2.waitKey(1) == 27:  # ESC key
                print("[INFO] Capture interrupted by user.")
                break

        print(
            f"[INFO] Capture completed. {images_saved} images saved to "
            f"'{class_dir}'."
        )

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def print_device_info(device: str) -> None:
    """
    Displays details about the active computation device (CPU or CUDA).

    Args:
        device (str): Either 'cuda' or 'cpu'.
    """
    print(f"Selected device: {device}")
    if device == "cuda":
        print("CUDA available:", torch.cuda.is_available())
        print("Total GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(
            torch.cuda.current_device())
        )
    else:
        print("No CUDA-compatible GPU available.")


def show_tensor_image(
    tensor_img: torch.Tensor,
    label: str = None,
    mean: list = None,
    std: list = None
) -> None:
    """
    Displays a normalized image tensor using matplotlib.

    Args:
        tensor_img (Tensor): Tensor image with shape (C, H, W).
        label (str, optional): Optional title/label.
        mean (list, optional): Mean values used for normalization.
        std (list, optional): Std values used for normalization.
    """
    img = tensor_img.numpy().transpose(1, 2, 0)
    if mean is not None and std is not None:
        img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.axis("off")
    if label is not None:
        plt.title(f"Label: {label}")
    plt.show()
