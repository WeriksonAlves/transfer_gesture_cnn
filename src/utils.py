import numpy as np
import matplotlib.pyplot as plt


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
