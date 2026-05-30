"""Useful functions"""

import numpy as np
import matplotlib.pyplot as plt

def show_sample(img_tensor, label):
    """Code borrowed from tutorial n°2"""
    
    img = img_tensor.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.title(f"Class: {label}, Shape: ({img_tensor.shape[0]},{img_tensor.shape[1]},{img_tensor.shape[2]})")
    plt.axis('off')
    plt.show()