import matplotlib.pyplot as plt
import numpy as np

def show_torch_image(img):
    # Check if the image has only one channel (grayscale)
    if img.shape[0] == 1:
        img = img.squeeze(0)  # Remove the channel dimension
    else:
        img = img.permute(1, 2, 0)
    
    img = img.numpy().astype(np.uint8)

    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.axis('off')
    plt.show()