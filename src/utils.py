import matplotlib.pyplot as plt


def show_torch_image(img):
    # Check if the image has only one channel (grayscale)
    if img.shape[0] == 1:
        img = img.squeeze(0).cpu().numpy()
    else:
        img = img.permute(1, 2, 0).cpu().numpy()

    plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
    plt.axis("off")
    plt.show()
