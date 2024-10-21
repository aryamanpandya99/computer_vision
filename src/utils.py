import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.nn import functional as F
import numpy as np
from numpy import linalg

def show_torch_image(img):
    # Check if the image has only one channel (grayscale)
    if img.shape[0] == 1:
        img = img.squeeze(0).cpu().numpy()
    else:
        img = img.permute(1, 2, 0).cpu().numpy()

    plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
    plt.axis("off")
    plt.show()


def get_features(
        images: torch.Tensor, 
        inception: nn.Module, 
        device: torch.device, 
        batch_size: int = 50
        ) -> np.ndarray:
    """
    Get features from an Inception model.
    """
    features = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i+batch_size].to(device)
        
        with torch.no_grad():
            feat = inception(F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False))
        
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features) 


def calculate_fid(real_images, inception, gen_fn, num_gen, batch_size=50, device='cpu'):
    """
    Calculate the Frechet Inception Distance (FID) between real and generated images.
    """
    real_features = get_features(real_images, inception, device, batch_size)
    gen_features = get_features(gen_fn(num_gen), inception, device, batch_size)
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean