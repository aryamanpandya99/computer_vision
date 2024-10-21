"""
Utility functions for general vision tasks.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy import linalg
from torch.nn import functional as F


def show_torch_image(img):
    # Check if the image has only one channel (grayscale)
    if img.shape[0] == 1:
        img = img.squeeze(0).cpu().numpy()
    else:
        img = img.permute(1, 2, 0).cpu().numpy()

    plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
    plt.axis("off")
    plt.show()


def plot_loss(all_train_loss: list, all_valid_loss: list) -> None:
    """
    Plot the training and validation loss.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(all_train_loss, label='Training Loss')
    plt.plot(all_valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def save_checkpoint(
        model: nn.Module, 
        epoch: int, 
        checkpoint_dir: str, 
        optim: torch.optim.Optimizer, 
        all_train_loss: list, 
        all_valid_loss: list,
        checkpoint_name_prefix: str,
    ):
    """
    Save a checkpoint of the model at the given epoch.
    Args:
        model: the model to save
        epoch: the current epoch
        checkpoint_dir: the directory to save the checkpoint in
        optim: the optimizer
        all_train_loss: the training loss
        all_valid_loss: the validation loss
        checkpoint_name_prefix: the prefix for the checkpoint name
    """
    
    checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name_prefix}_epoch_{epoch+1}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_loss': all_train_loss,
        'valid_loss': all_valid_loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved at {checkpoint_path}")


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


def calculate_fid(
        real_images: torch.Tensor, 
        inception: nn.Module, 
        gen_fn, 
        num_gen: int, 
        batch_size: int = 50, 
        device: torch.device = 'cpu'
    ) -> float:
    """
    Calculate the Frechet Inception Distance (FID) between real and generated images.
    Args:
        real_images: the real images
        inception: the Inception model
        gen_fn: the generator function
        num_gen: the number of generated images
        batch_size: the batch size
        device: the device to use
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)