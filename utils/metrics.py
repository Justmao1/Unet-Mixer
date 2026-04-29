import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def torchPSNR(tar_img, prd_img):
    """Compute PSNR between two torch tensors (values in [0, 1])."""
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    psnr = 20 * torch.log10(1 / rmse)
    return psnr


def torchSSIM(tar_img, prd_img):
    """Compute SSIM between two torch tensors."""
    tar_img_np = tar_img.squeeze().detach().cpu().numpy()
    prd_img_np = prd_img.squeeze().detach().cpu().numpy()

    # Handle dimension mismatch for grayscale / RGB
    if tar_img_np.ndim != prd_img_np.ndim:
        if tar_img_np.ndim == 2:
            tar_img_np = np.repeat(tar_img_np[:, :, np.newaxis], 3, axis=2)
        elif prd_img_np.ndim == 2:
            prd_img_np = np.repeat(prd_img_np[:, :, np.newaxis], 3, axis=2)

    ssim = compare_ssim(tar_img_np, prd_img_np, data_range=prd_img_np.max() - prd_img_np.min(),
                        win_size=3, multichannel=True if tar_img_np.ndim == 3 else False)
    return ssim
