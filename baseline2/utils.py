import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    psnr = 20 * torch.log10(1 / rmse)
    return psnr


def torchSSIM(tar_img, prd_img):
    # 将 PyTorch 张量转换为 NumPy 数组
    tar_img_np = tar_img.squeeze().detach().cpu().numpy()  # 去掉 batch 维度并转换为 numpy 数组
    prd_img_np = prd_img.squeeze().detach().cpu().numpy()

    # 如果维度不一致，将灰度图像扩展到 3 通道
    if tar_img_np.ndim != prd_img_np.ndim:
        if tar_img_np.ndim == 2:  # 如果 tar_img 是灰度图像，将其扩展到 3 通道
            tar_img_np = np.repeat(tar_img_np[:, :, np.newaxis], 3, axis=2)
        elif prd_img_np.ndim == 2:  # 如果 prd_img 是灰度图像，将其扩展到 3 通道
            prd_img_np = np.repeat(prd_img_np[:, :, np.newaxis], 3, axis=2)

    # 计算 SSIM，指定 win_size
    ssim = compare_ssim(tar_img_np, prd_img_np, data_range=prd_img_np.max() - prd_img_np.min(),
                        win_size=3, multichannel=True if tar_img_np.ndim == 3 else False)
    return ssim