import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    """Anti-aliasing downsampling module.

    Applies a low-pass blur filter before strided downsampling to reduce
    high-frequency artifacts and improve translation invariance.
    Based on Zhang et al. "Making Convolutional Networks Shift-Invariant Again"
    """

    def __init__(self, channels, stride=2, filter_size=3):
        super(BlurPool, self).__init__()
        self.stride = stride
        self.channels = channels

        # Build blur kernel (triangle/bilinear)
        if filter_size == 3:
            kernel = [1, 2, 1]
        elif filter_size == 5:
            kernel = [1, 4, 6, 4, 1]
        else:
            kernel = [1, 2, 1]

        # Normalize and form 2D kernel via outer product
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel_2d = kernel.unsqueeze(1) * kernel.unsqueeze(0)

        # Shape: (1, 1, kH, kW) — applied per-channel via groups
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight', kernel_2d)
        self.filter_size = filter_size

    def forward(self, x):
        # Apply blur filter per channel, then strided downsample
        channels = x.shape[1]
        weight = self.weight.expand(channels, 1, self.filter_size, self.filter_size)
        return F.conv2d(x, weight, stride=self.stride, padding=self.filter_size // 2, groups=channels)
