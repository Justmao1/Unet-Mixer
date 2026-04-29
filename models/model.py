import torch
import torch.nn as nn
from .unetmixer import UNet_mixer


class mymodel(nn.Module):
    """Top-level model wrapper for MRI image denoising.

    Currently operates in single-scale mode using one UNet_mixer.
    Multi-scale SKFF fusion path is available but disabled by default.
    """

    def __init__(self, in_channels=1):
        super(mymodel, self).__init__()
        self.umixer = UNet_mixer(in_channels, in_channels)

    def forward(self, x):
        return self.umixer(x)
