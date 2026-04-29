import torch
import torch.nn as nn


class activation_block(nn.Module):
    def __init__(self, outplane):
        super(activation_block, self).__init__()
        self.gelu = nn.GELU()
        self.outplane = outplane
        self.batchnorm = nn.BatchNorm2d(outplane)

    def forward(self, x):
        x = self.gelu(x)
        x = self.batchnorm(x)
        return x


class DepthwiseConv2d(nn.Module):
    def __init__(self, inplane, kernels_per_layer, outplane):
        super(DepthwiseConv2d, self).__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.kernels_per_layer = kernels_per_layer
        self.depthwise = nn.Conv2d(self.inplane, self.inplane * self.kernels_per_layer, kernel_size=3, padding=1,
                                   groups=self.inplane)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class ConvMixer(nn.Module):
    def __init__(self, inplane, kernels_per_layer, outplane, kernels_size):
        super(ConvMixer, self).__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.kernels_per_layer = kernels_per_layer
        self.kernel_size = kernels_size
        self.depthwise = DepthwiseConv2d(self.inplane, self.kernels_per_layer, self.outplane)
        self.pointwise = nn.Conv2d(self.inplane * self.kernels_per_layer, self.outplane, kernel_size=1)
        self.activation = activation_block(self.outplane)

    def forward(self, x):
        # Depthwise convolution
        x0 = x
        x = self.depthwise(x)
        x = x + x0  # Residual

        # Pointwise convolution
        x = self.pointwise(x)
        x = self.activation(x)
        return x
