import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """Coordinate Attention module.

    Captures long-range dependencies with precise positional information
    by embedding channel attention into spatial attention via two 1D
    global pooling operations (horizontal and vertical).
    Based on Hou et al. "Coordinate Attention for Efficient Mobile Network Design"
    """

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        mid_channels = max(8, in_channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # Horizontal and vertical pooling
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Concatenate along spatial dimension and apply shared conv
        x_cat = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)

        # Split back
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, 1, W)

        # Generate attention weights
        att_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        att_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)

        # Apply attention
        out = identity * att_h * att_w
        return out
