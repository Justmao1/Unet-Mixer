import torch
import torch.nn as nn
import torch.nn.functional as F
from .convmixer import ConvMixer
from .blurpool import BlurPool
from .coordinate_attention import CoordinateAttention


class conv_block(nn.Module):
    """Residual convolution block: two 3x3 convs with BatchNorm + ReLU, plus shortcut."""

    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv to match channels when in_channels != out_channels
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Residual connection
        x += shortcut
        x = self.relu(x)
        return x


class up_conv(nn.Module):
    """Upsample block: bilinear 2x upsample + 3x3 conv + BN + ReLU."""

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_mixer(nn.Module):
    """5-level U-Net with BlurPool anti-aliasing, Coordinate Attention, and ConvMixer bottleneck.

    Encoder: 5 stages, channels  c -> 8c -> 16c -> 32c -> 64c -> 128c
    Downsampling: BlurPool (low-pass filter + strided conv)
    Attention: Coordinate Attention after each encoder stage
    Bottleneck: ConvMixer (depthwise separable convolutions)
    Decoder: 4 stages with skip connections
    """

    def __init__(self, in_channels, out_channels):
        super(UNet_mixer, self).__init__()
        channel = in_channels

        # Encoder
        self.enc1 = conv_block(in_channels=channel, out_channels=channel * 8)
        self.enc2 = conv_block(in_channels=channel * 8, out_channels=channel * 16)
        self.enc3 = conv_block(in_channels=channel * 16, out_channels=channel * 32)
        self.enc4 = conv_block(in_channels=channel * 32, out_channels=channel * 64)
        self.enc5 = conv_block(in_channels=channel * 64, out_channels=channel * 128)

        # BlurPool anti-aliasing downsampling (replaces raw max_pool2d)
        self.blur1 = BlurPool(channel * 8)
        self.blur2 = BlurPool(channel * 16)
        self.blur3 = BlurPool(channel * 32)
        self.blur4 = BlurPool(channel * 64)

        # Coordinate Attention after each encoder stage
        self.ca1 = CoordinateAttention(channel * 8, channel * 8)
        self.ca2 = CoordinateAttention(channel * 16, channel * 16)
        self.ca3 = CoordinateAttention(channel * 32, channel * 32)
        self.ca4 = CoordinateAttention(channel * 64, channel * 64)
        self.ca5 = CoordinateAttention(channel * 128, channel * 128)

        # Bottleneck (ConvMixer)
        self.middle = ConvMixer(channel * 128, kernels_per_layer=1, outplane=channel * 128, kernels_size=7)

        # Decoder
        self.Up5 = up_conv(ch_in=channel * 128, ch_out=channel * 64)
        self.Up_conv5 = conv_block(in_channels=channel * 128, out_channels=channel * 64)

        self.Up4 = up_conv(ch_in=channel * 64, ch_out=channel * 32)
        self.Up_conv4 = conv_block(in_channels=channel * 64, out_channels=channel * 32)

        self.Up3 = up_conv(ch_in=channel * 32, ch_out=channel * 16)
        self.Up_conv3 = conv_block(in_channels=channel * 32, out_channels=channel * 16)

        self.Up2 = up_conv(ch_in=channel * 16, ch_out=channel * 8)
        self.Up_conv2 = conv_block(in_channels=channel * 16, out_channels=channel * 8)

        # Output 1x1 conv
        self.Conv_1x1 = nn.Conv2d(channel * 8, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder with BlurPool anti-aliasing + Coordinate Attention
        enc1 = self.enc1(x)
        enc1 = self.ca1(enc1)
        enc2 = self.enc2(self.blur1(enc1))
        enc2 = self.ca2(enc2)
        enc3 = self.enc3(self.blur2(enc2))
        enc3 = self.ca3(enc3)
        enc4 = self.enc4(self.blur3(enc3))
        enc4 = self.ca4(enc4)
        enc5 = self.enc5(self.blur4(enc4))
        enc5 = self.ca5(enc5)

        # Bottleneck ConvMixer
        enc5 = self.middle(enc5)

        # Decoder with skip connections
        up5 = self.Up5(enc5)
        up5 = torch.cat([up5, enc4], dim=1)
        up5 = self.Up_conv5(up5)

        up4 = self.Up4(up5)
        up4 = torch.cat([up4, enc3], dim=1)
        up4 = self.Up_conv4(up4)

        up3 = self.Up3(up4)
        up3 = torch.cat([up3, enc2], dim=1)
        up3 = self.Up_conv3(up3)

        up2 = self.Up2(up3)
        up2 = torch.cat([up2, enc1], dim=1)
        up2 = self.Up_conv2(up2)

        out = self.Conv_1x1(up2)
        return out
