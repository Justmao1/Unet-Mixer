import torch
import torch.nn as nn
import torch.nn.functional as F
from convmixer import ConvMixer


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道不一致，则使用 1x1 卷积进行匹配
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 残差连接
        x += shortcut
        x = self.relu(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_mixer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_mixer, self).__init__()
        channel = in_channels
        # 编码器部分
        self.enc1 = conv_block(in_channels=channel, out_channels=channel * 8)
        self.enc2 = conv_block(in_channels=channel * 8, out_channels=channel * 16)
        self.enc3 = conv_block(in_channels=channel * 16, out_channels=channel * 32)
        self.enc4 = conv_block(in_channels=channel * 32, out_channels=channel * 64)
        self.enc5 = conv_block(in_channels=channel * 64, out_channels=channel * 128)
        # 解码器部分
        self.middle = ConvMixer(channel * 128, kernels_per_layer=1, outplane=channel * 128, kernels_size=7)

        self.Up5 = up_conv(ch_in=channel * 128, ch_out=channel * 64)
        self.Up_conv5 = conv_block(in_channels=channel * 128, out_channels=channel * 64)

        self.Up4 = up_conv(ch_in=channel * 64, ch_out=channel * 32)
        self.Up_conv4 = conv_block(in_channels=channel * 64, out_channels=channel * 32)

        self.Up3 = up_conv(ch_in=channel * 32, ch_out=channel * 16)
        self.Up_conv3 = conv_block(in_channels=channel * 32, out_channels=channel * 16)

        self.Up2 = up_conv(ch_in=channel * 16, ch_out=channel * 8)
        self.Up_conv2 = conv_block(in_channels=channel * 16, out_channels=channel * 8)

        self.Conv_1x1 = nn.Conv2d(channel * 8, out_channels, kernel_size=1, stride=1, padding=0)

        # 最后的卷积层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))
        enc5 = self.enc5(F.max_pool2d(enc4, kernel_size=2))

        # 中间部分 (ConvMixer)
        # enc5 = self.middle(enc5)
        # enc5 = self.middle(enc5)
        # 解码器部分
        up5 = self.Up5(enc5)
        up5 = torch.cat([up5, enc4], dim=1)  # 跳跃连接
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

        # 最后的卷积层
        out = self.Conv_1x1(up2)
        return out
