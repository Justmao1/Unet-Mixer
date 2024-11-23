import torch
import torch.nn as nn
from convmixer import ConvMixer


class conv_block(nn.Sequential):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=1):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, bias=False))
        self.add_module("bn1", nn.BatchNorm2d(ch_out))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(ch_out, ch_out, kernel_size, padding=padding, bias=False))
        self.add_module("bn2", nn.BatchNorm2d(ch_out))
        self.add_module("relu2", nn.ReLU(inplace=True))


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
    def __init__(self, img_ch, output_ch):
        super(UNet_mixer, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        channel = img_ch
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=channel * 8)
        self.Conv2 = conv_block(ch_in=channel * 8, ch_out=channel * 16)
        self.Conv3 = conv_block(ch_in=channel * 16, ch_out=channel * 32)
        self.Conv4 = conv_block(ch_in=channel * 32, ch_out=channel * 64)
        self.Conv5 = conv_block(ch_in=channel * 64, ch_out=channel * 128)

        self.middle = ConvMixer(channel * 128, kernels_per_layer=1, outplane=channel * 128, kernels_size=7)

        self.Up5 = up_conv(ch_in=channel * 128, ch_out=channel * 64)
        self.Up_conv5 = conv_block(ch_in=channel * 128, ch_out=channel * 64)

        self.Up4 = up_conv(ch_in=channel * 64, ch_out=channel * 32)
        self.Up_conv4 = conv_block(ch_in=channel * 64, ch_out=channel * 32)

        self.Up3 = up_conv(ch_in=channel * 32, ch_out=channel * 16)
        self.Up_conv3 = conv_block(ch_in=channel * 32, ch_out=channel * 16)

        self.Up2 = up_conv(ch_in=channel * 16, ch_out=channel * 8)
        self.Up_conv2 = conv_block(ch_in=channel * 16, ch_out=channel * 8)

        self.Conv_1x1 = nn.Conv2d(channel * 8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x5 = self.middle(x5)
        x5 = self.middle(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)

        d1 = self.Conv_1x1(d2)

        return d1


# data = torch.randn(1, 3, 256, 256)
# model = UNet_mixer(3, 3)
# out = model(data)
# print(out.shape)
if __name__ == '__main__':
    device = 'cuda:0'
