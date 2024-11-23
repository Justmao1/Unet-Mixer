import torch
import torch.nn as nn
from unetmixer import UNet_mixer
from antidownsamp import anti_DownSample as antidown
from skff import SKFF


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 初始化
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class mymodel(nn.Module):
    def __init__(self, in_channels=3):
        super(mymodel, self).__init__()
        inch = 3
        self.downsamp1 = antidown(inch)
        self.downsamp2 = antidown(inch * 2)
        self.upsamp1 = UpSample(6, 3)
        self.upsamp2 = UpSample(12, 6)
        self.umixer1 = UNet_mixer(inch, inch)
        self.umixer2 = UNet_mixer(6, 6)
        self.umixer3 = UNet_mixer(12, 12)
        self.skff = SKFF(inch)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, stride=1, kernel_size=1, padding=0)

    def forward(self, x):
        down1 = x

        down1 = self.conv1(down1)
        #
        # down2 = self.downsamp1(down1)
        #
        # down3 = self.downsamp2(down2)  #4,2,128,128

        u1 = self.umixer1(down1)

        # u2 = self.umixer2(down2)
        #
        # u3 = self.umixer3(down3)
        # u2 = self.upsamp1(u2)
        # u3 = self.upsamp2(u3)
        # u3 = self.upsamp1(u3)
        # y = [u1, u2, u3]
        # y = self.skff(y)
        y = u1

        return y


# '''
# data = torch.randn(1, 3, 128, 128)
# model = mymodel(in_channels=3)
# out = model(data)
# print(out.shape)
# '''

# if __name__ == '__main__':
#     device = 'cuda:0'
#     model = mymodel(in_channels=1).to(device)
#     myput = torch.zeros((4, 1, 256, 256)).to(device)
#     print(model(myput).shape)
#
