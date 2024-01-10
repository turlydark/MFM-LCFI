from xception import TransferModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
import torch.fft as fft
import numpy as np

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

class Fre_FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(Fre_FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=2)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Two_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_fre = TransferModel(
            'xception', dropout=0.5, inc=6, return_fea=True)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = FeatureFusionModule()
        self.fea_fusion = Fre_FeatureFusionModule(in_chan=3*2, out_chan=3)

    def rgb_2_fre_features(self, x):
        # turn RGB into Gray
        # x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        # x = x_gray.unsqueeze(1)
        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        a_fea = torch.log(torch.abs(fft.fftshift(fft.fft2(x) + 1e-10)))
        p_fea = torch.angle(fft.fftshift(fft.fft2(x))) * 180 / np.pi

        return  a_fea, p_fea

    def features(self, x):
        # change Xception input channels
        a_fea, p_fea = self.rgb_2_fre_features(x)
        # y = self.fea_fusion(a_fea, p_fea)
        y = torch.cat((a_fea, p_fea), dim=1)

        x = self.xception_rgb.features(x)
        y = self.xception_fre.features(y)
        # print(x.size())
        # print(y.size())
        # exit(0)
        # torch.Size([1, 2048, 8, 8])
        # torch.Size([1, 2048, 8, 8])

        fea = self.fusion(x, y)
        # print(fea.size())
        # exit(0)
        # torch.Size([1, 2048, 8, 8])
        return fea

    def classifier(self, fea):
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original rgb

        Return:
        out: (B, 2) the output for loss computing
        fea: (B, 1024) the flattened features before the last FC
        '''
        out, fea = self.classifier(self.features(x))

        return out, fea

class Fre_Stream_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception_fre = TransferModel(
            'xception', dropout=0.5, inc=2, return_fea=True)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = Fre_FeatureFusionModule(in_chan=2, out_chan=2)

    def rgb_2_fre_features(self, x):
        # turn RGB into Gray
        # x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = 0.2126 * x[:, 0, :, :] + 0.7152 * x[:, 1, :, :] + 0.0722 * x[:, 2, :, :]
        x = x.unsqueeze(1)
        # print(x.size())
        # exit(0)

        # rescale to 0 - 255
        # x = (x + 1.) * 122.5

        a_fea = torch.log(torch.abs(fft.fftshift(fft.fft2(x) + 1e-10)))
        p_fea = torch.angle(fft.fftshift(fft.fft2(x))) * 180 / np.pi
        return  a_fea, p_fea

    def features(self, x):
        # change Xception input channels
        a_fea, p_fea = self.rgb_2_fre_features(x)
        # y = self.fea_fusion(a_fea, p_fea)
        # y = torch.cat((a_fea, p_fea), dim=1)
        # print(a_fea.size())
        # print(p_fea.size())
        y = self.fusion(a_fea, p_fea)
        # print(y.size())
        # exit(0)
        y = self.xception_fre.features(y)
        return y

    def classifier(self, fea):
        out, fea = self.xception_fre.classifier(fea)
        return out, fea

    def forward(self, x):
        out, fea = self.classifier(self.features(x))
        return out, fea

if __name__ == "__main__":
    # model = Two_Stream_Net()
    model = Fre_Stream_Net()
    # print(model)

    dummy = torch.rand((1,3,256,256))
    out, fea= model(dummy)
    print(out)
