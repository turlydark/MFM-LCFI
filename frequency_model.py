from xception import Xception
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
from attention import DualCrossModalAttention
from attention import CrossModalAttention

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class LFS_Head(nn.Module):
        def __init__(self, size, window_size, M):
            super(LFS_Head, self).__init__()

            self.window_size = window_size
            self._M = M

            # init DCT matrix
            self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
            self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1),
                                             requires_grad=False)

            self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

            # init filters
            self.filters = nn.ModuleList(
                [Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i + 1), norm=True) for i in
                 range(M)])
            # print(self.filters)
            # exit(0)
            self.cmat1 = CrossModalAttention(in_dim=6, ratio=5)
            self.cmat2 = CrossModalAttention(in_dim=6, ratio=5)
            # self.dcma = DualCrossModalAttention(in_dim=6, ratio=1)

        def forward(self, x):
            # turn RGB into Gray
            x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            x = x_gray.unsqueeze(1)

            # rescale to 0 - 255
            x = (x + 1.) * 122.5

            # calculate size
            N, C, W, H = x.size()
            S = self.window_size
            size_after = int((W - S + 8) / 2) + 1
            assert size_after == 149

            # sliding window unfold and DCT
            x_unfold = self.unfold(x)  # [N, C * S * S, L]   L:block num
            L = x_unfold.size()[2]
            x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
            x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

            # 获取dct变换后的幅度谱和相位谱
            x_dct_magnitude_spectrum = x_dct
            x_dct_phase_spectrum = torch.angle(x_dct)
            # print(x_dct_phase_spectrum.size())
            # print(x_dct_magnitude_spectrum.size())
            # exit(0)

            # M kernels filtering
            magnitude_list = []
            for i in range(self._M):
                y = torch.abs(x_dct_magnitude_spectrum)
                y = torch.log10(y + 1e-15)
                y = self.filters[i](y)
                y = torch.sum(y, dim=[2, 3, 4])
                y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)  # [N, 1, 149, 149]
                magnitude_list.append(y)
            magnitude_feature = torch.cat(magnitude_list, dim=1)

            phase_list = []
            for i in range(self._M):
                # x_dct_phase_spectrum = torch.tensor(x_dct_phase_spectrum)
                #注释掉此句
                # x_dct_phase_spectrum = x_dct_phase_spectrum.clone().detach().requires_grad_(True)
                # print(torch.is_tensor(x_dct_phase_spectrum))
                # print(x_dct_phase_spectrum)
                # exit(0)
                y = torch.abs(x_dct_phase_spectrum)
                y = torch.log10(y + 1e-15)
                y = self.filters[i](y)
                y = torch.sum(y, dim=[2, 3, 4])
                y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)  # [N, 1, 149, 149]
                phase_list.append(y)
            phase_feature = torch.cat(phase_list, dim=1)

            # print("magnitude_feature :", magnitude_feature.size())
            # print("phase_feature :", phase_feature.size())
            # exit(0)
            # torch.Size([1, 12, 149, 149])
            # torch.Size([1, 12, 149, 149])



            # magnitude_feature,phase_feature = self.dcma(magnitude_feature, phase_feature)


            # 可运行代码
            out = torch.cat((magnitude_feature, phase_feature), dim=1)
            # torch.Size([8, 12, 149, 149])
            # print(out.size())
            # exit(0)

            # out = self.conv2d_out(out)
            # # torch.Size([8, 6, 149, 149])
            # print(out.size())
            # exit(0)

            # out1 = self.cmat1(magnitude_feature, phase_feature)
            # out2 = self.cmat2(phase_feature, magnitude_feature)
            # out = torch.cat((out1, out2), dim=1)

            # print('Out Size :', out.size())
            # exit(0)
            return out

            #原始输出y
            # y_list = []
            # for i in range(self._M):
            #     # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            #     # y = torch.abs(y)
            #     # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            #     # y = torch.log10(y + 1e-15)
            #     y = torch.abs(x_dct)
            #     y = torch.log10(y + 1e-15)
            #     y = self.filters[i](y)
            #     y = torch.sum(y, dim=[2, 3, 4])
            #     y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)  # [N, 1, 149, 149]
            #     y_list.append(y)
            # out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
            # print(out.size())
            # # torch.Size([8, 6, 149, 149])
            # exit(0)


class FreProcessing(nn.Module):
        def __init__(self, num_classes=2, img_width=299, img_height=299, LFS_window_size=10, LFS_stride=2, LFS_M=6,
                     mode='LFS', device=None):
            super(FreProcessing, self).__init__()
            assert img_width == img_height
            img_size = img_width
            self.num_classes = num_classes
            self.mode = mode
            self.window_size = LFS_window_size
            self._LFS_M = LFS_M

            # # init branches
            # if mode == 'FAD' or mode == 'Both':
            #     self.FAD_head = FAD_Head(img_size)
            #     self.init_xcep_FAD()

            if mode == 'LFS' or mode == 'Both':
                self.LFS_head = LFS_Head(img_size, LFS_window_size, LFS_M)
                self.init_xcep_LFS()

            if mode == 'Original':
                self.init_xcep()

            # classifier
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(4096 if self.mode == 'Both' or self.mode == 'Mix' else 2048, num_classes)
            self.dp = nn.Dropout(p=0.2)

        # def init_xcep_FAD(self):
        #     self.FAD_xcep = Xception(self.num_classes)
        #
        #     # To get a good performance, using ImageNet-pretrained Xception model is recommended
        #     state_dict = get_xcep_state_dict()
        #     conv1_data = state_dict['conv1.weight'].data
        #
        #     self.FAD_xcep.load_state_dict(state_dict, False)
        #
        #     # copy on conv1
        #     # let new conv1 use old param to balance the network
        #     self.FAD_xcep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        #     for i in range(4):
        #         self.FAD_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / 4.0
        def init_xcep_LFS(self):
            self.LFS_xcep = Xception(self.num_classes)

            # To get a good performance, using ImageNet-pretrained Xception model is recommended
            state_dict = get_xcep_state_dict()
            conv1_data = state_dict['conv1.weight'].data

            self.LFS_xcep.load_state_dict(state_dict, False)

            # copy on conv1
            # let new conv1 use old param to balance the network
            self.LFS_xcep.conv1 = nn.Conv2d(self._LFS_M * 2, 32, 3, 1, 0, bias=False)
            for i in range(int(self._LFS_M * 2 / 3)):
                self.LFS_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / float(self._LFS_M / 3.0)

        def init_xcep(self):
            self.xcep = Xception(self.num_classes)

            # To get a good performance, using ImageNet-pretrained Xception model is recommended
            state_dict = get_xcep_state_dict()
            self.xcep.load_state_dict(state_dict, False)

        def forward(self, x):

            # print(x.size())
            # torch.Size([16, 3, 299, 299])
            fea_LFS = self.LFS_head(x)

            # print(fea_LFS.size())
            # exit(0)

            fea_LFS = self.LFS_xcep.features(fea_LFS)
            fea_LFS = self._norm_fea(fea_LFS)
            # print(fea_LFS.size())
            # exit(0)
            # torch.Size([16, 2048])
            f = self.dp(fea_LFS)
            # print(f.size())
            # exit(0)
            # torch.Size([16, 2048])
            f = self.fc(f)
            # print(f.size())
            # exit(0)
            # torch.Size([16, 2])
            y = f
            return y, f

        def _norm_fea(self, fea):
            f = self.relu(fea)
            f = F.adaptive_avg_pool2d(f, (1, 1))
            f = f.view(f.size(0), -1)
            return f

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def get_xcep_state_dict(pretrained_path='./xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict

if __name__ == "__main__":
    mode = 'LFS'  # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreProcessing(mode=mode, device=my_device)
    # print(model)
    dummy = torch.randn((16,3,299,299))
    out = model(dummy)
    print(out)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))