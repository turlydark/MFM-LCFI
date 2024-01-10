import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xception import xception
from efficientnet.model import EfficientNet
import kornia
import torchvision.models as torchm
from xception import Xception
import torch.nn.init as init
import types
from attention import DualCrossModalAttention
from attention import CrossModalAttention

# model = FreProcessing(mode=mode, device=my_device)
# mode = 'LFS' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
# my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.fc_last = nn.Linear(2048, 512)

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
            f = self.fc_last(f)
            # f = self.fc(f)
            # y = f
            # return y, f
            return f

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

class AttentionMap(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1

        self.num_attentions = out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                      padding=1)  # extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x) + 1
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]), mode='nearest')
        return x * mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(H, W), mode='bilinear', align_corners=True)
        if norm == 1:
            attentions = attentions + 1e-8
        if len(features.shape) == 4:
            feature_matrix = torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix = torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm == 1:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        if norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        if norm == 3:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1) + 1e-8
            feature_matrix /= w
        return feature_matrix


class Texture_Enhance_v1(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        # self.output_features=num_features
        self.output_features = num_features * 4
        self.output_features_d = num_features
        self.conv0 = nn.Conv2d(num_features, num_features, 1)
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features * 2, num_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * num_features)
        self.conv3 = nn.Conv2d(num_features * 3, num_features, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(3 * num_features)
        self.conv_last = nn.Conv2d(num_features * 4, num_features * 4, 1)
        self.bn4 = nn.BatchNorm2d(4 * num_features)
        self.bn_last = nn.BatchNorm2d(num_features * 4)

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                    mode='nearest')
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = torch.cat([feature_maps0, feature_maps1], dim=1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = torch.cat([feature_maps1_, feature_maps2], dim=1)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = torch.cat([feature_maps2_, feature_maps3], dim=1)
        feature_maps = self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True)))
        return feature_maps, feature_maps_d


class Texture_Enhance_v2(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.output_features = num_features
        self.output_features_d = num_features
        self.conv_extract = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv0 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 5, padding=2,
                               groups=num_attentions)
        self.conv1 = nn.Conv2d(num_features * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn1 = nn.BatchNorm2d(num_features * num_attentions)
        self.conv2 = nn.Conv2d(num_features * 2 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn2 = nn.BatchNorm2d(2 * num_features * num_attentions)
        self.conv3 = nn.Conv2d(num_features * 3 * num_attentions, num_features * num_attentions, 3, padding=1,
                               groups=num_attentions)
        self.bn3 = nn.BatchNorm2d(3 * num_features * num_attentions)
        self.conv_last = nn.Conv2d(num_features * 4 * num_attentions, num_features * num_attentions, 1,
                                   groups=num_attentions)
        self.bn4 = nn.BatchNorm2d(4 * num_features * num_attentions)
        self.bn_last = nn.BatchNorm2d(num_features * num_attentions)

        self.M = num_attentions

    def cat(self, a, b):
        B, C, H, W = a.shape
        c = torch.cat([a.reshape(B, self.M, -1, H, W), b.reshape(B, self.M, -1, H, W)], dim=2).reshape(B, -1, H, W)
        return c

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps = self.conv_extract(feature_maps)
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        if feature_maps.size(2) > feature_maps_d.size(2):
            # F.interpolate数组采样操作
            feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]),
                                                        mode='nearest')
        # 从当前图中，返回一个新的Variable。并将grad设为False，即不需要求导。
        # 当后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播。
        attention_maps = (
            torch.tanh(F.interpolate(attention_maps.detach(), (H, W), mode='bilinear', align_corners=True))).unsqueeze(
            2) if type(attention_maps) != tuple else 1
        feature_maps = feature_maps.unsqueeze(1)
        feature_maps = (feature_maps * attention_maps).reshape(B, -1, H, W)
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = self.cat(feature_maps0, feature_maps1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = self.cat(feature_maps1_, feature_maps2)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = self.cat(feature_maps2_, feature_maps3)
        feature_maps = F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True))),
                              inplace=True)
        feature_maps = feature_maps.reshape(B, -1, N, H, W)
        return feature_maps, feature_maps_d


class Auxiliary_Loss_v2(nn.Module):
    def __init__(self, M, N, C, alpha=0.05, margin=1, inner_margin=[0.1, 5]):
        super().__init__()
        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, (H, W), mode='bilinear', align_corners=True)
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers
        center_momentum = feature_matrix - feature_centers
        real_mask = (y == 0).view(-1, 1, 1)
        fcts = self.alpha * torch.mean(center_momentum * real_mask, dim=0) + feature_centers
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd, torch.distributed.ReduceOp.SUM)
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin = self.inner_margin[y]
        intra_class_loss = F.relu(
            torch.norm(feature_matrix - fcts, dim=[1, 2]) * torch.sign(inner_margin) - inner_margin)
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0
        for j in range(M):
            for k in range(j + 1, M):
                inter_class_loss += F.relu(self.margin - torch.dist(fcts[j], fcts[k]), inplace=False)
        inter_calss_loss = inter_class_loss / M / self.alpha
        # fmd=attentions.flatten(2)
        # diverse_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss + inter_class_loss, feature_matrix


class Auxiliary_Loss_v1(nn.Module):
    def __init__(self, M, N, C, alpha=0.05, margin=1, inner_margin=[0.01, 0.02]):
        super().__init__()
        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, (H, W), mode='bilinear', align_corners=True)
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers.detach()
        center_momentum = feature_matrix - feature_centers
        fcts = self.alpha * torch.mean(center_momentum, dim=0) + feature_centers
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd, torch.distributed.ReduceOp.SUM)
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin = torch.gather(self.inner_margin.repeat(B, 1), 1, y.unsqueeze(1))
        intra_class_loss = F.relu(torch.norm(feature_matrix - fcts, dim=-1) - inner_margin)
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0
        for j in range(M):
            for k in range(j + 1, M):
                inter_class_loss += F.relu(self.margin - torch.dist(fcts[j], fcts[k]), inplace=False)
        inter_calss_loss = inter_class_loss / M / self.alpha
        # fmd=attentions.flatten(2)
        # inter_class_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss + inter_class_loss, feature_matrix


class MAT(nn.Module):
    def __init__(self, net='xception', feature_layer='b3', attention_layer='final', num_classes=2, M=8, mid_dims=256, \
                 dropout_rate=0.5, drop_final_rate=0.5, pretrained=False, alpha=0.05, size=(380, 380), margin=1,
                 inner_margin=[0.01, 0.02]):
        super(MAT, self).__init__()
        self.num_classes = num_classes
        self.M = M
        if 'xception' in net:
            self.net = xception(num_classes)
        elif net.split('-')[0] == 'efficientnet':
            self.net = EfficientNet.from_pretrained(net, advprop=True, num_classes=num_classes)

        # 注意 num_features 和 feature_layer
        self.feature_layer = feature_layer
        self.attention_layer = attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        num_features = layers[self.feature_layer].shape[1]
        # 通道数
        # print("num_features", num_features)
        self.mid_dims = mid_dims
        if pretrained:
            a = torch.load(pretrained, map_location='cpu')
            keys = {i: a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys = a['state_dict']
            self.net.load_state_dict(keys, strict=False)
        self.attentions = AttentionMap(layers[self.attention_layer].shape[1], self.M)
        self.atp = AttentionPooling()
        self.texture_enhance = Texture_Enhance_v2(num_features, M)
        # 出处 num_features = layers[self.feature_layer].shape[1]
        self.num_features = self.texture_enhance.output_features
        # output_features_d 为没有纹理信息的特征，即经过了
        self.num_features_d = self.texture_enhance.output_features_d
        self.projection_local = nn.Sequential(nn.Linear(M * self.num_features, mid_dims), nn.Hardswish(),
                                              nn.Linear(mid_dims, mid_dims))
        self.project_final = nn.Linear(layers['final'].shape[1], mid_dims)
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(mid_dims * 2, mid_dims), nn.Hardswish(),
                                                    nn.Linear(mid_dims, num_classes))
        self.auxiliary_loss = Auxiliary_Loss_v2(M, self.num_features_d, num_classes, alpha, margin, inner_margin)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        self.dropout_final = nn.Dropout(drop_final_rate, inplace=True)
        # self.center_loss=Center_Loss(self.num_features*M,num_classes)

    def train_batch(self, x, y, jump_aux=False, drop_final=False):
        layers = self.net(x)
        if self.feature_layer == 'logits':
            logits = layers['logits']
            loss = F.cross_entropy(logits, y)
            return dict(loss=loss, logits=logits)
        feature_maps = layers[self.feature_layer]
        raw_attentions = layers[self.attention_layer]
        attention_maps_ = self.attentions(raw_attentions)
        dropout_mask = self.dropout(torch.ones([attention_maps_.shape[0], self.M, 1], device=x.device))
        attention_maps = attention_maps_ * torch.unsqueeze(dropout_mask, -1)
        feature_maps, feature_maps_d = self.texture_enhance(feature_maps, attention_maps_)
        feature_maps_d = feature_maps_d - feature_maps_d.mean(dim=[2, 3], keepdim=True)
        feature_maps_d = feature_maps_d / (torch.std(feature_maps_d, dim=[2, 3], keepdim=True) + 1e-8)
        feature_matrix_ = self.atp(feature_maps, attention_maps_)
        feature_matrix = feature_matrix_ * dropout_mask

        B, M, N = feature_matrix.size()
        if not jump_aux:
            aux_loss, feature_matrix_d = self.auxiliary_loss(feature_maps_d, attention_maps_, y)
        else:
            feature_matrix_d = self.atp(feature_maps_d, attention_maps_)
            aux_loss = 0
        feature_matrix = feature_matrix.view(B, -1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']
        attention_maps = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps, norm=1).squeeze(1)
        final = self.dropout_final(final)
        projected_final = F.hardswish(self.project_final(final))
        # projected_final=self.dropout(projected_final.view(B,1,-1)).view(B,-1)
        if drop_final:
            projected_final *= 0
        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        ensemble_loss = F.cross_entropy(ensemble_logit, y)
        return dict(ensemble_loss=ensemble_loss, aux_loss=aux_loss, attention_maps=attention_maps_,
                    ensemble_logit=ensemble_logit, feature_matrix=feature_matrix_, feature_matrix_d=feature_matrix_d)

    def forward(self, x, y=0, train_batch=False, AG=None):
        if train_batch:
            if AG is None:
                return self.train_batch(x, y)
            else:
                loss_pack = self.train_batch(x, y)
                with torch.no_grad():
                    Xaug, index = AG.agda(x, loss_pack['attention_maps'])
                # self.eval()
                loss_pack2 = self.train_batch(Xaug, y, jump_aux=False)
                # self.train()
                loss_pack['AGDA_ensemble_loss'] = loss_pack2['ensemble_loss']
                loss_pack['AGDA_aux_loss'] = loss_pack2['aux_loss']
                one_hot = F.one_hot(index, self.M)
                loss_pack['match_loss'] = torch.mean(
                    torch.norm(loss_pack2['feature_matrix_d'] - loss_pack['feature_matrix_d'], dim=-1) * (
                                torch.ones_like(one_hot) - one_hot))
                return loss_pack
        layers = self.net(x)
        if self.feature_layer == 'logits':
            logits = layers['logits']
            return logits
        raw_attentions = layers[self.attention_layer]
        attention_maps = self.attentions(raw_attentions)
        feature_maps = layers[self.feature_layer]
        feature_maps, feature_maps_d = self.texture_enhance(feature_maps, attention_maps)
        feature_matrix = self.atp(feature_maps, attention_maps)
        B, M, N = feature_matrix.size()
        feature_matrix = self.dropout(feature_matrix)
        feature_matrix = feature_matrix.view(B, -1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']
        attention_maps2 = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps2, norm=1).squeeze(1)
        projected_final = F.hardswish(self.project_final(final))
        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        return ensemble_logit


class ThreeShallowLayerMAT(nn.Module):
    def __init__(self, net='xception', feature_layer1='b1', feature_layer2='b2', feature_layer3='b3',
                 attention_layer='final', num_classes=2, M=8, mid_dims=256, \
                 dropout_rate=0.5, drop_final_rate=0.5, pretrained=False, alpha=0.05, size=(380, 380), margin=1,
                 inner_margin=[0.01, 0.02]):
        super(ThreeShallowLayerMAT, self).__init__()
        self.num_classes = num_classes
        self.M = M
        if 'xception' in net:
            self.net = xception(num_classes)
        elif net.split('-')[0] == 'efficientnet':
            self.net = EfficientNet.from_pretrained(net, advprop=True, num_classes=num_classes)

        # 注意 num_features 和 feature_layer
        self.feature_layer1 = feature_layer1
        self.feature_layer2 = feature_layer2
        self.feature_layer3 = feature_layer3

        self.attention_layer = attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))

        # num_features1=layers[self.feature_layer1].shape[1]
        # num_features2=layers[self.feature_layer2].shape[1]
        # num_features3=layers[self.feature_layer3].shape[1]
        # 通道数
        # print("num_features", num_features)

        self.mid_dims = mid_dims
        if pretrained:
            a = torch.load(pretrained, map_location='cpu')
            keys = {i: a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys = a['state_dict']
            self.net.load_state_dict(keys, strict=False)
        self.attentions = AttentionMap(layers[self.attention_layer].shape[1], self.M)
        self.atp = AttentionPooling()

        self.texture_enhance1 = Texture_Enhance_v2(layers[self.feature_layer1].shape[1], M)
        self.texture_enhance2 = Texture_Enhance_v2(layers[self.feature_layer2].shape[1], M)
        self.texture_enhance3 = Texture_Enhance_v2(layers[self.feature_layer3].shape[1], M)

        # 出处 num_features = layers[self.feature_layer].shape[1]
        self.num_features1 = self.texture_enhance1.output_features
        self.num_features2 = self.texture_enhance2.output_features
        self.num_features3 = self.texture_enhance3.output_features
        # output_features_d 为没有纹理信息的特征，即经过了
        self.num_features_d = self.texture_enhance1.output_features_d

        self.projection_local = nn.Sequential(
            nn.Linear(M * (self.num_features1 + self.num_features2 + self.num_features3), mid_dims), nn.Hardswish(),
            nn.Linear(mid_dims, mid_dims))
        # print(self.projection_local)
        # Sequential(
        #     (0): Linear(in_features=640, out_features=256, bias=True)
        # (1): Hardswish()
        # (2): Linear(in_features=256, out_features=256, bias=True)
        # )
        self.project_final = nn.Linear(layers['final'].shape[1], mid_dims)
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(mid_dims * 2, mid_dims), nn.Hardswish(),
                                                    nn.Linear(mid_dims, num_classes))
        self.auxiliary_loss = Auxiliary_Loss_v2(M, self.num_features_d, num_classes, alpha, margin, inner_margin)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        self.dropout_final = nn.Dropout(drop_final_rate, inplace=True)
        # self.center_loss=Center_Loss(self.num_features*M,num_classes)

        # feature_maps1 = layers[self.feature_layer1]
        # feature_maps2 = layers[self.feature_layer2]
        # feature_maps3 = layers[self.feature_layer3]
        # print(feature_maps1.size())
        # print(feature_maps2.size())
        # print(feature_maps3.size())
        # exit(0)
        # torch.Size([1, 16, 190, 190])
        # torch.Size([1, 24, 95, 95])
        # torch.Size([1, 40, 47, 47])

        # raw_attentions = layers[self.attention_layer]
        # attention_maps_ = self.attentions(raw_attentions)
        # dropout_mask = self.dropout(torch.ones([attention_maps_.shape[0], self.M, 1]))
        # attention_maps = attention_maps_ * torch.unsqueeze(dropout_mask, -1)
        #
        # feature_maps1, feature_maps_d1 = self.texture_enhance1(feature_maps1, attention_maps_)
        # feature_maps2, feature_maps_d2 = self.texture_enhance2(feature_maps2, attention_maps_)
        # feature_maps3, feature_maps_d3 = self.texture_enhance3(feature_maps3, attention_maps_)

        # print(feature_maps1.size(), feature_maps_d1.size())
        # print(feature_maps2.size(), feature_maps_d2.size())
        # print(feature_maps3.size(), feature_maps_d3.size())
        # exit(0)
        # torch.Size([1, 8, 16, 190, 190]) torch.Size([1, 16, 11, 11])
        # torch.Size([1, 8, 24, 95, 95]) torch.Size([1, 24, 11, 11])
        # torch.Size([1, 8, 40, 47, 47]) torch.Size([1, 40, 11, 11])

    def train_batch(self, x, y, jump_aux=False, drop_final=False):
        layers = self.net(x)
        if self.feature_layer3 == 'logits':
            logits = layers['logits']
            loss = F.cross_entropy(logits, y)
            return dict(loss=loss, logits=logits)

        # 此处修改feature map
        feature_maps1 = layers[self.feature_layer1]
        feature_maps2 = layers[self.feature_layer2]
        feature_maps3 = layers[self.feature_layer3]

        raw_attentions = layers[self.attention_layer]
        attention_maps_ = self.attentions(raw_attentions)
        dropout_mask = self.dropout(torch.ones([attention_maps_.shape[0], self.M, 1], device=x.device))
        attention_maps = attention_maps_ * torch.unsqueeze(dropout_mask, -1)

        feature_maps1, feature_maps_d1 = self.texture_enhance1(feature_maps1, attention_maps_)
        feature_maps2, feature_maps_d2 = self.texture_enhance2(feature_maps2, attention_maps_)
        feature_maps3, feature_maps_d3 = self.texture_enhance3(feature_maps3, attention_maps_)

        # print(feature_maps1.shape(), feature_maps_d1.shape())
        # print(feature_maps2.shape(), feature_maps_d2.shape())
        # print(feature_maps3.shape(), feature_maps_d3.shape())
        # exit(0)
        # torch.Size([1, 8, 16, 190, 190]) torch.Size([1, 16, 11, 11])
        # torch.Size([1, 8, 24, 95, 95]) torch.Size([1, 24, 11, 11])
        # torch.Size([1, 8, 40, 47, 47]) torch.Size([1, 40, 11, 11])

        feature_maps_d1 = feature_maps_d1 - feature_maps_d1.mean(dim=[2, 3], keepdim=True)
        # feature_maps_d2=feature_maps_d2 - feature_maps_d2.mean(dim=[2,3],keepdim=True)
        # feature_maps_d3=feature_maps_d3 - feature_maps_d3.mean(dim=[2,3],keepdim=True)
        feature_maps_d = feature_maps_d1 / (torch.std(feature_maps_d1, dim=[2, 3], keepdim=True) + 1e-8)

        feature_matrix_1 = self.atp(feature_maps1, attention_maps_)
        feature_matrix_2 = self.atp(feature_maps2, attention_maps_)
        feature_matrix_3 = self.atp(feature_maps3, attention_maps_)
        feature_matrix = feature_matrix_1 * dropout_mask
        # feature_matrix=feature_matrix_2*dropout_mask
        # feature_matrix=feature_matrix_3*dropout_mask

        B, M, N = feature_matrix.size()
        if not jump_aux:
            aux_loss, feature_matrix_d = self.auxiliary_loss(feature_maps_d, attention_maps_, y)
        else:
            feature_matrix_d = self.atp(feature_maps_d, attention_maps_)
            aux_loss = 0
        feature_matrix = feature_matrix.view(B, -1)
        # Hardswish激活函数，在MobileNetV3架构中被提出，相较于swish函数，具有数值稳定性好，计算速度快等优点
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']
        attention_maps = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps, norm=1).squeeze(1)
        final = self.dropout_final(final)
        projected_final = F.hardswish(self.project_final(final))
        # projected_final=self.dropout(projected_final.view(B,1,-1)).view(B,-1)
        if drop_final:
            projected_final *= 0
        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        ensemble_loss = F.cross_entropy(ensemble_logit, y)
        return dict(ensemble_loss=ensemble_loss, aux_loss=aux_loss, attention_maps=attention_maps_,
                    ensemble_logit=ensemble_logit, feature_matrix=feature_matrix_1, feature_matrix_d=feature_matrix_d)

    def forward(self, x, y=0, train_batch=False, AG=None):
        # 不使用
        if train_batch:
            if AG is None:
                return self.train_batch(x, y)
            else:
                loss_pack = self.train_batch(x, y)
                with torch.no_grad():
                    Xaug, index = AG.agda(x, loss_pack['attention_maps'])
                # self.eval()
                loss_pack2 = self.train_batch(Xaug, y, jump_aux=False)
                # self.train()
                loss_pack['AGDA_ensemble_loss'] = loss_pack2['ensemble_loss']
                loss_pack['AGDA_aux_loss'] = loss_pack2['aux_loss']
                one_hot = F.one_hot(index, self.M)
                loss_pack['match_loss'] = torch.mean(
                    torch.norm(loss_pack2['feature_matrix_d'] - loss_pack['feature_matrix_d'], dim=-1) * (
                                torch.ones_like(one_hot) - one_hot))
                return loss_pack
        layers = self.net(x)
        if self.feature_layer3 == 'logits':
            logits = layers['logits']
            return logits
        raw_attentions = layers[self.attention_layer]
        # MARK  self.attentions = AttentionMap(layers[self.attention_layer].shape[1], self.M)
        attention_maps = self.attentions(raw_attentions)

        feature_maps1 = layers[self.feature_layer1]
        feature_maps2 = layers[self.feature_layer2]
        feature_maps3 = layers[self.feature_layer3]

        feature_maps1, feature_maps_d1 = self.texture_enhance1(feature_maps1, attention_maps)
        feature_maps2, feature_maps_d2 = self.texture_enhance2(feature_maps2, attention_maps)
        feature_maps3, feature_maps_d3 = self.texture_enhance3(feature_maps3, attention_maps)

        feature_matrix1 = self.atp(feature_maps1, attention_maps)
        feature_matrix2 = self.atp(feature_maps2, attention_maps)
        feature_matrix3 = self.atp(feature_maps3, attention_maps)
        # print(feature_matrix1.size())
        # print(feature_matrix2.size())
        # print(feature_matrix3.size())
        feature_matrix = torch.cat((feature_matrix1, feature_matrix2, feature_matrix3), 2)
        # print(feature_matrix.size())
        # exit(0)
        # torch.Size([1, 8, 16])
        # torch.Size([1, 8, 24])
        # torch.Size([1, 8, 40])
        # torch.Size([1, 8, 80])
        B, M, N = feature_matrix.size()
        # print(B,M,N)    1, 8, 80
        # print(B1,M1,N1) 1, 8, 16
        feature_matrix = self.dropout(feature_matrix)
        feature_matrix = feature_matrix.view(B, -1)
        # print(feature_matrix.size())
        # torch.Size([1, 640])
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']
        attention_maps2 = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps2, norm=1).squeeze(1)
        projected_final = F.hardswish(self.project_final(final))

        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        # print(feature_matrix.size())
        # exit(0)
        # torch.Size([1, 512])

        # ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        # return feature_matrix, ensemble_logit

        return feature_matrix


def load_state(net, ckpt):
    sd = net.state_dict()
    nd = {}
    for i in ckpt:
        if i in sd and sd[i].shape == ckpt[i].shape:
            nd[i] = ckpt[i]
    net.load_state_dict(nd, strict=False)


class netrunc(nn.Module):
    def __init__(self, net='xception', feature_layer='b3', num_classes=2, dropout_rate=0.5, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        if 'xception' in net:
            self.net = xception(num_classes, escape=feature_layer)
        elif net.split('-')[0] == 'efficientnet':
            self.net = EfficientNet.from_pretrained(net, advprop=True, num_classes=num_classes, escape=feature_layer)
        self.feature_layer = feature_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, 100, 100))
        num_features = layers[self.feature_layer].shape[1]
        if pretrained:
            a = torch.load(pretrained, map_location='cpu')
            keys = {i: a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys = a['state_dict']
            load_state(self.net, keys)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.texture_enhance = Texture_Enhance_v2(num_features, 1)
        self.num_features = self.texture_enhance.output_features
        self.fc = nn.Linear(self.num_features, self.num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        layers = self.net(x)
        feature_maps = layers[self.feature_layer]
        feature_maps = self.texture_enhance(feature_maps, (0.2, 0.2))
        x = self.pooling(feature_maps)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ThreeShallowLayerMATwithF3Net(nn.Module):
    def __init__(self, model_name="efficientnet-b4"):
        super(ThreeShallowLayerMATwithF3Net, self).__init__()
        self.threeMat = ThreeShallowLayerMAT(model_name, size=(299, 299))
        self.freModel = FreProcessing(mode='LFS', device="cuda")
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(512 * 2, 512), nn.Hardswish(), nn.Linear(512, 2))
        # Sequential(
        #     (0): Linear(in_features=512, out_features=256, bias=True)
        # (1): Hardswish()
        # (2): Linear(in_features=256, out_features=2, bias=True)
        # )
    def forward(self, x):
        feature_matrix = self.threeMat(x)
        # print(feature_matrix.size())
        freModel_feature  = self.freModel(x)
        # print(freModel_feature.size())
        feature = torch.cat((feature_matrix, freModel_feature), dim=1)
        # print(feature.size())
        # exit(0)
        out = self.ensemble_classifier_fc(feature)
        return out




if __name__ == "__main__":
    model = ThreeShallowLayerMATwithF3Net()
    dummy = torch.rand((1, 3, 299, 299))
    out  = model(dummy)
    print(out)

    # model = ThreeShallowLayerMAT('efficientnet-b0', size=(380,380))
    # model1 = FreProcessing(mode='LFS', device="cuda")
    # # print(model)
    # dummy = torch.rand((1, 3, 299, 299))
    # ensemble_logit = model1(dummy)
    # print(ensemble_logit)