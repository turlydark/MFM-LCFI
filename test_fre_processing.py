from xception import TransferModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ChannelAttention, SpatialAttention, DualCrossModalAttention
import torch.fft as fft
import numpy as np




def rgb_2_fre_features(x):
    # turn RGB into Gray
    # x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
    # x = x_gray.unsqueeze(1)

    # rescale to 0 - 255
    x = (x + 1.) * 122.5

    # a_s_patches = tf.math.log(tf.math.abs(tf.signal.fftshift(tf.signal.fft2d(patches))) + 1e-10)
    # p_s_patches = tf.math.angle(tf.signal.fftshift(tf.signal.fft2d(patches))) * 180 / tf.constant(np.pi)

    a_fea = torch.log(torch.abs(fft.fftshift(fft.fft2(x) + 1e-10)))
    p_fea = torch.angle(fft.fftshift(fft.fft2(x))) * 180 / np.pi
    print(a_fea.size())
    print(p_fea.size())
    # torch.Size([1, 3, 256, 256])
    # torch.Size([1, 3, 256, 256])


    return a_fea, p_fea

if __name__ == "__main__":
    dummy = torch.rand((1,3,256,256))
    a, p  = rgb_2_fre_features(dummy)
    # print(a)
    # print(p)