from os import listdir
from os.path import join
import random
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import io
import numpy as np
import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import datetime
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, LeakyReLU, Add, Dense, Flatten, \
    UpSampling2D, PReLU
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, LeakyReLU, Add, Dense, Flatten, \
    UpSampling2D, PReLU
import itertools
from IPython import display
import datetime
import time
import gc
from torch import nn
import os
import math
import pandas as pd


def get_patches(img_arr, size=64, stride=32):
    patched_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1
    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping
        for i in range(i_max):
            for j in range(i_max):
                for k in range(5):
                    patched_list.append(img_arr[i * stride: i * stride + size, j * stride: j * stride + size,
                                        k * 36: k * 36 + 64, ])
    else:
        raise ValueError("img_arr.ndim must be equal 4")
    return np.stack(patched_list)

"""
reconstruct patches data
"""
def reconstruct_patch(img_arr, org_img_size, stride=64, size=32):
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")
    if size is None:
        size = img_arr.shape[2]
    if stride is None:
        stride = size
    nm_layers = img_arr.shape[4]
    i_max = (org_img_size[0] // stride ) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride ) + 1 - (size // stride)
    k_max = (org_img_size[2] // 36 ) + 1 - (64 // 36)
    total_nm_images = img_arr.shape[0] // (i_max*j_max*k_max)
    print(img_arr.shape[0])
    images_list = []
    kk=0
    for img_count in range(total_nm_images):
        img_bg = np.zeros((org_img_size[0],org_img_size[1],org_img_size[2],nm_layers), dtype=img_arr[0].dtype)
        print("nm_layers"+str(nm_layers))
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for layer in range(nm_layers):
                        img_bg[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        k * 36: k * 36 + 64,
                        layer,
                        ] = img_arr[kk, :, :, :, layer]
                    kk += 1
        images_list.append(img_bg)
    return np.stack(images_list)


def reconstruct_patch2(img_arr, org_img_size, stride=64, size=32):
    nm_layers = img_arr.shape[4]
    i_max = (org_img_size[0] // stride ) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride ) + 1 - (size // stride)
    k_max = (org_img_size[2] // 36 ) + 1 - (64 // 36)
    total_nm_images = img_arr.shape[0] // (i_max*j_max*k_max)
    print(img_arr.shape[0])
    images_list = []
    kk=0
    for img_count in range(total_nm_images):
        img_bg = np.zeros((org_img_size[0],org_img_size[1],org_img_size[2],nm_layers), dtype=img_arr[0].dtype)
        img_bg[0: 32,0: 32,0:36,1,] = img_arr[0, :,:,:, 1]
        img_bg[0: 32, 0: 32, 36:72, 1, ] = (img_arr[0, :, :, :, 1] + img_arr[1, :, :, :, 1])/2
        img_bg[(i+1) * stride: (i+1) + stride,j * stride: j + stride,k * 36: k + 36,1,] = (img_arr[kk, (i+1) * stride: i + stride,j * stride: j + stride,k * 36: k + 36, 1] + img_arr[kk+1, i * stride: i + stride,j * stride: j + stride,k * 36: k + 36, 1])/2
        images_list.append(img_bg)
    return np.stack(images_list)

# Costum dataset 생성
class MriDataset(Dataset):
    def __init__(self, path2img_patch, transform=False):
        super().__init__()
        self.path2a = join(path2img_patch, 'a')
        self.path2b = join(path2img_patch, 'b')
        self.img_filenames = [x for x in sorted(listdir(self.path2a))]

    def __getitem__(self, index):
        a = np.load(join(self.path2a, self.img_filenames[index]))
        b = np.load(join(self.path2b, self.img_filenames[index]))

        return a, b

    def __len__(self):
        return len(self.img_filenames)


class MriDataset2(Dataset):
    def __init__(self, path2img_patch, transform=False):
        super().__init__()
        self.path2a = join(path2img_patch, 'a')
        self.path2b = join(path2img_patch, 'b')
        self.img_filenames = [x for x in sorted(listdir(self.path2a))]

    def __getitem__(self, index):
        a = np.load(join(self.path2a, self.img_filenames[index]))
        a = a.astype('float32')
        a = torch.from_numpy(a)
        a = a.unsqueeze(0)
        b = np.load(join(self.path2b, self.img_filenames[index]))
        b = b.astype('float32')
        b = torch.from_numpy(b)
        b = b.unsqueeze(0)

        return a, b

    def __len__(self):
        return len(self.img_filenames)


# 데이터셋 불러오기
# path2img = '/content/drive/MyDrive/datasets/dataset4/train'
# path2img_test = '/content/drive/MyDrive/datasets/dataset4/test'
path2img_patch = '/home/milab/LJH/NYH/myHDnGAN/3ddata2/train'
path2img_test_patch = '/home/milab/LJH/NYH/myHDnGAN/3ddata2/test'

# patch버전
train_ds_patch = MriDataset(path2img_patch)
train_ds_test_patch = MriDataset(path2img_test_patch)
train_ds_test_patch2 = MriDataset2(path2img_test_patch)

patched_list_a2 = []
patched_list_b2 = []
for a, b in train_ds_patch:
    patched_cube_a = get_patches(img_arr=a, size=64, stride=32)
    patched_list_a = []
    for j in range(0, 245):
        patched_cube_a_j = patched_cube_a[j, :, :, :]
        patched_cube_a_j = patched_cube_a_j.astype('float32')
        patched_cube_a_j = torch.from_numpy(patched_cube_a_j)
        patched_cube_a_j = patched_cube_a_j.unsqueeze(0)
        patched_list_a.append(patched_cube_a_j)
    patched_list_a2 = patched_list_a2 + patched_list_a

    patched_cube_b = get_patches(img_arr=b, size=64, stride=32)
    patched_list_b = []
    for j in range(0, 245):
        patched_cube_b_j = patched_cube_b[j, :, :, :]
        patched_cube_b_j = patched_cube_b_j.astype('float32')
        patched_cube_b_j = torch.from_numpy(patched_cube_b_j)
        patched_cube_b_j = patched_cube_b_j.unsqueeze(0)
        patched_list_b.append(patched_cube_b_j)
    patched_list_b2 = patched_list_b2 + patched_list_b

patched_list_a3 = []
patched_list_b3 = []
for a, b in train_ds_test_patch:
    patched_cube_a = get_patches(img_arr=a, size=64, stride=32)
    patched_list_a = []
    for j in range(0, 245):
        patched_cube_a_j = patched_cube_a[j, :, :, :]
        patched_cube_a_j = patched_cube_a_j.astype('float32')
        patched_cube_a_j = torch.from_numpy(patched_cube_a_j)
        patched_cube_a_j = patched_cube_a_j.unsqueeze(0)
        patched_cube_a_j = Variable(patched_cube_a_j, requires_grad=True)
        patched_list_a.append(patched_cube_a_j)
    patched_list_a3 = patched_list_a3 + patched_list_a

    patched_cube_b = get_patches(img_arr=b, size=64, stride=32)
    patched_list_b = []
    for j in range(0,245):
        patched_cube_b_j = patched_cube_b[j, :, :, :]
        patched_cube_b_j = patched_cube_b_j.astype('float32')
        patched_cube_b_j = torch.from_numpy(patched_cube_b_j)
        patched_cube_b_j = patched_cube_b_j.unsqueeze(0)
        patched_cube_b_j = Variable(patched_cube_b_j,requires_grad=True)
        patched_list_b.append(patched_cube_b_j)
    patched_list_b3 = patched_list_b3 + patched_list_b

#patch 데이터 로더 생성하기
train_dl_patch_a = DataLoader(patched_list_a2, batch_size=1, shuffle=False)
train_dl_patch_b = DataLoader(patched_list_b2, batch_size=1, shuffle=False)
train_dl_test_patch_a = DataLoader(patched_list_a3, batch_size=1, shuffle=False)
train_dl_test_patch_b = DataLoader(patched_list_b3, batch_size=1, shuffle=False)

train_dl_test = DataLoader(train_ds_test_patch2, batch_size=1, shuffle=False)

# Generator
# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn
from tensorflow.keras import layers
from tensorflow import keras


# Simple 3D U-Net
def conv_block_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
    )
    return model


def conv_trans_block_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        act_fn,
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim, out_dim, act_fn),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
    )
    return model


def conv_block_3_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim, out_dim, act_fn),
        conv_block_3d(out_dim, out_dim, act_fn),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
    )
    return model


class UnetGenerator_3d(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator_3d, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_2_3d(self.num_filter * 12, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = conv_block_2_3d(self.num_filter * 6, self.num_filter * 2, act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = conv_block_2_3d(self.num_filter * 3, self.num_filter * 1, act_fn)

        self.out = conv_block_3d(self.num_filter, out_dim, act_fn)

    def forward(self, x):
        down_1 = self.down_1(x)
        # print("down_1 : " + str(down_1.shape))
        pool_1 = self.pool_1(down_1)
        # print("pool_1 : " + str(pool_1.shape))
        down_2 = self.down_2(pool_1)
        # print("down_2 : " + str(down_2.shape))
        pool_2 = self.pool_2(down_2)
        # print("pool_2 : " + str(pool_2.shape))
        down_3 = self.down_3(pool_2)
        # print("down_3 : " + str(down_3.shape))
        pool_3 = self.pool_3(down_3)
        # print("down_3 : " + str(down_3.shape))

        bridge = self.bridge(pool_3)
        # print("bridge : " + str(bridge.shape))

        trans_1 = self.trans_1(bridge)
        # print("trans_1 : " + str(trans_1.shape))
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        # print("up_1 : " + str(up_1.shape))
        trans_2 = self.trans_2(up_1)
        # print("trans_2 : " + str(trans_2.shape))

        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)

        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)

        out = self.out(up_3)

        return out


# Discriminator
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        alpha = 0.2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        result = self.conv(x)
        # result = self.bn(result)
        return self.lrelu(result)


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Discriminator(nn.Module):

    def __init__(self, final_feature_map_size):
        super().__init__()
        alpha = 0.2
        assert final_feature_map_size > 0

        self.input_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(alpha)
        )

        self.blocks = nn.Sequential(
            ConvBlock(64, 64, 2),
            ConvBlock(64, 128, 1),
            ConvBlock(128, 128, 2),
            ConvBlock(128, 256, 1),
            ConvBlock(256, 256, 2),
            ConvBlock(256, 512, 1),
            ConvBlock(512, 512, 2),
        )

        img_size = final_feature_map_size
        dense_block_input_size = 512 * img_size * img_size

        self.output_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(img_size),
            Flatten(),
            nn.Linear(dense_block_input_size, 1024),
            nn.LeakyReLU(alpha),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        assert x.shape[2] >= 64 and x.shape[3] >= 64
        return self.output_block(self.blocks(self.input_block(x)))


from torch import nn
from torchvision import models


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        model = models.vgg19(pretrained=True)
        model.eval()

        fifth_conv_layer_index = 26
        features = model.features
        self.feature_map_extractor = nn.Sequential(*list(model.features)[:fifth_conv_layer_index + 1])
        self.feature_map_extractor.eval()
        for param in self.feature_map_extractor.parameters():
            param.requires_grad = False

        self.mse = nn.L1Loss()

    def forward(self, real_image, generated_image):
        # print("real_image : " + str(real_image.shape))
        # print("generated_image : " + str(generated_image.shape))
        assert real_image.shape == generated_image.shape

        loss = self.mse(self.feature_map_extractor(generated_image), self.feature_map_extractor(real_image))

        return loss


def gdloss(real, fake):
    dreal = real[:, :, 1:, 1:, 1:] - real[:, :, :-1, :-1, :-1]  # [BCHWD]
    dfake = fake[:, :, 1:, 1:, 1:] - fake[:, :, :-1, :-1, :-1]
    gd_loss = torch.sum((torch.abs(dreal) - torch.abs(dfake)) ** 2, dim=(0, 1, 2, 3, 4))

    return gd_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

ssim_loss = SSIM3D(window_size = 11)

class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_critrion = nn.MSELoss()

    def forward(self, output_labels, target_labels):
        return self.loss_critrion(output_labels, target_labels)


def gdloss(real, fake):
    dreal = real[:, :, 1:, 1:, 1:] - real[:, :, :-1, :-1, :-1]  # [BCHWD]
    dfake = fake[:, :, 1:, 1:, 1:] - fake[:, :, :-1, :-1, :-1]
    gd_loss = torch.sum((torch.abs(dreal) - torch.abs(dfake)) ** 2, dim=(0, 1, 2, 3, 4))

    return gd_loss


def save_state():
    import datetime
    import os

    state = {
        'epoch': epoch+1,
        'discriminator_state_dict': D.state_dict(),
        'generator_state_dict': G.state_dict(),
        'training_results': training_results,
        'DISCRIMINATOR_FINAL_FEATURE_MAP_SIZE': DISCRIMINATOR_FINAL_FEATURE_MAP_SIZE,
        'RESIDUAL_BLOCKS': RESIDUAL_BLOCKS

    }

    file_name = 'ssim1_real' + str(datetime.datetime.now()) + '.pth'
    file_path = os.path.join('/home/milab/LJH/NYH/myHDnGAN/Models', file_name)
    torch.save(state, file_path)
    return file_path


def load_state(file_name):
    import os
    import torch

    saved_file_src = '/home/milab/LJH/NYH/myHDnGAN/Models'
    file_path = os.path.join(saved_file_src, file_name)
    if os.path.isfile(file_path):
        return torch.load(file_path)
    else:
        return None


# Network Parameter
###############################
BATCH_SIZE = 1
EPOCH_NUM = 50
###############################
# Discriminator
###############################
DISCRIMINATOR_FINAL_FEATURE_MAP_SIZE = 10
###############################
RESIDUAL_BLOCKS = 16
# Optimizers
###############################
lr = 0.001

import torch.optim as optim

D = Discriminator(DISCRIMINATOR_FINAL_FEATURE_MAP_SIZE).cuda()
G = UnetGenerator_3d(in_dim=1, out_dim=1, num_filter=4).cuda()

D_loss = DiscriminatorLoss()
bce = nn.BCELoss()
learning_rate_G = 0.002
beta_1 = 0.5
optimizerG = optim.Adam(G.parameters(), lr=learning_rate_G, betas=(beta_1, 0.5))

optim_generator = optim.Adam(G.parameters(), lr, weight_decay=1e-4)
optim_discriminator = optim.Adam(D.parameters(), lr, weight_decay=1e-4)
# StepLR_G = torch.optim.lr_scheduler.StepLR(optim_generator, step_size=10, gamma=0.85)
# StepLR_D = torch.optim.lr_scheduler.StepLR(optim_discriminator, step_size=10, gamma=0.85)
adversarial_criterion = nn.MSELoss()
mean_generator_adversarial_loss = 0.0
mean_generator_l2_loss = 0.0
mean_generator_gdl_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

# Create optimizers for the discriminator and generator
d_optimizer = optim.SGD(D.parameters(), lr)

###############################
# Load training state if exists
###############################
file_name = 'ssim72021-11-21 11:07:47.735603.pth'
state = load_state(file_name)
old_state_exists = state is not None

if old_state_exists:
    print('loading old state from', file_name)
    G.load_state_dict(state['generator_state_dict'])
    D.load_state_dict(state['discriminator_state_dict'])
else:
    print("starting from the beginning")

import random

# Training
INTERLEAV_TRAINING_LIMIT = -1
# For logging the losses
EPOCH_LOG_INTERVAL = 1
BATCH_LOG_INTERVAL = 1
SAVE_MODEL_INTERVAL = 1

sigmoid = nn.Sigmoid()

loss_hist = {'gen': [],
             'dis': []}

G_LOSS = "G_LOSS"
G_L2_LOSS = "G_L2_LOSS"
G_ADV_LOSS = "G_ADV_LOSS"
G_TOTAL_LOSS = "G_TOTAL_LOSS"
G_TRAINING_ITERATIONS = "G_TRAINING_ITERATIONS"
D_REAL_LOSS = "D_REAL_LOSS"
D_FAKE_LOSS = "D_FAKE_LOSS"
D_REAL_TRAINING_ITERATIONS = "D_REAL_TRAINING_ITERATIONS"
D_FAKE_TRAINING_ITERATIONS = "D_FAKE_TRAINING_ITERATIONS"
D_CORRECT_PREDICTIONS = "D_CORRECT_PREDICTIONS"
CURRENT_TRAINED_IMAGES = "CURRENT_TRAINED_IMAGES"
D_ACC = "D_ACC"

if old_state_exists:
    training_results = state['training_results']
    START_EPOCH = state['epoch']
else:
    training_results = {
        G_LOSS: [], G_L2_LOSS: [], G_ADV_LOSS: [], G_TOTAL_LOSS: [], G_TRAINING_ITERATIONS: [],
        D_REAL_LOSS: [], D_FAKE_LOSS: [], D_REAL_TRAINING_ITERATIONS: [], D_FAKE_TRAINING_ITERATIONS: [],
        D_ACC: []
    }
    START_EPOCH = 1

train_on_fake = True

for epoch in range(START_EPOCH, EPOCH_NUM+1):

    running_results = {
        G_LOSS: 0, G_L2_LOSS: 0, G_ADV_LOSS: 0, G_TOTAL_LOSS: 0, G_TRAINING_ITERATIONS: 0,
        D_REAL_LOSS: 0, D_FAKE_LOSS: 0, D_REAL_TRAINING_ITERATIONS: 0, D_FAKE_TRAINING_ITERATIONS: 0,
        D_CORRECT_PREDICTIONS: 0,
        CURRENT_TRAINED_IMAGES: 0
    }

    D.train()
    G.train()

    for batch_id, (a, b) in enumerate(zip(train_dl_patch_a,train_dl_patch_b)):
        a=a.cuda()
        b=b.cuda()
        high_real_patches = b
        high_real_patch = b
        # high_real_patches = b
        # high_real_patch = b
        low_patch = a

        ###############################
        # Choose which netwrok to train
        ###############################

        # assert running_results[D_CORRECT_PREDICTIONS] <= running_results[CURRENT_TRAINED_IMAGES]

        try:
            acc = running_results[D_CORRECT_PREDICTIONS] / running_results[CURRENT_TRAINED_IMAGES]
        except:
            acc = 0.5

        g_train = acc > 0.3
        d_train = acc < 0.85

        ######### Train generator #########
        if g_train:
            optimizerG.zero_grad()
            generated_image = G(a)

            D_high_gen_p = []
            for i in range(64):
                high_gen_i = generated_image[:, :, :, :, i]
                high_gen_i = high_gen_i.squeeze(3)
                D_high_gen_i = D(high_gen_i)
                D_high_gen_p.append(D_high_gen_i)

            D_high_gen_p2 = []
            for i in range(64):
                high_gen_i2 = generated_image[:, :, :, i, :]
                high_gen_i2 = high_gen_i2.squeeze(3)
                D_high_gen_i2 = D(high_gen_i2)
                D_high_gen_p2.append(D_high_gen_i2)

            D_high_gen_p3 = []
            for i in range(64):
                high_gen_i3 = generated_image[:, :, i, :, :]
                high_gen_i3 = high_gen_i3.squeeze(3)
                D_high_gen_i3 = D(high_gen_i3)
                D_high_gen_p3.append(D_high_gen_i3)

            DD_high_gen_p = torch.cat(D_high_gen_p, dim=1)
            DD_high_gen_p2 = torch.cat(D_high_gen_p2, dim=1)
            DD_high_gen_p3 = torch.cat(D_high_gen_p3, dim=1)
            D_high_gen = torch.stack([DD_high_gen_p, DD_high_gen_p2, DD_high_gen_p3], dim=1)
            target_real = torch.ones(D_high_gen.shape)

            #generator_gdl_loss = 1e-7 * gdloss(b, generated_image)
            generator_gdl_loss = 1 - ssim_loss(b, generated_image)
            mean_generator_gdl_loss += generator_gdl_loss

            generator_l2_loss = nn.L1Loss()(b, generated_image)
            mean_generator_l2_loss += generator_l2_loss

            generator_adversarial_loss = nn.BCEWithLogitsLoss()(D_high_gen.cuda(), target_real.cuda())
            mean_generator_adversarial_loss += generator_adversarial_loss

            #generator_total_loss = generator_gdl_loss + generator_l2_loss + 0.001 * generator_adversarial_loss
            generator_total_loss = 1 * generator_gdl_loss + 0.995 * generator_l2_loss + 0.005 * generator_adversarial_loss
            #generator_total_loss = 1.95 * generator_gdl_loss + 0.995 * generator_l2_loss + 0.005 * generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss

            generator_total_loss.backward()
            optimizerG.step()

            running_results[G_LOSS] += generator_gdl_loss.item() * BATCH_SIZE
            running_results[G_L2_LOSS] += generator_l2_loss.item() * BATCH_SIZE
            running_results[G_ADV_LOSS] += generator_adversarial_loss.item() * BATCH_SIZE
            running_results[G_TOTAL_LOSS] += generator_total_loss.item() * BATCH_SIZE
            running_results[G_TRAINING_ITERATIONS] += 1
            running_results[CURRENT_TRAINED_IMAGES] += BATCH_SIZE
            # running_results[D_CORRECT_PREDICTIONS] += (sigmoid(D_high_gen).cpu().detach().numpy() <= 0.5).sum()
            running_results[D_CORRECT_PREDICTIONS] += (sigmoid(D_high_gen) <= 0.5).sum()

            ######### Train discriminator #########
        if d_train:
            d_optimizer.zero_grad()
            if train_on_fake:
                generated_image = G(a)
                dd_fake_loss = []
                for i in range(64):
                    discriminator_generated_image = generated_image[:, :, :, :, i]
                    discriminator_generated_image = discriminator_generated_image.squeeze(3)
                    D_fake_output = D(discriminator_generated_image)
                    dd_fake_loss.append(D_fake_output)

                dd_fake_loss2 = []
                for i in range(64):
                    discriminator_generated_image2 = generated_image[:, :, :, i, :]
                    discriminator_generated_image2 = discriminator_generated_image2.squeeze(3)
                    D_fake_output2 = D(discriminator_generated_image2)
                    dd_fake_loss2.append(D_fake_output2)

                dd_fake_loss3 = []
                for i in range(64):
                    discriminator_generated_image3 = generated_image[:, :, i, :, :]
                    discriminator_generated_image3 = discriminator_generated_image3.squeeze(3)
                    D_fake_output3 = D(discriminator_generated_image3)
                    dd_fake_loss3.append(D_fake_output3)

                ddd_fake_loss1 = torch.cat(dd_fake_loss, dim=1)
                ddd_fake_loss2 = torch.cat(dd_fake_loss2, dim=1)
                ddd_fake_loss3 = torch.cat(dd_fake_loss3, dim=1)
                ddd_fake_loss = torch.stack([ddd_fake_loss1,ddd_fake_loss2,ddd_fake_loss3],dim=1)
                target = torch.zeros_like(ddd_fake_loss)
                d_fake_loss = D_loss(ddd_fake_loss, target)
                d_fake_loss.backward()

                running_results[D_FAKE_LOSS] += d_fake_loss.item() * BATCH_SIZE
                running_results[D_FAKE_TRAINING_ITERATIONS] += 1
                running_results[D_CORRECT_PREDICTIONS] += (sigmoid(D_fake_output) <= 0.5).sum()

            else:
                dd_real_loss = []
                for i in range(64):
                    b_new = b[:, :, :, :, i]
                    b_new = b_new.squeeze(3)
                    D_real_output = D(b_new)
                    dd_real_loss.append(D_real_output)

                dd_real_loss2 = []
                for i in range(64):
                    b_new2 = b[:, :, :, i, :]
                    b_new2 = b_new2.squeeze(3)
                    D_real_output2 = D(b_new2)
                    dd_real_loss2.append(D_real_output2)

                dd_real_loss3 = []
                for i in range(64):
                    b_new3 = b[:, :, i, :, :]
                    b_new3 = b_new3.squeeze(3)
                    D_real_output3 = D(b_new3)
                    dd_real_loss3.append(D_real_output3)

                ddd_real_loss = torch.cat(dd_real_loss, dim=1)
                ddd_real_loss2 = torch.cat(dd_real_loss2, dim=1)
                ddd_real_loss3 = torch.cat(dd_real_loss3, dim=1)
                ddd_real_loss = torch.stack([ddd_real_loss,ddd_real_loss2,ddd_real_loss3],dim=1)
                target = torch.ones_like(ddd_real_loss)
                d_real_loss = D_loss(ddd_real_loss, target)
                d_real_loss.backward()

                running_results[D_REAL_LOSS] += d_real_loss.item() * BATCH_SIZE
                running_results[D_REAL_TRAINING_ITERATIONS] += 1
                running_results[D_CORRECT_PREDICTIONS] += (sigmoid(D_real_output) > 0.5).sum()

            train_on_fake = not train_on_fake
            d_optimizer.step()
            running_results[CURRENT_TRAINED_IMAGES] += BATCH_SIZE

        ###############################
        # Logging
        ###############################

        total_d_iterations = running_results[D_REAL_TRAINING_ITERATIONS] + running_results[D_FAKE_TRAINING_ITERATIONS]
        total_d_loss = running_results[D_REAL_LOSS] + running_results[D_FAKE_LOSS]

        g_images = running_results[G_TRAINING_ITERATIONS] * BATCH_SIZE + 1
        d_real_images = running_results[D_REAL_TRAINING_ITERATIONS] * BATCH_SIZE + 1
        d_fake_images = (running_results[D_FAKE_TRAINING_ITERATIONS] * BATCH_SIZE + 1)

        if batch_id % BATCH_LOG_INTERVAL == 0:
            print(
                '[%d/%d/%d] Acc_D: %.4f Corr_D :%d Used_IMG_D: %d Loss_D: %.4f R_Loss_D: %.4f F_Loss_D: %.4f Loss_G: %.4f L2_G: %.4f Adv_G: %.4f Total_G: %.4f D_Train: %d G_Train: %d' % (
                    batch_id,
                    epoch,
                    EPOCH_NUM,

                    acc,
                    running_results[D_CORRECT_PREDICTIONS],
                    running_results[CURRENT_TRAINED_IMAGES],

                    total_d_loss / (total_d_iterations * BATCH_SIZE),
                    running_results[D_REAL_LOSS] / d_real_images,
                    running_results[D_FAKE_LOSS] / d_fake_images,

                    running_results[G_LOSS] / g_images,
                    running_results[G_L2_LOSS] / g_images,
                    running_results[G_ADV_LOSS] / g_images,
                    running_results[G_TOTAL_LOSS] / g_images,

                    total_d_iterations,
                    running_results[G_TRAINING_ITERATIONS]
                ))

    if epoch % EPOCH_LOG_INTERVAL == 0:
        g_images = running_results[G_TRAINING_ITERATIONS] * BATCH_SIZE + 1
        d_real_images = running_results[D_REAL_TRAINING_ITERATIONS] * BATCH_SIZE + 1
        d_fake_images = (running_results[D_FAKE_TRAINING_ITERATIONS] * BATCH_SIZE + 1)

        training_results[G_LOSS].append(running_results[G_LOSS] / g_images)
        training_results[G_L2_LOSS].append(running_results[G_L2_LOSS] / g_images)
        training_results[G_ADV_LOSS].append(running_results[G_ADV_LOSS] / g_images)
        training_results[G_TOTAL_LOSS].append(running_results[G_TOTAL_LOSS] / g_images)
        training_results[G_TRAINING_ITERATIONS].append(running_results[G_TRAINING_ITERATIONS])
        training_results[D_REAL_LOSS].append(running_results[D_REAL_LOSS] / d_real_images)
        training_results[D_FAKE_LOSS].append(running_results[D_FAKE_LOSS] / d_fake_images)
        training_results[D_REAL_TRAINING_ITERATIONS].append(running_results[D_REAL_TRAINING_ITERATIONS])
        training_results[D_FAKE_TRAINING_ITERATIONS].append(running_results[D_FAKE_TRAINING_ITERATIONS])
        training_results[D_ACC].append(
            running_results[D_CORRECT_PREDICTIONS] / running_results[CURRENT_TRAINED_IMAGES] + 1)

    if epoch % SAVE_MODEL_INTERVAL == 0:
        print("saving model state", save_state())

#####TEST##########


# EPOCH_MSE = 'EPOCH_MSE'
# EPOCH_SSIM = 'EPOCH_SSIM'
#
# valing_results = {EPOCH_MSE: 0}
dataset_size = len(train_dl_patch_a.dataset)
result_dir = '/home/milab/LJH/NYH/myHDnGAN/results'
# j = 0
# patched_list = []
# for a, b in zip(train_dl_test_patch_a,train_dl_test_patch_b):
#     a=a.cuda()
#     a=Variable(a,requires_grad=True)
#     b=b.cuda()
#     SR = G(a)
#
#
#     with torch.no_grad():
#         j = j + 1
#         a=a.squeeze()
#         b=b.squeeze()
#         SR=SR.squeeze()
#         orig_imgs = a.detach().cpu()
#         fake_imgs = SR.detach().cpu()
#         real_imgs = b.detach().cpu()
#         #np.save(os.path.join(dir_save, 'wave_%03d.npy' % j), orig_imgs.squeeze(0))
#         #np.save(os.path.join(dir_save, 'grappa_%03d.npy' % j), real_imgs.squeeze(0))
#         #np.save(os.path.join(dir_save, 'hdngan_%03d.npy' % j), fake_imgs.squeeze(0))
#         fake_imgs = fake_imgs.unsqueeze(0)
#         patched_list.append(fake_imgs)
#         patch = torch.cat(patched_list, dim=0)
#
# with torch.no_grad():
#     print("patch"+str(patch.shape))
#     reconstructed = np.squeeze(reconstruct_patch(img_arr=np.expand_dims(patch.detach().cpu(),axis=-1),
#                                              org_img_size=(256,256,208), stride=32))
#     np.save(os.path.join(dir_save, 'ssim1_real2.npy'), reconstructed)




#
# rangelist = [(0,0),(0,36),(0,72),(0,108),(0,144),
#              (32,0),(32,36),(32,72),(32,108),(32,144),
#              (64,0),(64,36),(64,72),(64,108),(64,144),
#              (96,0),(96,36),(96,72),(96,108),(96,144),
#              (128,0),(128,36),(128,72),(128,108),(128,144),
#              (160,0),(160,36),(160,72),(160,108),(160,144),
#              (192,0),(192,36),(192,72),(192,108),(192,144)]
#
# with torch.no_grad():
#     G.eval()
#     loss_arr = []
#
#     for batch, (a,b) in enumerate(train_dl_test):
#
#         b=b.cuda()
#         a=a.cuda()
#
#         label_fullsize = b
#         input_fullsize = a
#         print(label_fullsize.shape)
#         print(input_fullsize.shape)
#         output_list = []

        # stride = 6
        # ratio1 = [0.9, 0.7, 0.5]
        # ratio2 = [0.5, 0.7, 0.9]
stride = 16
ratio1 = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
ratio2 = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# ratio1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# ratio2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

ratio1 = torch.tensor(ratio1)
ratio2 = torch.tensor(ratio2)
ratio1_z = ratio1[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
ratio2_z = ratio2[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
ratio1_y = ratio1[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
ratio2_y = ratio2[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
ratio1_x = ratio1[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
ratio2_x = ratio2[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

ratio1_z = ratio1_z.cuda()
ratio2_z = ratio2_z.cuda()
ratio1_y = ratio1_y.cuda()
ratio2_y = ratio2_y.cuda()
ratio1_x = ratio1_x.cuda()
ratio2_x = ratio2_x.cuda()

with torch.no_grad():
    G.eval()
    loss_arr = []

    for id,(a,b) in enumerate(train_dl_test,1):
        a=a.cuda()
        b=b.cuda()
        label_fullsize = b
        input_fullsize = a
        print(label_fullsize.shape)
        print(input_fullsize.shape)

        ids = 64 - stride
        x_slice = label_fullsize.shape[3] // ids + 1
        y_slice = label_fullsize.shape[4] // ids + 1
        z_slice = label_fullsize.shape[2] // ids + 1
        # print(x_slice)
        # print(y_slice)
        # print(z_slice)

        # output_z_list = []

        # for k in range(z_slice):
        #     z = k*ids if k != z_slice-1 else label_fullsize.shape[2] - 64

        #     input = input_fullsize[:,:,z:z+64,0:0+64,0:0+64]
        #     output = net(input).clone().detach()
        #     output_z_list.append(output)

        # out_cat = output_z_list[0][:,:,0:ids,:,:]

        # for k2 in range(z_slice-1):
        #     if k2 == z_slice - 1 - 1:
        #         idx_st = ids*(k2+1) - (label_fullsize.shape[2] - 64)
        #         out1 = output_z_list[k2][:,:,ids:ids+(stride//2),:,:].mul(ratio1) + \
        #             output_z_list[k2+1][:,:,idx_st:idx_st+stride//2,:,:].mul(1-ratio1)
        #         out2 = output_z_list[k2][:,:,ids+stride//2:ids+stride,:,:].mul(1-ratio2) + \
        #             output_z_list[k2+1][:,:,idx_st+stride//2:idx_st+stride,:,:].mul(ratio2)
        #         out_next = output_z_list[k2+1][:,:,idx_st+stride:label_fullsize.shape[2],:,:]
        #         out_cat = torch.cat((out_cat, out1, out2, out_next), dim=2)
        #     else:
        #         out1 = output_z_list[k2][:,:,ids:ids+(stride//2),:,:].mul(ratio1) + \
        #             output_z_list[k2+1][:,:,0:stride//2,:,:].mul(1-ratio1)
        #         out2 = output_z_list[k2][:,:,ids+stride//2:ids+stride,:,:].mul(1-ratio2) + \
        #             output_z_list[k2+1][:,:,stride//2:stride,:,:].mul(ratio2)
        #         out_next = output_z_list[k2+1][:,:,stride:ids,:,:]
        #         out_cat = torch.cat((out_cat, out1, out2, out_next), dim=2)
        #     print('for loop')
        #     print(out_cat.shape)

        output_x_list = []

        for i in range(x_slice):
            if i == x_slice - 1:
                x = label_fullsize.shape[3] - 64
            else:
                x = i * ids

            output_y_list = []

            for j in range(y_slice):
                if j == y_slice - 1:
                    y = label_fullsize.shape[4] - 64
                else:
                    y = j * ids

                output_z_list = []

                for k in range(z_slice):
                    z = k * ids if k != z_slice - 1 else label_fullsize.shape[2] - 64

                    input = input_fullsize[:, :, z:z + 64, x:x + 64, y:y + 64]
                    output = G(input).clone().detach()

                    #outputs = G(input)
                    # output = outputs[2]
                    # # output = ( outputs[0] + outputs[1] + outputs[2] ) / 3
                    # output = output.clone().detach()
                    output_z_list.append(output)

                out_cat = output_z_list[0][:, :, 0:ids, :, :]

                for k2 in range(z_slice - 1):
                    if k2 == z_slice - 1 - 1:
                        idx_st = ids * (k2 + 1) - (label_fullsize.shape[2] - 64)
                        out1 = output_z_list[k2][:, :, ids:ids + (stride // 2), :, :].mul(ratio1_z) + \
                               output_z_list[k2 + 1][:, :, idx_st:idx_st + stride // 2, :, :].mul(1 - ratio1_z)
                        out2 = output_z_list[k2][:, :, ids + stride // 2:ids + stride, :, :].mul(1 - ratio2_z) + \
                               output_z_list[k2 + 1][:, :, idx_st + stride // 2:idx_st + stride, :, :].mul(
                                   ratio2_z)
                        out_next = output_z_list[k2 + 1][:, :, idx_st + stride:64, :, :]
                        out_cat = torch.cat((out_cat, out1, out2, out_next), dim=2)
                    else:
                        out1 = output_z_list[k2][:, :, ids:ids + (stride // 2), :, :].mul(ratio1_z) + \
                               output_z_list[k2 + 1][:, :, 0:stride // 2, :, :].mul(1 - ratio1_z)
                        out2 = output_z_list[k2][:, :, ids + stride // 2:ids + stride, :, :].mul(1 - ratio2_z) + \
                               output_z_list[k2 + 1][:, :, stride // 2:stride, :, :].mul(ratio2_z)
                        out_next = output_z_list[k2 + 1][:, :, stride:ids, :, :]
                        out_cat = torch.cat((out_cat, out1, out2, out_next), dim=2)

                output_y_list.append(out_cat)

            out_cat_2 = output_y_list[0][:, :, :, :, 0:ids]

            for j2 in range(y_slice - 1):
                if j2 == y_slice - 1 - 1:
                    idx_st = ids * (j2 + 1) - (label_fullsize.shape[4] - 64)
                    out1 = output_y_list[j2][:, :, :, :, ids:ids + stride // 2].mul(ratio1_y) + \
                           output_y_list[j2 + 1][:, :, :, :, idx_st:idx_st + stride // 2].mul(1 - ratio1_y)
                    out2 = output_y_list[j2][:, :, :, :, ids + stride // 2:ids + stride].mul(1 - ratio2_y) + \
                           output_y_list[j2 + 1][:, :, :, :, idx_st + stride // 2:idx_st + stride].mul(ratio2_y)
                    out_next = output_y_list[j2 + 1][:, :, :, :, idx_st + stride:64]
                    out_cat_2 = torch.cat((out_cat_2, out1, out2, out_next), dim=4)
                else:
                    out1 = output_y_list[j2][:, :, :, :, ids:ids + stride // 2].mul(ratio1_y) + \
                           output_y_list[j2 + 1][:, :, :, :, 0:stride // 2].mul(1 - ratio1_y)
                    out2 = output_y_list[j2][:, :, :, :, ids + stride // 2:ids + stride].mul(1 - ratio2_y) + \
                           output_y_list[j2 + 1][:, :, :, :, stride // 2:stride].mul(ratio2_y)
                    out_next = output_y_list[j2 + 1][:, :, :, :, stride:ids]
                    out_cat_2 = torch.cat((out_cat_2, out1, out2, out_next), dim=4)

            output_x_list.append(out_cat_2)

        out_cat_3 = output_x_list[0][:, :, :, 0:ids, :]

        for i2 in range(x_slice - 1):
            if i2 == x_slice - 1 - 1:
                idx_st = ids * (i2 + 1) - (label_fullsize.shape[3] - 64)
                out1 = output_x_list[i2][:, :, :, ids:ids + stride // 2, :].mul(ratio1_x) + \
                       output_x_list[i2 + 1][:, :, :, idx_st:idx_st + stride // 2, :].mul(1 - ratio1_x)
                out2 = output_x_list[i2][:, :, :, ids + stride // 2:ids + stride, :].mul(1 - ratio2_x) + \
                       output_x_list[i2 + 1][:, :, :, idx_st + stride // 2:idx_st + stride, :].mul(ratio2_x)
                out_next = output_x_list[i2 + 1][:, :, :, idx_st + stride:64, :]
                out_cat_3 = torch.cat((out_cat_3, out1, out2, out_next), dim=3)
            else:
                out1 = output_x_list[i2][:, :, :, ids:ids + stride // 2, :].mul(ratio1_x) + \
                       output_x_list[i2 + 1][:, :, :, 0:stride // 2, :].mul(1 - ratio1_x)
                out2 = output_x_list[i2][:, :, :, ids + stride // 2:ids + stride, :].mul(1 - ratio2_x) + \
                       output_x_list[i2 + 1][:, :, :, stride // 2:stride, :].mul(ratio2_x)
                out_next = output_x_list[i2 + 1][:, :, :, stride:ids, :]
                out_cat_3 = torch.cat((out_cat_3, out1, out2, out_next), dim=3)

        # print('all append')
        # print(out_cat_3.shape)

        # input = torch.to_numpy(input_fullsize)
        # label = torch.to_numpy(label_fullsize)
        # output = torch.to_numpy(out_cat_3)
        with torch.no_grad():
            label = label_fullsize.detach().cpu()
            input = input_fullsize.detach().cpu()
            output = out_cat_3.detach().cpu()


            # np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label.squeeze())
            # np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input.squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'ssim7_05.npy'), output.squeeze())
            # io.savemat(os.path.join(result_dir, 'numpy', 'label_%04d.mat' % id), {'data': label.squeeze()})
            # io.savemat(os.path.join(result_dir, 'numpy', 'input_%04d.mat' % id), {'data': input.squeeze()})
            # io.savemat(os.path.join(result_dir, 'numpy', 'output_%04d.mat' % id), {'data': output.squeeze()})







