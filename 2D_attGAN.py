from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F
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
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Costum dataset 생성
class MriDataset(Dataset):
    def __init__(self, path2img,transform=False):
        super().__init__()
        self.path2a = join(path2img, 'a')
        self.path2b = join(path2img, 'b')
        self.img_filenames = [x for x in sorted(listdir(self.path2a))]
        # self.transform = transform

    def __getitem__(self, index):
        a = np.load(join(self.path2a, self.img_filenames[index]))
        a = a.astype('float32')
        a = torch.from_numpy(a)
        a = a.unsqueeze(0)
        a = np.repeat(a[..., np.newaxis], 3, axis=0)
        a = a.squeeze()

        b = np.load(join(self.path2b, self.img_filenames[index]))
        b = b.astype('float32')
        b = torch.from_numpy(b)
        b = b.unsqueeze(0)
        b = np.repeat(b[..., np.newaxis], 3, axis=0)
        b = b.squeeze()

        return a, b

    def __len__(self):
        return len(self.img_filenames)


# 데이터셋 불러오기
path2img = '/home/milab/LJH/NYH/myHDnGAN/sagittal_data/train'
path2img_test = '/home/milab/LJH/NYH/myHDnGAN/sagittal_data/test'
train_ds = MriDataset(path2img)
train_ds_test = MriDataset(path2img_test)

# 데이터 로더 생성하기
train_dl = DataLoader(train_ds, batch_size=4, shuffle=False)
train_dl_test = DataLoader(train_ds_test, batch_size=1, shuffle=False)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        int(in_planes)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, att = True):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.att = att
        if self.att:
            self.ca = ChannelAttention(out_ch)
            self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        if self.att:
            x = self.ca(x) * x
            x = self.sa(x) * x
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, False)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet_att(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet_att, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # return F.sigmoid(x)
        return x

from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        alpha = 0.2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
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
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
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


def PerceptualLoss(real,fake):


    model = models.vgg19(pretrained=True)
    model.eval()

    fifth_conv_layer_index = 26
    features = model.features
    feature_map_extractor = nn.Sequential(*list(model.features)[:fifth_conv_layer_index + 1])
    feature_map_extractor.eval()
    for param in feature_map_extractor.parameters():
        param.requires_grad = False

    mse = nn.L1Loss()

    loss = mse(feature_map_extractor(fake), feature_map_extractor(real))

    return loss




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
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
    C2 = 0.03 ** 2 #0.02로

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # sigma12앞에 2뺴보기
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


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
ssim_loss = SSIM(window_size = 11)


class GeneratorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
        self.discrimenator_loss = nn.BCEWithLogitsLoss()
        self.image_loss = nn.MSELoss()
        self.image_loss2 = nn.L1Loss()



    def forward(self, real_imges, generated_images, output_labels, target_labels):
        self.perc_loss = self.perceptual_loss(real_imges, generated_images)
        self.adv_loss = self.discrimenator_loss(output_labels, target_labels)
        self.img_loss = self.image_loss(generated_images, real_imges)
        self.img_loss2 = self.image_loss2(generated_images, real_imges)

        return self.img_loss + self.perc_loss + 0.001 * self.adv_loss


from torch import nn


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_critrion = nn.BCEWithLogitsLoss()

    def forward(self, output_labels, target_labels):
        return self.loss_critrion(output_labels, target_labels)


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

    file_name = '2d_sag_model_att_ssim_percep ' + str(datetime.datetime.now()) + '.pth'
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
BATCH_SIZE = 4
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

D = Discriminator(DISCRIMINATOR_FINAL_FEATURE_MAP_SIZE)
G = UNet_att()


D_loss = DiscriminatorLoss()
#G_loss = GeneratorLoss()


# Create optimizers for the discriminator and generator
d_optimizer = optim.SGD(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr=0.0001, weight_decay=0.00001)

#optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

###############################
# Load training state if exists
###############################
file_name = 'l_att_ssim 2021-11-15 13:35:00.305693.pth'
state = load_state(file_name)

old_state_exists = state is not None

if old_state_exists:
  print('loading old state from', file_name)
  G.load_state_dict(state['generator_state_dict'])
  D.load_state_dict(state['discriminator_state_dict'])
else:
  print("starting from the beginning")


#D, G = D.cuda(), G.cuda()
#D_loss, G_loss = D_loss.cuda(), G_loss.cuda()

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
G_ADV_LOSS = "G_ADV_LOSS"
G_SSIM_LOSS = "G_SSIM_LOSS"
G_PERC_LOSS = "G_PERC_LOSS"
G_IMG_LOSS = "G_IMG_LOSS"
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
        G_LOSS: [], G_ADV_LOSS: [], G_SSIM_LOSS: [], G_PERC_LOSS: [], G_IMG_LOSS: [], G_TRAINING_ITERATIONS: [],
        D_REAL_LOSS: [], D_FAKE_LOSS: [], D_REAL_TRAINING_ITERATIONS: [], D_FAKE_TRAINING_ITERATIONS: [],
        D_ACC: []
    }
    START_EPOCH = 1

train_on_fake = True

for epoch in range(START_EPOCH, EPOCH_NUM):

    running_results = {
        G_LOSS: 0, G_ADV_LOSS: 0, G_SSIM_LOSS: 0, G_PERC_LOSS: 0, G_IMG_LOSS: 0, G_TRAINING_ITERATIONS: 0,
        D_REAL_LOSS: 0, D_FAKE_LOSS: 0, D_REAL_TRAINING_ITERATIONS: 0, D_FAKE_TRAINING_ITERATIONS: 0,
        D_CORRECT_PREDICTIONS: 0,
        CURRENT_TRAINED_IMAGES: 0
    }

    D.train()
    G.train()

    for batch_id, (a, b) in enumerate(train_dl):
        # b, a = b.cuda(), a.cuda()

        ###############################
        # Choose which netwrok to train
        ###############################

        assert running_results[D_CORRECT_PREDICTIONS] <= running_results[CURRENT_TRAINED_IMAGES]

        try:
            acc = running_results[D_CORRECT_PREDICTIONS] / running_results[CURRENT_TRAINED_IMAGES]
        except:
            acc = 0.5

        g_train = acc > 0.3
        d_train = acc < 0.85

        ###############################
        # Train the Generator
        ###############################

        if g_train:
            g_optimizer.zero_grad()

            generated_image = G(a)

            D_fake_output = D(generated_image)

            # with torch.no_grad():
            # generated_image2=np.repeat(generated_image[..., np.newaxis], 3,axis=1)
            # generated_image2=generated_image2.squeeze()
            # b2=np.repeat(b[..., np.newaxis], 3,axis=1)
            # b2=b2.squeeze()

            # The target is to make the discriminator belive that all the images are real
            #g_loss = G_loss(b, generated_image, D_fake_output, torch.ones_like(D_fake_output) * 0.9)
            g_L1_loss = nn.L1Loss()(generated_image, b)
            g_ssim_loss = 1 - ssim_loss(generated_image, b)
            g_adv_loss = nn.BCEWithLogitsLoss()(D_fake_output, torch.ones_like(D_fake_output) * 0.9)
            g_perc_loss = PerceptualLoss(b, generated_image)
            g_loss = 0.005 * g_adv_loss + 0.995 * g_L1_loss + 0.95 * g_ssim_loss + 0.95 * g_perc_loss
            g_loss.backward()
            g_optimizer.step()

            running_results[G_LOSS] += g_loss.item() * BATCH_SIZE
            running_results[G_ADV_LOSS] += g_adv_loss.item() * BATCH_SIZE
            running_results[G_SSIM_LOSS] += g_ssim_loss.item() * BATCH_SIZE
            running_results[G_PERC_LOSS] += g_perc_loss.item() * BATCH_SIZE
            running_results[G_IMG_LOSS] += g_L1_loss.item() * BATCH_SIZE
            running_results[G_TRAINING_ITERATIONS] += 1
            running_results[CURRENT_TRAINED_IMAGES] += BATCH_SIZE
            running_results[D_CORRECT_PREDICTIONS] += (sigmoid(D_fake_output).cpu().detach().numpy() <= 0.5).sum()

        ###############################
        # Train the discriminator
        ###############################

        if d_train:

            d_optimizer.zero_grad()
            # If random number > 0.5 train on fake data else train on real

            if train_on_fake:
                generated_image = G(a)
                D_fake_output = D(generated_image.detach())
                # The goal is to make the discriminator get the fake images right with smooth factor
                target = torch.zeros_like(D_fake_output) + 0.1
                d_fake_loss = D_loss(D_fake_output, target)
                d_fake_loss.backward()

                running_results[D_FAKE_LOSS] += d_fake_loss.item() * BATCH_SIZE
                running_results[D_FAKE_TRAINING_ITERATIONS] += 1
                running_results[D_CORRECT_PREDICTIONS] += (sigmoid(D_fake_output).cpu().detach().numpy() <= 0.5).sum()
            else:
                D_real_output = D(b)
                # The goal is to make the discriminator get the real images right with smooth factor
                target = torch.ones_like(D_real_output) * 0.9
                d_real_loss = D_loss(D_real_output, target)
                d_real_loss.backward()

                running_results[D_REAL_LOSS] += d_real_loss.item() * BATCH_SIZE
                running_results[D_REAL_TRAINING_ITERATIONS] += 1
                running_results[D_CORRECT_PREDICTIONS] += (sigmoid(D_real_output).cpu().detach().numpy() > 0.5).sum()

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
                '[%d/%d/%d] Acc_D: %.4f Corr_D :%d Used_IMG_D: %d Loss_D: %.4f R_Loss_D: %.4f F_Loss_D: %.4f Loss_G: %.4f Adv_G: %.4f Perc_G: %.4f Img_G: %.4f D_Train: %d G_Train: %d' % (
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
                    running_results[G_ADV_LOSS] / g_images,
                    running_results[G_PERC_LOSS] / g_images,
                    running_results[G_IMG_LOSS] / g_images,

                    total_d_iterations,
                    running_results[G_TRAINING_ITERATIONS]
                ))

    if epoch % EPOCH_LOG_INTERVAL == 0:
        g_images = running_results[G_TRAINING_ITERATIONS] * BATCH_SIZE + 1
        d_real_images = running_results[D_REAL_TRAINING_ITERATIONS] * BATCH_SIZE + 1
        d_fake_images = (running_results[D_FAKE_TRAINING_ITERATIONS] * BATCH_SIZE + 1)

        training_results[G_LOSS].append(running_results[G_LOSS] / g_images)
        training_results[G_ADV_LOSS].append(running_results[G_ADV_LOSS] / g_images)
        training_results[G_PERC_LOSS].append(running_results[G_PERC_LOSS] / g_images)
        training_results[G_IMG_LOSS].append(running_results[G_IMG_LOSS] / g_images)
        training_results[G_TRAINING_ITERATIONS].append(running_results[G_TRAINING_ITERATIONS])
        training_results[D_REAL_LOSS].append(running_results[D_REAL_LOSS] / d_real_images)
        training_results[D_FAKE_LOSS].append(running_results[D_FAKE_LOSS] / d_fake_images)
        training_results[D_REAL_TRAINING_ITERATIONS].append(running_results[D_REAL_TRAINING_ITERATIONS])
        training_results[D_FAKE_TRAINING_ITERATIONS].append(running_results[D_FAKE_TRAINING_ITERATIONS])
        training_results[D_ACC].append(
            running_results[D_CORRECT_PREDICTIONS] / running_results[CURRENT_TRAINED_IMAGES] + 1)

    if epoch % SAVE_MODEL_INTERVAL == 0:
        print("saving model state", save_state())

import os
import math




EPOCH_MSE = 'EPOCH_MSE'
EPOCH_SSIM = 'EPOCH_SSIM'

valing_results = {EPOCH_MSE: 0}
dataset_size = len(train_dl_test.dataset)
j = 0
dir_save = "/home/milab/LJH/NYH/myHDnGAN/results"
for batch_id, (a, b) in enumerate(train_dl_test):
    # a = a.cuda()
    # b = b.cuda()

    SR = G(a)

    # valing_results[EPOCH_MSE] += ((SR - b) ** 2).data.mean() * BATCH_SIZE
    #
    # total_mse_loss = valing_results[EPOCH_MSE] / dataset_size
    # psnr = 10 * math.log10(1 / total_mse_loss)
    #
    # print("MSE: %.4f  PSNR: %.4f" % (total_mse_loss, psnr))

    with torch.no_grad():
        j = j + 1
        orig_imgs = a.detach().cpu()
        fake_imgs = SR.detach().cpu()
        real_imgs = b.detach().cpu()
        #np.save(os.path.join(dir_save, 'wave_%03d.npy' % j), orig_imgs.squeeze(0))
        #np.save(os.path.join(dir_save, 'grappa_%03d.npy' % j), real_imgs.squeeze(0))
        np.save(os.path.join(dir_save, '2d_sag_att_ssim_percep_%03d.npy' % j), fake_imgs.squeeze(0))
        # break
