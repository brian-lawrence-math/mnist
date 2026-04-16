import numpy as np
import struct
from array import array
from os.path import join

import random
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class ConvModel(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.h = h

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn1 = nn.BatchNorm2d(h)
        self.conv2 = nn.Conv2d(
            in_channels=h, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn2 = nn.BatchNorm2d(h)

        self.pool1 = nn.AdaptiveMaxPool2d((14, 14))
        self.drop1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(
            in_channels=h, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn3 = nn.BatchNorm2d(h)
        self.conv4 = nn.Conv2d(
            in_channels=h, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn4 = nn.BatchNorm2d(h)

        self.pool2 = nn.AdaptiveMaxPool2d((7, 7))
        self.drop2 = nn.Dropout(0.1)

        self.conv5 = nn.Conv2d(
            in_channels=h, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn5 = nn.BatchNorm2d(h)
        self.conv6 = nn.Conv2d(
            in_channels=h, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn6 = nn.BatchNorm2d(h)

        self.pool3 = nn.AdaptiveMaxPool2d((3, 3))

        self.conv7 = nn.Conv2d(
            in_channels=h, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn7 = nn.BatchNorm2d(h)
        self.conv8 = nn.Conv2d(
            in_channels=h, out_channels=h, kernel_size=(3, 3), padding="same"
        )
        self.bn8 = nn.BatchNorm2d(h)

        self.pool4 = nn.AdaptiveMaxPool2d((1, 1))
        # avg pool: ~4000 below 0.200, 5000 ~0.150
        # max pool: ~1000 below 0.200, 2000 ~0.120, 5000 ~0.060, acc. 98.2%
        # 2 max pools: ~500 below 0.200, 2000 ~0.060, acc. 97.9%, stopping at 2000
        # 3 max pools: similar, acc. 98.1%
        # wider: 98.7%
        self.lin1 = nn.Linear(h, h)
        self.lin2 = nn.Linear(h, 10)

    def forward(self, x):
        h = self.h

        # normalize
        x = (x / 128) - 0.5
        x.unsqueeze_(-3)
        # (1, 28, 28)
        x = self.conv1(x)
        # (20, 28, 28)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(x)

        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn3(x)

        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn4(x)

        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn5(x)

        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn6(x)

        x = self.pool3(x)

        x = self.conv7(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn7(x)

        x = self.conv8(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn8(x)

        x = self.pool4(x)
        x = x.reshape(
            (
                -1,
                h,
            )
        )
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.lin2(x)

        return x
