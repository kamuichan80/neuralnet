#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""GoogLeNet / Inception v1 (Szegedy et al., 2014), config-driven build."""

import torch
import torch.nn as nn

from classifiers.layers import ConvBNReLU


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = ConvBNReLU(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, ch3x3red, kernel_size=1),
            ConvBNReLU(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, ch5x5red, kernel_size=1),
            ConvBNReLU(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ConvBNReLU(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branches = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(branches, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = ConvBNReLU(64, 64, kernel_size=1)
        self.conv3 = ConvBNReLU(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        # N x 192 x 28 x 28

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        # N x 480 x 14 x 14

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        # N x 832 x 7 x 7

        x = self.inception5a(x)
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(cfg):
    return GoogLeNet(num_classes=cfg.model.num_classes)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from yolov6.utils.config import Config

    cfg = Config.fromfile(ROOT / 'configs' / 'googlenet.py')
    print(build_model(cfg))
