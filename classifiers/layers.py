#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Shared building block for classifiers/googlenet.py, YOLOv6 layers/common.py-style naming."""

import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU (GoogLeNet's original BasicConv2d)."""

    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
