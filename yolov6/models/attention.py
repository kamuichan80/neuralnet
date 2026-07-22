#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Coordinate attention (paper Section 3.4), applied to the deepest backbone feature map."""

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Encodes long-range dependencies along the H and W axes separately (Eq. 7-9), then
    re-weights the input with direction-aware attention gates (Eq. 10-12)."""

    def __init__(self, ch, reduction=32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, ch, 1)
        self.conv_w = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = self.pool_h(x)  # (B, C, H, 1), Eq. 7
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1), Eq. 8

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))  # Eq. 9

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        g_h = torch.sigmoid(self.conv_h(x_h))  # Eq. 10
        g_w = torch.sigmoid(self.conv_w(x_w))  # Eq. 11
        return x * g_h * g_w  # Eq. 12
