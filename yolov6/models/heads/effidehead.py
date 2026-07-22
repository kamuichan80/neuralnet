#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Efficient decoupling head (paper Section 3.3), YOLOv6 heads/effidehead.py-style naming."""

import torch
import torch.nn as nn

from yolov6.layers.common import ConvBNSiLU, RepVGGBlock


class EfficientDecoupledHead(nn.Module):
    """
    Anchor-free, hybrid-channel decoupled head (Figure 7b): the classification branch uses
    a single 3x3 conv while the shared regression/objectness branch uses two stacked 3x3
    convs, before splitting into box regression, objectness and classification predictions.
    """

    def __init__(self, in_ch, num_classes, width=None):
        super().__init__()
        width = width or in_ch // 2
        self.stem = RepVGGBlock(in_ch, width, 3, 1)

        self.cls_conv = ConvBNSiLU(width, width, 3, 1)
        self.cls_pred = nn.Conv2d(width, num_classes, 1)

        self.reg_conv = nn.Sequential(ConvBNSiLU(width, width, 3, 1), ConvBNSiLU(width, width, 3, 1))
        self.reg_pred = nn.Conv2d(width, 4, 1)
        self.obj_pred = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        feat = self.stem(x)
        cls_out = self.cls_pred(self.cls_conv(feat))
        reg_feat = self.reg_conv(feat)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)
        return torch.cat([reg_out, obj_out, cls_out], dim=1)
