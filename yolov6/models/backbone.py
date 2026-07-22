#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOv7-ELAN-style backbone with coordinate attention before the final CSPSPPF
(paper Section 2.2, 3.4)."""

import torch.nn as nn

from yolov6.layers.common import CSPSPPF, ELAN, MP1, ConvBNSiLU
from yolov6.models.attention import CoordinateAttention
from yolov6.utils.torch_utils import make_divisible


class Backbone(nn.Module):
    """
    Channel widths are derived once from `base_channels`, scaled by `width_multiple`
    (YOLOv6-style), instead of being hardcoded at every layer. `base_channels` defaults to
    the paper's own widths, so `width_multiple=1.0` reproduces them exactly.
    """

    def __init__(self, base_channels=(32, 64, 128, 256, 512, 1024), width_multiple=1.0):
        super().__init__()
        chs = [make_divisible(c * width_multiple, 8) for c in base_channels]

        self.stem = nn.Sequential(
            ConvBNSiLU(3, chs[0], 3, 1),
            ConvBNSiLU(chs[0], chs[1], 3, 2),
            ConvBNSiLU(chs[1], chs[1], 3, 1),
        )
        self.down1 = ConvBNSiLU(chs[1], chs[2], 3, 2)
        self.elan1 = ELAN(chs[2], chs[3])  # stride 4

        self.mp1 = MP1(chs[3])
        self.elan2 = ELAN(chs[3], chs[4])  # stride 8

        self.mp2 = MP1(chs[4])
        self.elan3 = ELAN(chs[4], chs[5])  # stride 16

        self.mp3 = MP1(chs[5])
        self.elan4 = ELAN(chs[5], chs[5])  # stride 32
        self.ca = CoordinateAttention(chs[5])
        self.csp_sppf = CSPSPPF(chs[5], chs[4])

        # (c2, c3, c4, c5): the four channel counts Neck.__init__ needs to build itself
        self.out_channels = (chs[3], chs[4], chs[5], chs[4])

    def forward(self, x):
        x = self.elan1(self.down1(self.stem(x)))
        c2 = x  # stride 4: shallow feature used by the neck's cross-layer connection

        x = self.elan2(self.mp1(x))
        c3 = x  # stride 8

        x = self.elan3(self.mp2(x))
        c4 = x  # stride 16

        x = self.ca(self.elan4(self.mp3(x)))
        c5 = self.csp_sppf(x)  # stride 32

        return c2, c3, c4, c5
