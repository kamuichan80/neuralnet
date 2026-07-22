#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
PAN neck extended with a cross-layer connection (paper Section 3.2): besides the usual
adjacent-level lateral skips, the shallow stride-4 backbone feature (c2) is downsampled and
spliced into both top-down fusion nodes so low-level edge/texture detail, which would
otherwise be diluted by repeated up-sampling, reaches the deeper feature maps directly.
No YOLOv6 equivalent exists for this connection — it's the paper's own contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.layers.common import ELANH, MP2, ConvBNSiLU


class Neck(nn.Module):
    """
    Every internal width is a fraction of `p2` (the shallow c2 channel count), matching the
    ratios `Backbone`'s original hardcoded channels used — so it scales consistently with
    `Backbone`'s `width_multiple` instead of needing its own separate config.
    """

    def __init__(self, in_channels):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        p2 = c2

        self.reduce_p5 = ConvBNSiLU(c5, p2, 1, 1)
        self.lateral_c4 = ConvBNSiLU(c4, p2, 1, 1)
        self.cross_c2_to_p4 = nn.Sequential(ConvBNSiLU(c2, p2 // 2, 3, 2), ConvBNSiLU(p2 // 2, p2 // 2, 3, 2))
        self.elanh_p4 = ELANH(p2 * 2 + p2 // 2, p2)

        self.reduce_p4 = ConvBNSiLU(p2, p2 // 2, 1, 1)
        self.lateral_c3 = ConvBNSiLU(c3, p2 // 2, 1, 1)
        self.cross_c2_to_p3 = ConvBNSiLU(c2, p2 // 4, 3, 2)
        self.elanh_p3 = ELANH(p2 // 2 * 2 + p2 // 4, p2 // 2)

        self.mp2_p3 = MP2(p2 // 2)
        self.elanh_p4_out = ELANH(p2 // 2 + p2, p2)

        self.mp2_p4 = MP2(p2)
        self.elanh_p5_out = ELANH(p2 + c5, c5)

        # (p3, p4, p5) output channels, for DECModule/EfficientDecoupledHead construction
        self.out_channels = (p2 // 2, p2, c5)

    def forward(self, c2, c3, c4, c5):
        p5 = c5

        up_p5 = F.interpolate(self.reduce_p5(p5), scale_factor=2, mode='nearest')
        p4_td = self.elanh_p4(torch.cat([up_p5, self.lateral_c4(c4), self.cross_c2_to_p4(c2)], dim=1))

        up_p4 = F.interpolate(self.reduce_p4(p4_td), scale_factor=2, mode='nearest')
        p3_out = self.elanh_p3(torch.cat([up_p4, self.lateral_c3(c3), self.cross_c2_to_p3(c2)], dim=1))

        p4_out = self.elanh_p4_out(torch.cat([self.mp2_p3(p3_out), p4_td], dim=1))
        p5_out = self.elanh_p5_out(torch.cat([self.mp2_p4(p4_out), p5], dim=1))

        return p3_out, p4_out, p5_out
