#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DEC-YOLO: Surface Defect Detection Algorithm for Laser Nozzles.
Li, S.; Deng, H.; Zhou, F.; Zheng, Y. Electronics 2025, 14, 1279.
https://doi.org/10.3390/electronics14071279

Assembles Backbone + Neck + (DECModule, EfficientDecoupledHead) per scale, and
`build_model(cfg, num_classes, device)`, mirroring YOLOv6's models/yolo.py entry point.
"""

import torch
import torch.nn as nn

from yolov6.models.backbone import Backbone
from yolov6.models.dec import DECModule
from yolov6.models.heads.effidehead import EfficientDecoupledHead
from yolov6.models.neck import Neck
from yolov6.utils.torch_utils import initialize_weights


class DECYOLO(nn.Module):
    """
    DEC-YOLO detection model. Returns one anchor-free prediction map per scale
    (strides 8, 16, 32), each with 4 (box) + 1 (objectness) + num_classes channels.
    """

    def __init__(self, num_classes=3, base_channels=(32, 64, 128, 256, 512, 1024),
                 width_multiple=1.0, depth_multiple=1.0, num_codewords=32):
        super().__init__()
        self.backbone = Backbone(base_channels, width_multiple)
        self.neck = Neck(self.backbone.out_channels)

        p3_ch, p4_ch, p5_ch = self.neck.out_channels
        # DenseNetModule's repeat count is DEC-YOLO's one depth-scalable dimension; ELAN/
        # ELANH are fixed 4-branch topologies and aren't repeat-based, so depth_multiple
        # doesn't touch them.
        num_layers = max(round(4 * depth_multiple), 1)

        self.dec_p3 = DECModule(p3_ch, p3_ch, num_layers=num_layers, num_codewords=num_codewords)
        self.dec_p4 = DECModule(p4_ch, p4_ch, num_layers=num_layers, num_codewords=num_codewords)
        self.dec_p5 = DECModule(p5_ch, p5_ch, num_layers=num_layers, num_codewords=num_codewords)

        self.head_p3 = EfficientDecoupledHead(p3_ch, num_classes)
        self.head_p4 = EfficientDecoupledHead(p4_ch, num_classes)
        self.head_p5 = EfficientDecoupledHead(p5_ch, num_classes)

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c2, c3, c4, c5)

        out_p3 = self.head_p3(self.dec_p3(p3))
        out_p4 = self.head_p4(self.dec_p4(p4))
        out_p5 = self.head_p5(self.dec_p5(p5))

        return out_p3, out_p4, out_p5


def build_model(cfg, num_classes=None, device=None):
    """Builds DECYOLO from a Config object (see configs/dec_yolo.py)."""
    model_cfg = cfg.model
    model = DECYOLO(
        num_classes=num_classes if num_classes is not None else model_cfg.head.num_classes,
        base_channels=model_cfg.backbone.base_channels,
        width_multiple=model_cfg.width_multiple,
        depth_multiple=model_cfg.depth_multiple,
        num_codewords=model_cfg.dec.num_codewords,
    )
    initialize_weights(model)
    return model.to(device) if device is not None else model


if __name__ == "__main__":
    # scratch, uneven surface, contour damage (the paper's laser-nozzle defect dataset)
    model = DECYOLO(num_classes=3)
    x = torch.randn(2, 3, 640, 640)
    outputs = model(x)
    for stride, out in zip((8, 16, 32), outputs):
        print(f"stride {stride}: {tuple(out.shape)}")
