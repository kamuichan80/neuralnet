#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""VGG16 (Simonyan & Zisserman, 2014), config-driven build."""

import torch
import torch.nn as nn


def _make_features(cfgs):
    """Builds the conv stack from a torchvision-style channel list: an int is a
    Conv2d(out_channels) + ReLU, 'M' is a 2x2 max-pool."""
    layers = []
    in_ch = 3
    for v in cfgs:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


class VGG16Net(nn.Module):
    def __init__(self, num_classes=1000, cfgs=None):
        super().__init__()
        cfgs = cfgs or [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = _make_features(cfgs)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(cfg):
    return VGG16Net(num_classes=cfg.model.num_classes, cfgs=cfg.model.cfgs)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from yolov6.utils.config import Config

    cfg = Config.fromfile(ROOT / 'configs' / 'vgg16net.py')
    print(build_model(cfg))
