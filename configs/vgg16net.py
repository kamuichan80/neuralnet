#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""VGG16 config (Simonyan & Zisserman, 2014). `cfgs` is the conv-stack channel list, in
torchvision's own vgg.py style: an int is a Conv2d(out_channels) + ReLU, 'M' is a 2x2 max-pool."""

model = dict(
    type='VGG16Net',
    num_classes=1000,
    cfgs=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
)
