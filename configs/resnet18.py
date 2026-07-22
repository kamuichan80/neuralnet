#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""ResNet-18 config (He et al., 2015). `layers` is the BasicBlock repeat count per stage."""

model = dict(
    type='ResNet18',
    num_classes=1000,
    layers=[2, 2, 2, 2],
)
