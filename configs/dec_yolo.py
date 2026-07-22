#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DEC-YOLO architecture config, YOLOv6 configs/*.py style: `depth_multiple`/`width_multiple`
scale the model built by yolov6.models.yolo.build_model(cfg, num_classes, device).

`width_multiple` scales every channel count in the backbone/neck/dec/head (all derived from
`backbone.base_channels`, see yolov6/models/backbone.py). `depth_multiple` only scales
DenseNetModule's repeat count inside the DEC module (yolov6/models/dec.py) — the paper's
ELAN/ELANH blocks are fixed 4-branch topologies, not repeat-based, so they aren't
depth-scaled. Defaults reproduce the paper's own architecture exactly.
"""

model = dict(
    type='DECYOLO',
    depth_multiple=1.0,
    width_multiple=1.0,
    backbone=dict(
        type='Backbone',
        base_channels=[32, 64, 128, 256, 512, 1024],
    ),
    neck=dict(
        type='Neck',
    ),
    dec=dict(
        type='DECModule',
        num_codewords=32,
    ),
    head=dict(
        type='EfficientDecoupledHead',
        num_classes=3,  # scratch, uneven_surface, contour_damage
    ),
)
