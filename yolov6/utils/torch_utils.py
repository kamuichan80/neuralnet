#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Torch helpers shared across models: channel rounding, BN fusion, weight init."""

import torch
import torch.nn as nn


def make_divisible(x, divisor=8):
    """Round `x` to the nearest multiple of `divisor` (never below `divisor`)."""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


def fuse_conv_bn(conv, bn):
    """Fuse a Conv2d immediately followed by BatchNorm2d into a single Conv2d."""
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


def fuse_model(model):
    """Switch every RepVGGBlock in `model` to its re-parameterized deploy form, in place."""
    from yolov6.layers.common import RepVGGBlock

    for m in model.modules():
        if isinstance(m, RepVGGBlock):
            m.switch_to_deploy()
    return model


def initialize_weights(model):
    """BN eps/momentum and inplace activations, the usual YOLO init."""
    for m in model.modules():
        t = type(m)
        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
            m.inplace = True
