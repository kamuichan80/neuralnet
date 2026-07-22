#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DEC module: DenseNet branch + explicit visual center (EVC) branch (paper Section 3.1).
No YOLOv6 equivalent exists for this — it's the paper's own contribution — so only its
file location and naming follow the repo's YOLOv6-style layout, not its internals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.layers.common import ConvBNSiLU


class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.conv = ConvBNSiLU(in_ch, growth_rate, 3, 1)

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)


class DenseNetModule(nn.Module):
    """
    Dense block where every layer is concatenated with all previous outputs (Eq. 1,
    xn = fn([x1, ..., xn-1])), run alongside a plain convolutional side branch; the two
    are fused by concatenation (Figure 3). `num_layers` is DEC-YOLO's one depth-scalable
    count (see configs/dec_yolo.py's `depth_multiple`).
    """

    def __init__(self, in_ch, out_ch, num_layers=4, growth_rate=None):
        super().__init__()
        growth_rate = growth_rate or max(in_ch // num_layers, 8)
        self.stem = ConvBNSiLU(in_ch, in_ch, 1, 1)
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.dense_block = nn.Sequential(*layers)
        self.side_branch = ConvBNSiLU(in_ch, in_ch, 1, 1)
        self.fuse = ConvBNSiLU(ch + in_ch, out_ch, 1, 1)

    def forward(self, x):
        dense_out = self.dense_block(self.stem(x))
        side_out = self.side_branch(x)
        return self.fuse(torch.cat([dense_out, side_out], dim=1))


class DepthwiseConvModule(nn.Module):
    """First residual block of the lightweight MLP (Eq. 2): X' = DConv(GN(Xin)) + Xin."""

    def __init__(self, ch):
        super().__init__()
        self.gn = nn.GroupNorm(1, ch)
        self.dconv = nn.Conv2d(ch, ch, kernel_size=1, groups=ch, bias=True)
        self.gamma = nn.Parameter(1e-2 * torch.ones(ch))

    def forward(self, x):
        out = self.dconv(self.gn(x)) * self.gamma.view(1, -1, 1, 1)
        return x + out


class ChannelMLPModule(nn.Module):
    """Second residual block of the lightweight MLP (Eq. 3, 4): MLP(Xin) = CMLP(GN(X')) + X'."""

    def __init__(self, ch, r=4):
        super().__init__()
        self.gn = nn.GroupNorm(1, ch)
        hidden = max(ch // r, 4)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, ch, 1)
        self.gamma = nn.Parameter(1e-2 * torch.ones(ch))

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(self.gn(x)))) * self.gamma.view(1, -1, 1, 1)
        return x + out


class LightweightMLP(nn.Module):
    """Lightweight MLP branch of the EVC module (Figure 5): models global channel dependencies."""

    def __init__(self, ch, r=4):
        super().__init__()
        self.dwconv = DepthwiseConvModule(ch)
        self.cmlp = ChannelMLPModule(ch, r)

    def forward(self, x):
        return self.cmlp(self.dwconv(x))


class LVC(nn.Module):
    """
    Learnable Visual Center (Figure 6). Encodes local features against a learnable codebook
    of K prototypes with per-codeword scaling factors, then uses the aggregated descriptor to
    gate the input feature map so the model emphasizes regions matching salient local patterns.
    """

    def __init__(self, ch, num_codewords=32):
        super().__init__()
        self.encode_conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.codebook = nn.Parameter(torch.empty(num_codewords, ch))
        self.scale = nn.Parameter(torch.empty(num_codewords))
        nn.init.normal_(self.codebook, 0, 1.0 / (num_codewords * ch) ** 0.5)
        nn.init.uniform_(self.scale, -1, 0)

        self.gate = nn.Sequential(
            nn.Linear(ch, ch),
            nn.ReLU(inplace=True),
            nn.Linear(ch, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.encode_conv(x)
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, N, C)

        residual = tokens.unsqueeze(2) - self.codebook.view(1, 1, -1, c)  # (B, N, K, C)
        dist = residual.pow(2).sum(-1)  # (B, N, K)
        assign = torch.softmax(self.scale.view(1, 1, -1) * dist, dim=-1)  # soft assignment to codewords
        encoded = torch.einsum('bnk,bnkc->bkc', assign, residual)  # (B, K, C), aggregated per codeword
        descriptor = F.relu(encoded, inplace=True).mean(dim=1)  # (B, C)

        gain = self.gate(descriptor).view(b, c, 1, 1)
        return x * gain


class EVC(nn.Module):
    """Explicit Visual Center (Figure 4): stem block feeding a lightweight MLP and an LVC
    branch in parallel, fused by concatenation (Eq. 5, 6)."""

    def __init__(self, in_ch, out_ch, num_codewords=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp_branch = LightweightMLP(out_ch)
        self.lvc_branch = LVC(out_ch, num_codewords)
        self.fuse = ConvBNSiLU(out_ch * 2, out_ch, 1, 1)

    def forward(self, x):
        stem_out = self.stem(x)
        return self.fuse(torch.cat([self.mlp_branch(stem_out), self.lvc_branch(stem_out)], dim=1))


class DECModule(nn.Module):
    """
    DEC module (DenseNet-EVC composite, Figure 2): fuses a DenseNet branch (local feature
    reuse, stable gradient propagation) with an EVC branch (global + local salient-region
    modeling) by concatenation. Applied before each detection head.
    """

    def __init__(self, in_ch, out_ch, num_layers=4, num_codewords=32):
        super().__init__()
        half = out_ch // 2
        self.dense_branch = DenseNetModule(in_ch, half, num_layers=num_layers)
        self.evc_branch = EVC(in_ch, half, num_codewords)
        self.fuse = ConvBNSiLU(half * 2, out_ch, 1, 1)

    def forward(self, x):
        return self.fuse(torch.cat([self.dense_branch(x), self.evc_branch(x)], dim=1))
