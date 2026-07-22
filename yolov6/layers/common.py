#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Reusable conv blocks, YOLOv6 layers/common.py style.

ConvBNSiLU, MP1/MP2 and ELAN/ELANH are DEC-YOLO's YOLOv7-lineage blocks (Section 2.2 of the
paper), kept architecturally as-is. RepVGGBlock and CSPSPPF are swapped in for the paper's
RepConv/SPPCSPC to match YOLOv6's actual implementations:
  - RepVGGBlock: same train-time multi-branch idea as RepConv, plus YOLOv6's real
    re-parameterization (`switch_to_deploy()` fuses the 3x3/1x1/identity branches into one
    conv, exactly, not an approximation). Activation kept as SiLU to match the paper's
    YOLOv7 lineage (YOLOv6 itself defaults this block to ReLU).
  - CSPSPPF: the same 7-conv CSP topology as the paper's SPPCSPC, but replaces the three
    parallel differently-sized max-pools with YOLOv6's cheaper trick of applying one
    kernel-size max-pool sequentially three times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    return k // 2 if p is None else p


class ConvBNSiLU(nn.Module):
    """Conv-BN-SiLU, the basic building block of the backbone/neck (paper Section 2.2)."""

    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepVGGBlock(nn.Module):
    """
    RepVGG-style re-parameterizable conv (YOLOv6 layers/common.py): a 3x3 branch, a 1x1
    branch, and an optional identity branch are trained in parallel, then fused into a
    single conv at deploy time via `switch_to_deploy()`.
    """

    def __init__(self, in_ch, out_ch, k=3, s=1, groups=1, deploy=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.groups = groups
        self.deploy = deploy
        self.act = nn.SiLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_ch, out_ch, k, s, autopad(k), groups=groups, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_ch) if out_ch == in_ch and s == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, autopad(k), groups=groups, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, s, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel, bn = branch[0].weight, branch[1]
        else:  # identity BatchNorm2d: represent as a (groups-aware) identity kernel
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_ch // self.groups
                kernel_value = torch.zeros(
                    (self.in_ch, input_dim, self.k, self.k), device=branch.weight.device
                )
                for i in range(self.in_ch):
                    kernel_value[i, i % input_dim, self.k // 2, self.k // 2] = 1
                self.id_tensor = kernel_value
            kernel, bn = self.id_tensor, branch
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _pad_1x1_to_kxk(self, kernel1x1):
        if isinstance(kernel1x1, int):
            return 0
        pad = self.k // 2
        return F.pad(kernel1x1, [pad, pad, pad, pad])

    def get_equivalent_kernel_bias(self):
        kernel_dense, bias_dense = self._fuse_bn_tensor(self.rbr_dense)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        kernel = kernel_dense + self._pad_1x1_to_kxk(kernel_1x1) + kernel_id
        bias = bias_dense + bias_1x1 + bias_id
        return kernel, bias

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_ch, self.out_ch, self.k, self.s, autopad(self.k),
            groups=self.groups, bias=True,
        ).to(kernel.device)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for p in self.parameters():
            p.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class MP1(nn.Module):
    """MP downsample block (paper Figure 1): max-pool branch + strided-conv branch, channels unchanged."""

    def __init__(self, ch):
        super().__init__()
        half = ch // 2
        self.pool_branch = nn.Sequential(nn.MaxPool2d(2, 2), ConvBNSiLU(ch, half, 1, 1))
        self.conv_branch = nn.Sequential(ConvBNSiLU(ch, half, 1, 1), ConvBNSiLU(half, half, 3, 2))

    def forward(self, x):
        return torch.cat([self.pool_branch(x), self.conv_branch(x)], dim=1)


class MP2(MP1):
    """Same topology as MP1, used for the PAN neck downsampling stages (paper Figure 1)."""


class ELAN(nn.Module):
    """Efficient Layer Aggregation Network block used in the backbone (paper Section 2.2)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4
        self.cv1 = ConvBNSiLU(in_ch, mid, 1, 1)
        self.cv2 = ConvBNSiLU(in_ch, mid, 1, 1)
        self.cv3 = ConvBNSiLU(mid, mid, 3, 1)
        self.cv4 = ConvBNSiLU(mid, mid, 3, 1)
        self.cv5 = ConvBNSiLU(mid, mid, 3, 1)
        self.cv6 = ConvBNSiLU(mid, mid, 3, 1)
        self.cv_out = ConvBNSiLU(mid * 4, out_ch, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y3 = self.cv4(self.cv3(y2))
        y4 = self.cv6(self.cv5(y3))
        return self.cv_out(torch.cat([y1, y2, y3, y4], dim=1))


class ELANH(nn.Module):
    """ELAN-H block used in the PAN neck (paper Figure 1); same idea as ELAN, narrower branches."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 2
        branch = mid // 2
        self.cv1 = ConvBNSiLU(in_ch, mid, 1, 1)
        self.cv2 = ConvBNSiLU(in_ch, mid, 1, 1)
        self.cv3 = ConvBNSiLU(mid, branch, 3, 1)
        self.cv4 = ConvBNSiLU(branch, branch, 3, 1)
        self.cv5 = ConvBNSiLU(branch, branch, 3, 1)
        self.cv6 = ConvBNSiLU(branch, branch, 3, 1)
        self.cv_out = ConvBNSiLU(mid * 2 + branch * 2, out_ch, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y3 = self.cv4(self.cv3(y2))
        y4 = self.cv6(self.cv5(y3))
        return self.cv_out(torch.cat([y1, y2, y3, y4], dim=1))


class CSPSPPF(nn.Module):
    """
    CSP-style SPPF (YOLOv6 layers/common.py), ending the backbone (paper Section 2.2). Same
    7-conv cross-stage-partial topology as SPPCSPC, but pools sequentially instead of in
    parallel across multiple kernel sizes.
    """

    def __init__(self, in_ch, out_ch, kernel_size=5, e=0.5):
        super().__init__()
        mid = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, mid, 1, 1)
        self.cv2 = ConvBNSiLU(in_ch, mid, 1, 1)
        self.cv3 = ConvBNSiLU(mid, mid, 3, 1)
        self.cv4 = ConvBNSiLU(mid, mid, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size, 1, kernel_size // 2)
        self.cv5 = ConvBNSiLU(mid * 4, mid, 1, 1)
        self.cv6 = ConvBNSiLU(mid, mid, 3, 1)
        self.cv7 = ConvBNSiLU(mid * 2, out_ch, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.pool(x1)
        y2 = self.pool(y1)
        y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.pool(y2)], dim=1)))
        return self.cv7(torch.cat([y0, y3], dim=1))
