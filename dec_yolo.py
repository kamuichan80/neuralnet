"""
DEC-YOLO: Surface Defect Detection Algorithm for Laser Nozzles.
Li, S.; Deng, H.; Zhou, F.; Zheng, Y. Electronics 2025, 14, 1279.
https://doi.org/10.3390/electronics14071279

YOLOv7 baseline extended with:
  - DEC module: a DenseNet branch fused with an explicit visual center (EVC) branch,
    itself made of a lightweight MLP (global dependencies) and a learnable visual
    center (LVC, local salient regions), applied before each detection head.
  - Cross-layer connection: an extra shallow-to-deep skip (beyond adjacent-level PAN
    connections) so low-level texture/edge information reaches the deeper neck nodes.
  - Efficient decoupling head: an anchor-free decoupled head with an asymmetric
    hybrid-channel design (1 conv for classification, 2 for regression/objectness).
  - Coordinate attention (CA) applied to the deepest backbone feature map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    return k // 2 if p is None else p


class CBS(nn.Module):
    """Conv-BN-SiLU, the basic building block of YOLOv7 (Section 2.2)."""

    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """Re-parameterizable multi-branch conv used at the head (Section 2.2), train-time form."""

    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, autopad(k), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, s, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.identity = nn.BatchNorm2d(in_ch) if in_ch == out_ch and s == 1 else None
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        id_out = self.identity(x) if self.identity is not None else 0
        return self.act(self.dense(x) + self.conv1x1(x) + id_out)


class MP1(nn.Module):
    """MP downsample block (Figure 1): max-pool branch + strided-conv branch, channels unchanged."""

    def __init__(self, ch):
        super().__init__()
        half = ch // 2
        self.pool_branch = nn.Sequential(nn.MaxPool2d(2, 2), CBS(ch, half, 1, 1))
        self.conv_branch = nn.Sequential(CBS(ch, half, 1, 1), CBS(half, half, 3, 2))

    def forward(self, x):
        return torch.cat([self.pool_branch(x), self.conv_branch(x)], dim=1)


class MP2(MP1):
    """Same topology as MP1, used for the PAN neck downsampling stages (Figure 1)."""


class ELAN(nn.Module):
    """Efficient Layer Aggregation Network block used in the backbone (Section 2.2)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4
        self.cv1 = CBS(in_ch, mid, 1, 1)
        self.cv2 = CBS(in_ch, mid, 1, 1)
        self.cv3 = CBS(mid, mid, 3, 1)
        self.cv4 = CBS(mid, mid, 3, 1)
        self.cv5 = CBS(mid, mid, 3, 1)
        self.cv6 = CBS(mid, mid, 3, 1)
        self.cv_out = CBS(mid * 4, out_ch, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y3 = self.cv4(self.cv3(y2))
        y4 = self.cv6(self.cv5(y3))
        return self.cv_out(torch.cat([y1, y2, y3, y4], dim=1))


class ELANH(nn.Module):
    """ELAN-H block used in the PAN neck (Figure 1); same idea as ELAN, narrower branches."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 2
        branch = mid // 2
        self.cv1 = CBS(in_ch, mid, 1, 1)
        self.cv2 = CBS(in_ch, mid, 1, 1)
        self.cv3 = CBS(mid, branch, 3, 1)
        self.cv4 = CBS(branch, branch, 3, 1)
        self.cv5 = CBS(branch, branch, 3, 1)
        self.cv6 = CBS(branch, branch, 3, 1)
        self.cv_out = CBS(mid * 2 + branch * 2, out_ch, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y3 = self.cv4(self.cv3(y2))
        y4 = self.cv6(self.cv5(y3))
        return self.cv_out(torch.cat([y1, y2, y3, y4], dim=1))


class SPPCSPC(nn.Module):
    """Spatial pyramid pooling + cross-stage partial connection, ending the backbone (Section 2.2)."""

    def __init__(self, in_ch, out_ch, pool_sizes=(5, 9, 13)):
        super().__init__()
        mid = out_ch // 2
        self.cv1 = CBS(in_ch, mid, 1, 1)
        self.cv2 = CBS(in_ch, mid, 1, 1)
        self.cv3 = CBS(mid, mid, 3, 1)
        self.cv4 = CBS(mid, mid, 1, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in pool_sizes])
        self.cv5 = CBS(mid * (len(pool_sizes) + 1), mid, 1, 1)
        self.cv6 = CBS(mid, mid, 3, 1)
        self.cv7 = CBS(mid * 2, out_ch, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x1 = self.cv6(self.cv5(torch.cat([x1] + [p(x1) for p in self.pools], dim=1)))
        x2 = self.cv2(x)
        return self.cv7(torch.cat([x1, x2], dim=1))


# --------------------------------------------------------------------------------------
# DEC module: DenseNet branch + explicit visual center (EVC) branch (Section 3.1)
# --------------------------------------------------------------------------------------

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.conv = CBS(in_ch, growth_rate, 3, 1)

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)


class DenseNetModule(nn.Module):
    """
    Dense block where every layer is concatenated with all previous outputs (Eq. 1,
    xn = fn([x1, ..., xn-1])), run alongside a plain convolutional side branch; the two
    are fused by concatenation (Figure 3).
    """

    def __init__(self, in_ch, out_ch, num_layers=4, growth_rate=None):
        super().__init__()
        growth_rate = growth_rate or max(in_ch // num_layers, 8)
        self.stem = CBS(in_ch, in_ch, 1, 1)
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.dense_block = nn.Sequential(*layers)
        self.side_branch = CBS(in_ch, in_ch, 1, 1)
        self.fuse = CBS(ch + in_ch, out_ch, 1, 1)

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
        self.fuse = CBS(out_ch * 2, out_ch, 1, 1)

    def forward(self, x):
        stem_out = self.stem(x)
        return self.fuse(torch.cat([self.mlp_branch(stem_out), self.lvc_branch(stem_out)], dim=1))


class DECModule(nn.Module):
    """
    DEC module (DenseNet-EVC composite, Figure 2): fuses a DenseNet branch (local feature
    reuse, stable gradient propagation) with an EVC branch (global + local salient-region
    modeling) by concatenation. Applied before each detection head.
    """

    def __init__(self, in_ch, out_ch, num_codewords=32):
        super().__init__()
        half = out_ch // 2
        self.dense_branch = DenseNetModule(in_ch, half)
        self.evc_branch = EVC(in_ch, half, num_codewords)
        self.fuse = CBS(half * 2, out_ch, 1, 1)

    def forward(self, x):
        return self.fuse(torch.cat([self.dense_branch(x), self.evc_branch(x)], dim=1))


# --------------------------------------------------------------------------------------
# Coordinate attention (Section 3.4)
# --------------------------------------------------------------------------------------

class CoordinateAttention(nn.Module):
    """Encodes long-range dependencies along the H and W axes separately (Eq. 7-9), then
    re-weights the input with direction-aware attention gates (Eq. 10-12)."""

    def __init__(self, ch, reduction=32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, ch, 1)
        self.conv_w = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = self.pool_h(x)  # (B, C, H, 1), Eq. 7
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1), Eq. 8

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))  # Eq. 9

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        g_h = torch.sigmoid(self.conv_h(x_h))  # Eq. 10
        g_w = torch.sigmoid(self.conv_w(x_w))  # Eq. 11
        return x * g_h * g_w  # Eq. 12


# --------------------------------------------------------------------------------------
# Efficient decoupling head (Section 3.3)
# --------------------------------------------------------------------------------------

class EfficientDecoupledHead(nn.Module):
    """
    Anchor-free, hybrid-channel decoupled head (Figure 7b): the classification branch uses
    a single 3x3 conv while the shared regression/objectness branch uses two stacked 3x3
    convs, before splitting into box regression, objectness and classification predictions.
    """

    def __init__(self, in_ch, num_classes, width=None):
        super().__init__()
        width = width or in_ch // 2
        self.stem = RepConv(in_ch, width, 3, 1)

        self.cls_conv = CBS(width, width, 3, 1)
        self.cls_pred = nn.Conv2d(width, num_classes, 1)

        self.reg_conv = nn.Sequential(CBS(width, width, 3, 1), CBS(width, width, 3, 1))
        self.reg_pred = nn.Conv2d(width, 4, 1)
        self.obj_pred = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        feat = self.stem(x)
        cls_out = self.cls_pred(self.cls_conv(feat))
        reg_feat = self.reg_conv(feat)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)
        return torch.cat([reg_out, obj_out, cls_out], dim=1)


# --------------------------------------------------------------------------------------
# Backbone, neck (with cross-layer connection) and full DEC-YOLO model
# --------------------------------------------------------------------------------------

class Backbone(nn.Module):
    """YOLOv7-style backbone with coordinate attention before the final SPPCSPC (Section 2.2, 3.4)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(CBS(3, 32, 3, 1), CBS(32, 64, 3, 2), CBS(64, 64, 3, 1))
        self.down1 = CBS(64, 128, 3, 2)
        self.elan1 = ELAN(128, 256)  # stride 4

        self.mp1 = MP1(256)
        self.elan2 = ELAN(256, 512)  # stride 8

        self.mp2 = MP1(512)
        self.elan3 = ELAN(512, 1024)  # stride 16

        self.mp3 = MP1(1024)
        self.elan4 = ELAN(1024, 1024)  # stride 32
        self.ca = CoordinateAttention(1024)
        self.sppcspc = SPPCSPC(1024, 512)

    def forward(self, x):
        x = self.elan1(self.down1(self.stem(x)))
        c2 = x  # stride 4, 256ch: shallow feature used by the cross-layer connection

        x = self.elan2(self.mp1(x))
        c3 = x  # stride 8, 512ch

        x = self.elan3(self.mp2(x))
        c4 = x  # stride 16, 1024ch

        x = self.ca(self.elan4(self.mp3(x)))
        c5 = self.sppcspc(x)  # stride 32, 512ch

        return c2, c3, c4, c5


class Neck(nn.Module):
    """
    PAN neck extended with a cross-layer connection (Section 3.2): besides the usual
    adjacent-level lateral skips, the shallow stride-4 backbone feature (c2) is downsampled
    and spliced into both top-down fusion nodes so low-level edge/texture detail, which would
    otherwise be diluted by repeated up-sampling, reaches the deeper feature maps directly.
    """

    def __init__(self):
        super().__init__()
        self.reduce_p5 = CBS(512, 256, 1, 1)
        self.lateral_c4 = CBS(1024, 256, 1, 1)
        self.cross_c2_to_p4 = nn.Sequential(CBS(256, 128, 3, 2), CBS(128, 128, 3, 2))
        self.elanh_p4 = ELANH(256 + 256 + 128, 256)

        self.reduce_p4 = CBS(256, 128, 1, 1)
        self.lateral_c3 = CBS(512, 128, 1, 1)
        self.cross_c2_to_p3 = CBS(256, 64, 3, 2)
        self.elanh_p3 = ELANH(128 + 128 + 64, 128)

        self.mp2_p3 = MP2(128)
        self.elanh_p4_out = ELANH(128 + 256, 256)

        self.mp2_p4 = MP2(256)
        self.elanh_p5_out = ELANH(256 + 512, 512)

    def forward(self, c2, c3, c4, c5):
        p5 = c5

        up_p5 = F.interpolate(self.reduce_p5(p5), scale_factor=2, mode='nearest')
        p4_td = self.elanh_p4(torch.cat([up_p5, self.lateral_c4(c4), self.cross_c2_to_p4(c2)], dim=1))

        up_p4 = F.interpolate(self.reduce_p4(p4_td), scale_factor=2, mode='nearest')
        p3_out = self.elanh_p3(torch.cat([up_p4, self.lateral_c3(c3), self.cross_c2_to_p3(c2)], dim=1))

        p4_out = self.elanh_p4_out(torch.cat([self.mp2_p3(p3_out), p4_td], dim=1))
        p5_out = self.elanh_p5_out(torch.cat([self.mp2_p4(p4_out), p5], dim=1))

        return p3_out, p4_out, p5_out


class DECYOLO(nn.Module):
    """
    DEC-YOLO detection model. Returns one anchor-free prediction map per scale
    (strides 8, 16, 32), each with 4 (box) + 1 (objectness) + num_classes channels.
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()

        self.dec_p3 = DECModule(128, 128)
        self.dec_p4 = DECModule(256, 256)
        self.dec_p5 = DECModule(512, 512)

        self.head_p3 = EfficientDecoupledHead(128, num_classes)
        self.head_p4 = EfficientDecoupledHead(256, num_classes)
        self.head_p5 = EfficientDecoupledHead(512, num_classes)

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c2, c3, c4, c5)

        out_p3 = self.head_p3(self.dec_p3(p3))
        out_p4 = self.head_p4(self.dec_p4(p4))
        out_p5 = self.head_p5(self.dec_p5(p5))

        return out_p3, out_p4, out_p5


if __name__ == "__main__":
    # scratch, uneven surface, contour damage (the paper's laser-nozzle defect dataset)
    model = DECYOLO(num_classes=3)
    x = torch.randn(2, 3, 640, 640)
    outputs = model(x)
    for stride, out in zip((8, 16, 32), outputs):
        print(f"stride {stride}: {tuple(out.shape)}")
