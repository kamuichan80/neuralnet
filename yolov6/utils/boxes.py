"""
Shared box utilities for DEC-YOLO's anchor-free head: grid-center construction, the
FCOS-style (l, t, r, b) -> xyxy decode, pairwise IoU, and greedy NMS. Used by both the
training loss (yolov6/core/loss.py) and the evaluator (yolov6/core/evaluator.py).
"""

import torch


def decode_grid(h, w, stride, device):
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'
    )
    cx = (xs.reshape(-1).float() + 0.5) * stride
    cy = (ys.reshape(-1).float() + 0.5) * stride
    return cx, cy


def decode_ltrb(reg, cx, cy, stride):
    """reg: (..., 4) raw (l, t, r, b) network outputs -> (..., 4) xyxy boxes in pixel space."""
    l, t, r, b = reg.exp().unbind(-1)
    x1 = cx - l * stride
    y1 = cy - t * stride
    x2 = cx + r * stride
    y2 = cy + b * stride
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1, boxes2, eps=1e-7):
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union


def nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS over a single class's boxes. Returns indices of kept boxes."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
