"""
Loss functions for DEC-YOLO's anchor-free efficient decoupling head (Section 3.3).

Each of the three head outputs is (B, 4 + 1 + num_classes, H, W): box regression as
FCOS-style (l, t, r, b) distances from the grid-cell center, objectness, and per-class
logits. Positives are assigned with FCOS's scale-based rule (center-in-box + a per-level
regression-distance range), which is a simpler stand-in for full SimOTA/OTA matching but
is enough to train the network end-to-end.
"""

import math

import torch
import torch.nn as nn


def bbox_ciou(pred, target, eps=1e-7):
    """Elementwise IoU and CIoU between two (N, 4) xyxy box tensors."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
    ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    parea = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    tarea = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = parea + tarea - inter + eps
    iou = inter / union

    cx1, cy1 = torch.min(px1, tx1), torch.min(py1, ty1)
    cx2, cy2 = torch.max(px2, tx2), torch.max(py2, ty2)
    c_diag = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    center_dist = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    pw, ph = (px2 - px1).clamp(min=eps), (py2 - py1).clamp(min=eps)
    tw, th = (tx2 - tx1).clamp(min=eps), (ty2 - ty1).clamp(min=eps)
    v = (4 / math.pi ** 2) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - center_dist / c_diag - alpha * v
    return iou, ciou


class DECYOLOLoss(nn.Module):
    def __init__(self, num_classes, strides=(8, 16, 32),
                 size_ranges=((0, 64), (64, 128), (128, 1e8)),
                 box_weight=5.0, obj_weight=1.0, cls_weight=1.0):
        super().__init__()
        assert len(strides) == len(size_ranges)
        self.num_classes = num_classes
        self.strides = strides
        self.size_ranges = size_ranges
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    @staticmethod
    def _grid_centers(h, w, stride, device):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'
        )
        cx = (xs.float() + 0.5) * stride
        cy = (ys.float() + 0.5) * stride
        return cx.reshape(-1), cy.reshape(-1)

    def forward(self, preds, targets):
        """
        preds: list of 3 tensors (B, 5 + num_classes, H, W), one per stride.
        targets: (M, 6) tensor of [image_idx, class_id, x1, y1, x2, y2] in input-pixel space.
        """
        device = preds[0].device
        batch_size = preds[0].shape[0]

        cx_list, cy_list, stride_list, level_sizes = [], [], [], []
        for stride, (lo, hi), p in zip(self.strides, self.size_ranges, preds):
            _, _, h, w = p.shape
            cx, cy = self._grid_centers(h, w, stride, device)
            cx_list.append(cx)
            cy_list.append(cy)
            stride_list.append(torch.full_like(cx, stride))
            level_sizes.append((lo, hi, cx.shape[0]))

        cx = torch.cat(cx_list)
        cy = torch.cat(cy_list)
        stride_per_point = torch.cat(stride_list)
        num_points = cx.shape[0]

        flat_preds = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(batch_size, -1, p.shape[1]) for p in preds], dim=1
        )
        reg_pred = flat_preds[..., :4]
        obj_pred = flat_preds[..., 4]
        cls_pred = flat_preds[..., 5:]

        min_reg_range = torch.cat([torch.full((n,), lo, device=device) for lo, hi, n in level_sizes])
        max_reg_range = torch.cat([torch.full((n,), hi, device=device) for lo, hi, n in level_sizes])

        obj_target = torch.zeros(batch_size, num_points, device=device)
        cls_target = torch.zeros(batch_size, num_points, self.num_classes, device=device)
        box_target = torch.zeros(batch_size, num_points, 4, device=device)
        pos_mask = torch.zeros(batch_size, num_points, dtype=torch.bool, device=device)

        for b in range(batch_size):
            gt = targets[targets[:, 0] == b]
            if gt.numel() == 0:
                continue
            gt_boxes = gt[:, 2:6]      # (G, 4) xyxy
            gt_cls = gt[:, 1].long()   # (G,)

            l = cx[:, None] - gt_boxes[None, :, 0]
            t = cy[:, None] - gt_boxes[None, :, 1]
            r = gt_boxes[None, :, 2] - cx[:, None]
            btm = gt_boxes[None, :, 3] - cy[:, None]
            ltrb = torch.stack([l, t, r, btm], dim=-1)  # (A, G, 4)

            inside_box = ltrb.min(dim=-1).values > 0
            max_ltrb = ltrb.max(dim=-1).values
            inside_range = (max_ltrb >= min_reg_range[:, None]) & (max_ltrb <= max_reg_range[:, None])
            valid = inside_box & inside_range  # (A, G)

            gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            area_for_points = gt_area[None, :].expand(num_points, -1).clone()
            area_for_points[~valid] = float('inf')

            min_area, best_gt = area_for_points.min(dim=1)
            has_positive = torch.isfinite(min_area)

            pos_mask[b] = has_positive
            obj_target[b] = has_positive.float()

            pos_idx = has_positive.nonzero(as_tuple=True)[0]
            if pos_idx.numel():
                matched_gt = best_gt[pos_idx]
                box_target[b, pos_idx] = gt_boxes[matched_gt]
                cls_target[b, pos_idx, gt_cls[matched_gt]] = 1.0

        l, t, r, btm = reg_pred.exp().unbind(-1)
        px1 = cx[None] - l * stride_per_point[None]
        py1 = cy[None] - t * stride_per_point[None]
        px2 = cx[None] + r * stride_per_point[None]
        py2 = cy[None] + btm * stride_per_point[None]
        box_pred = torch.stack([px1, py1, px2, py2], dim=-1)

        num_pos = pos_mask.sum().clamp(min=1)

        obj_loss = self.bce(obj_pred, obj_target).mean()

        if pos_mask.any():
            cls_loss = self.bce(cls_pred[pos_mask], cls_target[pos_mask]).sum() / num_pos
            _, ciou = bbox_ciou(box_pred[pos_mask], box_target[pos_mask])
            box_loss = (1 - ciou).sum() / num_pos
        else:
            # keep the graph connected to reg/cls heads even on an all-background batch
            cls_loss = cls_pred.sum() * 0
            box_loss = reg_pred.sum() * 0

        total = self.box_weight * box_loss + self.obj_weight * obj_loss + self.cls_weight * cls_loss
        return total, {
            'box_loss': box_loss.detach(),
            'obj_loss': obj_loss.detach(),
            'cls_loss': cls_loss.detach(),
            'num_pos': num_pos.detach().float() / batch_size,
        }
