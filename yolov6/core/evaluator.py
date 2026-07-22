#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
mAP evaluation logic for DEC-YOLO (paper Section 4.2.1).

Decodes the anchor-free head outputs into boxes, runs per-class NMS, greedily matches
predictions to ground truth by descending confidence at each of 10 IoU thresholds
(0.5:0.05:0.95, Eq. 16), and integrates precision-recall curves into AP (Eq. 16-17) plus
the paper's precision/recall/F1 (Eq. 13-15) taken at each class's best-F1 operating point.
"""

import torch

from yolov6.utils.boxes import box_iou, decode_grid, decode_ltrb, nms


@torch.no_grad()
def decode_predictions(preds, strides=(8, 16, 32)):
    """
    preds: list of 3 tensors (B, 5 + num_classes, H, W).
    Returns (boxes, scores): (B, A, 4) xyxy boxes and (B, A, num_classes) confidences
    (objectness * per-class probability), pooled across all three scales.
    """
    device = preds[0].device
    box_chunks, score_chunks = [], []
    for stride, p in zip(strides, preds):
        b, ch, h, w = p.shape
        p = p.permute(0, 2, 3, 1).reshape(b, h * w, ch)
        cx, cy = decode_grid(h, w, stride, device)
        boxes = decode_ltrb(p[..., :4], cx[None], cy[None], stride)  # (B, HW, 4)

        obj = torch.sigmoid(p[..., 4])
        cls = torch.sigmoid(p[..., 5:])
        scores = obj.unsqueeze(-1) * cls  # (B, HW, num_classes)

        box_chunks.append(boxes)
        score_chunks.append(scores)

    return torch.cat(box_chunks, dim=1), torch.cat(score_chunks, dim=1)


@torch.no_grad()
def postprocess(boxes, scores, conf_thres=0.001, iou_thres=0.65, max_det=300):
    """
    Per-image, per-class confidence filtering + NMS.
    boxes: (A, 4), scores: (A, num_classes). Returns (boxes, scores, class_ids).
    """
    num_classes = scores.shape[-1]
    out_boxes, out_scores, out_classes = [], [], []

    for c in range(num_classes):
        c_scores = scores[:, c]
        mask = c_scores > conf_thres
        if not mask.any():
            continue
        c_boxes, c_scores = boxes[mask], c_scores[mask]
        keep = nms(c_boxes, c_scores, iou_thres)
        out_boxes.append(c_boxes[keep])
        out_scores.append(c_scores[keep])
        out_classes.append(torch.full((keep.numel(),), c, dtype=torch.long))

    if not out_boxes:
        return boxes.new_zeros((0, 4)), boxes.new_zeros((0,)), torch.zeros((0,), dtype=torch.long)

    boxes = torch.cat(out_boxes)
    scores = torch.cat(out_scores)
    classes = torch.cat(out_classes)
    if boxes.shape[0] > max_det:
        top = scores.argsort(descending=True)[:max_det]
        boxes, scores, classes = boxes[top], scores[top], classes[top]
    return boxes, scores, classes


def compute_ap(recall, precision):
    """All-point interpolated AP: precision envelope + area under the P-R curve."""
    mrec = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    mpre = torch.cat([torch.ones(1), precision, torch.zeros(1)])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=True)[0]
    return torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()


def match_predictions(pred_boxes, pred_scores, pred_img_ids, gt_boxes, gt_img_ids, iou_threshold):
    """
    Greedily matches predictions of ONE class (sorted by descending confidence) against
    ground-truth boxes of the SAME class, image by image, at a single IoU threshold. Each
    GT box can be matched at most once. Returns a bool tensor tp aligned with `order`, and
    `order` itself so callers can re-sort scores/boxes consistently.
    """
    order = pred_scores.argsort(descending=True)
    sorted_boxes = pred_boxes[order]
    sorted_img_ids = pred_img_ids[order]

    tp = torch.zeros(sorted_boxes.shape[0], dtype=torch.bool)
    matched = {}
    for img_id in gt_img_ids.unique().tolist():
        matched[img_id] = torch.zeros((gt_img_ids == img_id).sum().item(), dtype=torch.bool)

    for i in range(sorted_boxes.shape[0]):
        img_id = sorted_img_ids[i].item()
        if img_id not in matched:
            continue
        gt_mask = gt_img_ids == img_id
        candidate_gt = gt_boxes[gt_mask]

        ious = box_iou(sorted_boxes[i].unsqueeze(0), candidate_gt).squeeze(0)
        ious = ious.masked_fill(matched[img_id], 0.0)
        best_iou, best_j = ious.max(dim=0)
        if best_iou >= iou_threshold:
            tp[i] = True
            matched[img_id][best_j] = True

    return tp, order


def evaluate_class(pred_boxes, pred_scores, pred_img_ids, gt_boxes, gt_img_ids, iou_thresholds):
    """Returns ({iou_threshold: AP}, precision, recall, f1) for one class, the latter three
    taken at the best-F1 point of the IoU=0.5 precision-recall curve."""
    num_gt = gt_boxes.shape[0]
    if num_gt == 0 or pred_boxes.shape[0] == 0:
        return {iou: 0.0 for iou in iou_thresholds}, 0.0, 0.0, 0.0

    aps = {}
    precision_at_50 = recall_at_50 = f1_at_50 = 0.0
    for iou_thr in iou_thresholds:
        tp, order = match_predictions(pred_boxes, pred_scores, pred_img_ids, gt_boxes, gt_img_ids, iou_thr)

        tp_cumsum = torch.cumsum(tp.float(), dim=0)
        fp_cumsum = torch.cumsum((~tp).float(), dim=0)
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-9)

        aps[iou_thr] = compute_ap(recall, precision)

        if abs(iou_thr - 0.5) < 1e-6:
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
            if f1.numel():
                best = f1.argmax()
                precision_at_50, recall_at_50, f1_at_50 = precision[best].item(), recall[best].item(), f1[best].item()

    return aps, precision_at_50, recall_at_50, f1_at_50


@torch.no_grad()
def evaluate(model, loader, num_classes, device, conf_thres=0.001, iou_thres_nms=0.65, class_names=None):
    """Runs the model over `loader` and computes mAP@0.5, mAP@0.5:0.95, precision, recall
    and F1 (mean over classes present in the ground truth), plus a per-class breakdown."""
    model.eval()
    class_names = class_names or [str(c) for c in range(num_classes)]

    pred_boxes_l, pred_scores_l, pred_classes_l, pred_img_ids_l = [], [], [], []
    gt_boxes_l, gt_classes_l, gt_img_ids_l = [], [], []

    img_id = 0
    for images, targets in loader:
        images = images.to(device)
        preds = model(images)
        boxes, scores = decode_predictions(preds)

        for b in range(images.shape[0]):
            det_boxes, det_scores, det_classes = postprocess(boxes[b], scores[b], conf_thres, iou_thres_nms)
            n = det_boxes.shape[0]
            pred_boxes_l.append(det_boxes.cpu())
            pred_scores_l.append(det_scores.cpu())
            pred_classes_l.append(det_classes.cpu())
            pred_img_ids_l.append(torch.full((n,), img_id, dtype=torch.long))

            gt = targets[targets[:, 0] == b]
            gt_boxes_l.append(gt[:, 2:6])
            gt_classes_l.append(gt[:, 1].long())
            gt_img_ids_l.append(torch.full((gt.shape[0],), img_id, dtype=torch.long))

            img_id += 1

    pred_boxes = torch.cat(pred_boxes_l) if pred_boxes_l else torch.zeros((0, 4))
    pred_scores = torch.cat(pred_scores_l) if pred_scores_l else torch.zeros((0,))
    pred_classes = torch.cat(pred_classes_l) if pred_classes_l else torch.zeros((0,), dtype=torch.long)
    pred_img_ids = torch.cat(pred_img_ids_l) if pred_img_ids_l else torch.zeros((0,), dtype=torch.long)

    gt_boxes = torch.cat(gt_boxes_l) if gt_boxes_l else torch.zeros((0, 4))
    gt_classes = torch.cat(gt_classes_l) if gt_classes_l else torch.zeros((0,), dtype=torch.long)
    gt_img_ids = torch.cat(gt_img_ids_l) if gt_img_ids_l else torch.zeros((0,), dtype=torch.long)

    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.5:0.05:0.95, Eq. 16

    per_class = {}
    for c in range(num_classes):
        p_mask = pred_classes == c
        g_mask = gt_classes == c
        num_gt = int(g_mask.sum())

        aps, precision, recall, f1 = evaluate_class(
            pred_boxes[p_mask], pred_scores[p_mask], pred_img_ids[p_mask],
            gt_boxes[g_mask], gt_img_ids[g_mask], iou_thresholds,
        )
        per_class[class_names[c]] = {
            'num_gt': num_gt,
            'AP@0.5': aps[0.5],
            'AP@0.5:0.95': sum(aps.values()) / len(aps),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    present = [v for v in per_class.values() if v['num_gt'] > 0]
    n = max(len(present), 1)
    summary = {
        'mAP@0.5': sum(v['AP@0.5'] for v in present) / n,
        'mAP@0.5:0.95': sum(v['AP@0.5:0.95'] for v in present) / n,
        'precision': sum(v['precision'] for v in present) / n,
        'recall': sum(v['recall'] for v in present) / n,
        'f1': sum(v['f1'] for v in present) / n,
    }

    return {'per_class': per_class, **summary}
