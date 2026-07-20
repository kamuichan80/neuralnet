"""
mAP evaluation for DEC-YOLO on a YOLO-format validation (or test) split (Section 4.2.1).

Reports, per class and averaged: AP@0.5, AP@0.5:0.95 (COCO-style, IoU 0.5:0.05:0.95),
plus precision, recall and F1-score — the same metrics Table 6 of the paper reports.

Example:
    python evaluate.py --data /path/to/laser_nozzle_dataset --weights runs/train/epoch_199.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from datasets import CLASS_NAMES, LaserNozzleDataset, collate_fn
from dec_yolo import DECYOLO
from metrics import evaluate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='dataset root; see datasets.py for layout')
    p.add_argument('--weights', required=True, help='checkpoint saved by train.py')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--num-classes', type=int, default=len(CLASS_NAMES))
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--conf-thres', type=float, default=0.001, help='pre-NMS confidence threshold')
    p.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_model(weights_path, num_classes, device):
    model = DECYOLO(num_classes=num_classes).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model


def print_results(results):
    header = f"{'class':<18}{'images':>8}{'AP@0.5':>10}{'AP@.5:.95':>12}{'P':>8}{'R':>8}{'F1':>8}"
    print(header)
    print('-' * len(header))
    for name, m in results['per_class'].items():
        print(f"{name:<18}{m['num_gt']:>8d}{m['AP@0.5']:>10.4f}{m['AP@0.5:0.95']:>12.4f}"
              f"{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}")
    print('-' * len(header))
    total_gt = sum(m['num_gt'] for m in results['per_class'].values())
    print(f"{'all':<18}{total_gt:>8d}{results['mAP@0.5']:>10.4f}{results['mAP@0.5:0.95']:>12.4f}"
          f"{results['precision']:>8.4f}{results['recall']:>8.4f}{results['f1']:>8.4f}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = LaserNozzleDataset(args.data, split=args.split, img_size=args.img_size, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    model = load_model(args.weights, args.num_classes, device)

    results = evaluate(
        model, loader, args.num_classes, device,
        conf_thres=args.conf_thres, iou_thres_nms=args.iou_thres,
        class_names=CLASS_NAMES[:args.num_classes],
    )
    print_results(results)


if __name__ == '__main__':
    main()
