#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Training script for DEC-YOLO on a YOLO-format surface-defect dataset (paper Section 4.1, 4.2).

Expected data layout:
    <data>/images/train/*.jpg   <data>/labels/train/*.txt
    <data>/images/val/*.jpg     <data>/labels/val/*.txt

Example:
    python tools/train.py --data /path/to/laser_nozzle_dataset --epochs 200 --batch-size 8 --lr 0.001
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.loss import DECYOLOLoss
from yolov6.data.datasets import CLASS_NAMES, LaserNozzleDataset, collate_fn
from yolov6.models.yolo import build_model
from yolov6.utils.config import Config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='dataset root; see module docstring for layout')
    p.add_argument('--conf-file', default='configs/dec_yolo.py', help='model config, see configs/dec_yolo.py')
    p.add_argument('--num-classes', type=int, default=len(CLASS_NAMES))
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--momentum', type=float, default=0.937)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', default=None, help='checkpoint path to resume from')
    p.add_argument('--save-dir', default='runs/train')
    p.add_argument('--save-every', type=int, default=10, help='checkpoint interval, in epochs')
    p.add_argument('--log-every', type=int, default=50, help='log interval, in steps')
    return p.parse_args()


def build_dataloaders(args):
    train_set = LaserNozzleDataset(args.data, split='train', img_size=args.img_size, augment=True)
    val_set = LaserNozzleDataset(args.data, split='val', img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, val_loader


def _format_metrics(metrics):
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    totals, n = {}, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        loss, parts = criterion(preds, targets)
        totals['loss'] = totals.get('loss', 0.0) + loss.item()
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, args):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'args': vars(args),
    }, path)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    cfg = Config.fromfile(args.conf_file)
    model = build_model(cfg, num_classes=args.num_classes, device=device)
    criterion = DECYOLOLoss(num_classes=args.num_classes)
    train_loader, val_loader = build_dataloaders(args)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {}
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            preds = model(images)
            loss, parts = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running['loss'] = running.get('loss', 0.0) + loss.item()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item()

            if (i + 1) % args.log_every == 0:
                avg = {k: v / (i + 1) for k, v in running.items()}
                print(f"epoch {epoch} [{i + 1}/{len(train_loader)}] {_format_metrics(avg)}")

        scheduler.step()
        avg = {k: v / len(train_loader) for k, v in running.items()}
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s {_format_metrics(avg)}")

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch} val {_format_metrics(val_metrics)}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, args)
            print(f"saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
