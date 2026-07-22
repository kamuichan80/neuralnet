#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLO-format dataset loading for DEC-YOLO (Section 4.1 of the paper).

Expected directory layout, matching the output of the "Labelling" tool the paper uses:
    root/images/train/*.jpg
    root/images/val/*.jpg
    root/labels/train/*.txt
    root/labels/val/*.txt

Each label file has one line per box: `class_id cx cy w h`, all normalized to [0, 1]
relative to the image width/height (standard YOLO format).
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# scratch, uneven surface, contour damage — the paper's three defect categories
CLASS_NAMES = ['scratch', 'uneven_surface', 'contour_damage']


def letterbox(image, new_size=640, fill=114):
    """Resize to a square canvas while preserving aspect ratio, padding the rest."""
    w, h = image.size
    scale = min(new_size / w, new_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new('RGB', (new_size, new_size), (fill, fill, fill))
    pad_x, pad_y = (new_size - new_w) // 2, (new_size - new_h) // 2
    canvas.paste(image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


class LaserNozzleDataset(Dataset):
    """Laser-nozzle surface-defect dataset in YOLO-txt format."""

    def __init__(self, root, split='train', img_size=640, augment=None):
        self.img_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'labels', split)
        self.img_size = img_size
        self.augment = augment if augment is not None else (split == 'train')

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        self.img_files = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith(IMG_EXTENSIONS)
        )
        if not self.img_files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def _load_labels(self, img_file):
        stem = os.path.splitext(img_file)[0]
        path = os.path.join(self.label_dir, stem + '.txt')
        boxes = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = parts
                    boxes.append([float(cls), float(cx), float(cy), float(w), float(h)])
        return np.array(boxes, dtype=np.float32).reshape(-1, 5)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        image = Image.open(os.path.join(self.img_dir, img_file)).convert('RGB')
        labels = self._load_labels(img_file)  # (N, 5) = class, cx, cy, w, h (normalized)

        ow, oh = image.size
        boxes = labels[:, 1:5].copy()
        boxes[:, [0, 2]] *= ow
        boxes[:, [1, 3]] *= oh
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        image, scale, pad_x, pad_y = letterbox(image, self.img_size)
        xyxy *= scale
        xyxy[:, [0, 2]] += pad_x
        xyxy[:, [1, 3]] += pad_y

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            x1 = xyxy[:, 0].copy()
            xyxy[:, 0] = self.img_size - xyxy[:, 2]
            xyxy[:, 2] = self.img_size - x1

        if self.augment:
            image = self._hsv_jitter(image)

        img_arr = np.asarray(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()

        if len(labels):
            target = np.concatenate([labels[:, 0:1], xyxy], axis=1).astype(np.float32)
        else:
            target = np.zeros((0, 5), dtype=np.float32)

        return img_tensor, torch.from_numpy(target)

    @staticmethod
    def _hsv_jitter(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        """Random HSV gains, same magnitudes the paper uses for its mosaic augmentation."""
        hsv = np.asarray(image.convert('HSV'), dtype=np.float32)
        gains = 1.0 + np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain]
        hsv = np.clip(hsv * gains, 0, 255).astype(np.uint8)
        return Image.fromarray(hsv, mode='HSV').convert('RGB')


def collate_fn(batch):
    """Stacks images and concatenates per-image targets into a single (M, 6) tensor of
    [image_idx, class_id, x1, y1, x2, y2]."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    out = []
    for i, t in enumerate(targets):
        if t.numel():
            idx_col = torch.full((t.shape[0], 1), i, dtype=t.dtype)
            out.append(torch.cat([idx_col, t], dim=1))
    all_targets = torch.cat(out, dim=0) if out else torch.zeros((0, 6), dtype=torch.float32)
    return images, all_targets
