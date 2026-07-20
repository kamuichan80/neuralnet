# neuralnet

PyTorch implementations of classic CNN architectures and a defect-detection model, built
from scratch for study.

## Classification models

| File | Architecture | Notes |
| --- | --- | --- |
| [alexnet.py](alexnet.py) | AlexNet | 5 conv layers + 3 FC layers |
| [vgg16net.py](vgg16net.py) | VGG16 | 13 conv layers + 3 FC layers |
| [googlenet.py](googlenet.py) | GoogLeNet (Inception v1) | Inception modules with auxiliary-free head |
| [resnet18.py](resnet18.py) | ResNet-18 | Residual `BasicBlock`s with identity/projection shortcuts |

Each model takes a `num_classes` argument (default `1000`, matching ImageNet) and expects
`224x224` RGB input.

```bash
pip install torch
python alexnet.py   # builds the model and prints its structure
```

Each file can also be imported as a module without side effects:

```python
from resnet18 import ResNet18, BasicBlock

model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=10)
```

## DEC-YOLO (surface defect detection)

[dec_yolo.py](dec_yolo.py) implements **DEC-YOLO**, from Li, S.; Deng, H.; Zhou, F.; Zheng, Y.
["DEC-YOLO: Surface Defect Detection Algorithm for Laser Nozzles"](https://doi.org/10.3390/electronics14071279),
*Electronics* 2025, 14, 1279. It extends a YOLOv7-style backbone/neck with:

- **DEC module** — a DenseNet branch fused with an explicit visual center (EVC) branch
  (lightweight MLP + learnable visual center), applied before each detection head.
- **Cross-layer connection** — an extra shallow-to-deep skip in the PAN neck so low-level
  edge/texture detail reaches the deeper feature maps directly.
- **Efficient decoupling head** — an anchor-free decoupled head (1 conv for classification,
  2 for regression/objectness).
- **Coordinate attention** on the deepest backbone feature map.

```bash
python dec_yolo.py   # builds the model and prints output shapes for a dummy 640x640 batch
```

### Data format

`datasets.py`'s `LaserNozzleDataset` expects a YOLO-txt layout:

```
<root>/images/train/*.jpg   <root>/labels/train/*.txt
<root>/images/val/*.jpg     <root>/labels/val/*.txt
```

Each label file has one line per box: `class_id cx cy w h`, normalized to `[0, 1]`. The
default class set is `scratch`, `uneven_surface`, `contour_damage` (the paper's dataset).

### Training

```bash
python train.py --data /path/to/dataset --epochs 200 --batch-size 8 --lr 0.001
```

Trains with an FCOS-style anchor-free assignment, CIoU box loss, and BCE
objectness/classification loss (see [losses.py](losses.py)). Checkpoints are written to
`--save-dir` (default `runs/train`) and can be resumed with `--resume <path>`.

### Evaluation

```bash
python evaluate.py --data /path/to/dataset --weights runs/train/epoch_199.pt
```

Reports per-class and overall `AP@0.5`, `AP@0.5:0.95`, precision, recall and F1
(see [metrics.py](metrics.py)), matching the metrics in the paper's Table 6.

## Requirements

- Python 3.8+
- PyTorch (all models)
- NumPy, Pillow (DEC-YOLO data loading/training/evaluation only)

```bash
pip install -r requirements.txt
```
