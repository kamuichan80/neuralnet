# neuralnet

PyTorch implementations of classic CNN architectures and a defect-detection model, built
from scratch for study. Laid out and coded in the style of
[meituan/YOLOv6](https://github.com/meituan/YOLOv6): a `layers/` of reusable conv blocks,
per-model files under `models/`, config-object-driven construction
(`Config.fromfile()` + `build_model(cfg, ...)`), and CLI entry points under `tools/`.

## Layout

```
configs/        # architecture configs: depth/width multiples, channel lists, num_classes
classifiers/    # AlexNet, VGG16, GoogLeNet, ResNet-18
yolov6/
  layers/       # ConvBNSiLU, RepVGGBlock, CSPSPPF, MP1/MP2, ELAN/ELANH
  models/       # backbone, neck, dec module, coordinate attention, head, model assembly
  data/         # dataset loading
  core/         # loss, mAP evaluator
  utils/        # Config loader, make_divisible, BN fusion, weight init
tools/          # train.py, eval.py
```

## Classification models

| File | Architecture | Notes |
| --- | --- | --- |
| [classifiers/alexnet.py](classifiers/alexnet.py) | AlexNet | 5 conv layers + 3 FC layers |
| [classifiers/vgg16net.py](classifiers/vgg16net.py) | VGG16 | 13 conv layers + 3 FC layers, conv stack driven by a `cfgs` channel list |
| [classifiers/googlenet.py](classifiers/googlenet.py) | GoogLeNet (Inception v1) | Inception modules with auxiliary-free head |
| [classifiers/resnet18.py](classifiers/resnet18.py) | ResNet-18 | Residual `BasicBlock`s with identity/projection shortcuts |

Each model is built from its `configs/<model>.py` (`num_classes`, defaulting to `1000` to
match ImageNet) via a `build_model(cfg)` factory, and expects `224x224` RGB input.

```bash
pip install torch
python -m classifiers.alexnet   # loads configs/alexnet.py, builds the model, prints its structure
```

Each file can also be imported as a module without side effects:

```python
from classifiers.resnet18 import build_model
from yolov6.utils.config import Config

model = build_model(Config.fromfile('configs/resnet18.py'))
```

## DEC-YOLO (surface defect detection)

[yolov6/models/yolo.py](yolov6/models/yolo.py) implements **DEC-YOLO**, from Li, S.; Deng, H.;
Zhou, F.; Zheng, Y.
["DEC-YOLO: Surface Defect Detection Algorithm for Laser Nozzles"](https://doi.org/10.3390/electronics14071279),
*Electronics* 2025, 14, 1279. It extends a YOLOv7-style backbone/neck with:

- **DEC module** ([yolov6/models/dec.py](yolov6/models/dec.py)) — a DenseNet branch fused
  with an explicit visual center (EVC) branch (lightweight MLP + learnable visual center),
  applied before each detection head.
- **Cross-layer connection** ([yolov6/models/neck.py](yolov6/models/neck.py)) — an extra
  shallow-to-deep skip in the PAN neck so low-level edge/texture detail reaches the deeper
  feature maps directly.
- **Efficient decoupling head** ([yolov6/models/heads/effidehead.py](yolov6/models/heads/effidehead.py))
  — an anchor-free decoupled head (1 conv for classification, 2 for regression/objectness).
- **Coordinate attention** ([yolov6/models/attention.py](yolov6/models/attention.py)) on the
  deepest backbone feature map.

These are the paper's own contributions and have no YOLOv6 equivalent, so only their file
layout and naming follow YOLOv6's conventions — the architecture itself is unchanged. Two
blocks in [yolov6/layers/common.py](yolov6/layers/common.py) *do* overlap with real YOLOv6
modules and are swapped for YOLOv6's actual implementation, a deliberate style choice rather
than a paper-fidelity regression:

- `RepConv` → **`RepVGGBlock`**: gains YOLOv6's real train/deploy re-parameterization —
  `switch_to_deploy()` exactly fuses the 3x3/1x1/identity branches into a single conv for
  inference. Activation stays SiLU (matching the paper's YOLOv7 lineage; YOLOv6 itself
  defaults this block to ReLU).
- `SPPCSPC` → **`CSPSPPF`**: the same 7-conv CSP topology, but replaces three parallel
  differently-sized max-pools with YOLOv6's cheaper trick of applying one kernel-size
  max-pool sequentially three times.

Model construction is config-driven, YOLOv6-style: `configs/dec_yolo.py` sets
`depth_multiple` / `width_multiple` and the base channel list, and
`yolov6/models/yolo.py::build_model(cfg, num_classes)` builds the model from it.
`width_multiple` scales every channel count in the backbone/neck/dec/head; `depth_multiple`
scales the one genuinely repeatable count in the architecture — `DenseNetModule`'s layer
count inside the DEC module (`ELAN`/`ELANH` are fixed 4-branch topologies, not repeat-based,
so they aren't depth-scaled). Defaults reproduce the paper's own architecture exactly.

```bash
python -m yolov6.models.yolo   # builds the model and prints output shapes for a dummy 640x640 batch
```

### Data format

`yolov6/data/datasets.py`'s `LaserNozzleDataset` expects a YOLO-txt layout:

```
<root>/images/train/*.jpg   <root>/labels/train/*.txt
<root>/images/val/*.jpg     <root>/labels/val/*.txt
```

Each label file has one line per box: `class_id cx cy w h`, normalized to `[0, 1]`. The
default class set is `scratch`, `uneven_surface`, `contour_damage` (the paper's dataset).

### Training

```bash
python tools/train.py --data /path/to/dataset --conf-file configs/dec_yolo.py --epochs 200 --batch-size 8 --lr 0.001
```

Trains with an FCOS-style anchor-free assignment, CIoU box loss, and BCE
objectness/classification loss (see [yolov6/core/loss.py](yolov6/core/loss.py)).
Checkpoints are written to `--save-dir` (default `runs/train`) and can be resumed with
`--resume <path>`.

### Evaluation

```bash
python tools/eval.py --data /path/to/dataset --weights runs/train/epoch_199.pt --conf-file configs/dec_yolo.py
```

Reports per-class and overall `AP@0.5`, `AP@0.5:0.95`, precision, recall and F1
(see [yolov6/core/evaluator.py](yolov6/core/evaluator.py)), matching the metrics in the
paper's Table 6.

## Requirements

- Python 3.8+
- PyTorch (all models)
- NumPy, Pillow (DEC-YOLO data loading/training/evaluation only)

```bash
pip install -r requirements.txt
```
