# neuralnet

PyTorch implementations of classic CNN architectures, built from scratch for study.

## Models

| File | Architecture | Notes |
| --- | --- | --- |
| [alexnet.py](alexnet.py) | AlexNet | 5 conv layers + 3 FC layers |
| [vgg16net.py](vgg16net.py) | VGG16 | 13 conv layers + 3 FC layers |
| [googlenet.py](googlenet.py) | GoogLeNet (Inception v1) | Inception modules with auxiliary-free head |
| [resnet18.py](resnet18.py) | ResNet-18 | Residual `BasicBlock`s with identity/projection shortcuts |

Each model takes a `num_classes` argument (default `1000`, matching ImageNet) and expects
`224x224` RGB input.

## Usage

```bash
pip install torch
python alexnet.py   # builds the model and prints its structure
```

Each file can also be imported as a module without side effects:

```python
from resnet18 import ResNet18, BasicBlock

model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=10)
```

## Requirements

- Python 3.8+
- PyTorch
