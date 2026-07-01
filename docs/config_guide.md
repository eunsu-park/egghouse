# egghouse.config Usage Guide

A configuration management utility for ML/DL projects.

---

## Overview

BaseConfig is a dataclass-based configuration management class that can load settings from various sources:
- YAML files
- JSON files
- Environment variables
- CLI arguments

---

## Basic Usage

### Defining a Config Class

```python
from dataclasses import dataclass
from typing import Optional
from egghouse.config import BaseConfig

@dataclass
class TrainConfig(BaseConfig):
    """Training config"""
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    model_name: str = "resnet50"
    use_amp: bool = True
    checkpoint_path: Optional[str] = None
```

### Creating with Default Values

```python
config = TrainConfig()
print(config.lr)        # 0.001
print(config.epochs)    # 100
```

---

## YAML Files

### Loading (from_yaml)

```yaml
# config.yaml
lr: 0.0001
epochs: 200
batch_size: 64
model_name: efficientnet_b0
use_amp: true
checkpoint_path: /checkpoints/model.pt
```

```python
config = TrainConfig.from_yaml('config.yaml')
print(config.lr)  # 0.0001
```

### Saving (to_yaml)

```python
config = TrainConfig(lr=0.0005, epochs=150)
config.to_yaml('output_config.yaml')
```

Generated file:
```yaml
lr: 0.0005
epochs: 150
batch_size: 32
model_name: resnet50
use_amp: true
checkpoint_path: null
```

---

## JSON Files

### Loading (from_json)

```json
{
  "lr": 0.0001,
  "epochs": 200,
  "batch_size": 64,
  "model_name": "efficientnet_b0"
}
```

```python
config = TrainConfig.from_json('config.json')
```

### Saving (to_json)

```python
config = TrainConfig(lr=0.0005)
config.to_json('output_config.json', indent=4)
```

---

## Environment Variables

### Loading (from_env)

Environment variable name: `PREFIX` + `FIELD_NAME` (uppercase)

```bash
export TRAIN_LR=0.0001
export TRAIN_EPOCHS=200
export TRAIN_BATCH_SIZE=64
export TRAIN_USE_AMP=true
```

```python
config = TrainConfig.from_env(prefix="TRAIN_")
print(config.lr)         # 0.0001
print(config.epochs)     # 200
print(config.use_amp)    # True
```

### Type Conversion

Environment variables are strings, but they are automatically converted to match the field type:

| Type | Conversion rule |
|------|----------|
| `int` | `int(value)` |
| `float` | `float(value)` |
| `bool` | `true`, `1`, `yes`, `on` → True |
| `str` | Used as-is |
| `Optional[T]` | Converted to type T |

---

## CLI Arguments

### Loading (from_args)

```bash
# Use default values
python train.py

# Override values via CLI
python train.py --lr 0.0001 --epochs 200

# YAML base + CLI override
python train.py --config base.yaml --lr 0.0001

# JSON base + CLI override
python train.py --config base.json --epochs 300
```

```python
# train.py
from egghouse.config import BaseConfig
from dataclasses import dataclass

@dataclass
class TrainConfig(BaseConfig):
    lr: float = 0.001
    epochs: int = 100

if __name__ == '__main__':
    config = TrainConfig.from_args()
    print(config)
```

### Boolean Fields

```bash
# Set to True
python train.py --use_amp

# Set to False
python train.py --no-use_amp
```

### Auto-generated --help

```bash
python train.py --help
```

```
usage: train.py [-h] [--config CONFIG] [--lr LR] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--model_name MODEL_NAME]
                [--use_amp] [--no-use_amp]

Training config

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML/JSON config file
  --lr LR               lr (default: 0.001)
  --epochs EPOCHS       epochs (default: 100)
  --batch_size BATCH_SIZE
                        batch_size (default: 32)
  --use_amp             use_amp (default: True)
  --no-use_amp          Disable use_amp
```

---

## Priority

When using `from_args()`:

1. **CLI arguments** (highest priority)
2. **--config file** (YAML/JSON)
3. **dataclass default values** (lowest priority)

```bash
# base.yaml specifies lr=0.001, CLI specifies lr=0.0001
python train.py --config base.yaml --lr 0.0001
# → config.lr == 0.0001 (CLI takes priority)
```

---

## Nested Configuration

Nested dataclasses for complex configurations:

```python
from dataclasses import dataclass, field
from typing import List
from egghouse.config import BaseConfig

@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0001

@dataclass
class DataConfig:
    train_path: str = "/data/train"
    val_path: str = "/data/val"
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class TrainConfig(BaseConfig):
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    epochs: int = 100
```

YAML file:
```yaml
optimizer:
  name: sgd
  lr: 0.01
  weight_decay: 0.0005
data:
  train_path: /custom/train
  batch_size: 64
epochs: 200
```

**Note**: Nested dataclasses only work with YAML/JSON; CLI and environment variables only support a flat structure.

---

## Practical Examples

### ML Training Script

```python
# train.py
from dataclasses import dataclass
from typing import Optional
from egghouse.config import BaseConfig

@dataclass
class Config(BaseConfig):
    """Solar image classification training config"""
    # Model
    model: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 10

    # Training
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32

    # Data
    data_dir: str = "/data/solar"
    image_size: int = 224

    # Misc
    device: str = "cuda"
    seed: int = 42
    checkpoint: Optional[str] = None

def main():
    config = Config.from_args()
    print(config)

    # Save config (reproducibility)
    config.to_yaml(f'runs/{config.model}_config.yaml')

    # Training logic...

if __name__ == '__main__':
    main()
```

Run:
```bash
# Default settings
python train.py

# Custom config file + override
python train.py --config experiments/exp1.yaml --lr 0.0001 --epochs 200

# Using environment variables (e.g. Docker)
export CONFIG_LR=0.0001
export CONFIG_EPOCHS=200
python train.py  # when using from_env()
```

### Config Validation

```python
from dataclasses import dataclass, field
from egghouse.config import BaseConfig

@dataclass
class Config(BaseConfig):
    lr: float = 0.001
    epochs: int = 100

    def __post_init__(self):
        """Validate after loading config"""
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
```

---

## API Summary

### BaseConfig Class Methods

| Method | Description |
|--------|------|
| `from_yaml(path)` | Load from a YAML file |
| `from_json(path)` | Load from a JSON file |
| `from_env(prefix="")` | Load from environment variables |
| `from_args(args=None)` | Load from CLI arguments |
| `to_yaml(path)` | Save to a YAML file |
| `to_json(path, indent=2)` | Save to a JSON file |

---

## Dependencies

- **pyyaml**: YAML support

Installation:
```bash
pip install pyyaml
```

---

## Best Practices

1. **Set default values**: Specify reasonable default values for every field
2. **Type hints**: Add type hints to every field
3. **Documentation**: Describe settings with a docstring
4. **Validation**: Validate values in `__post_init__`
5. **Reproducibility**: Save the config to a file at the start of training
