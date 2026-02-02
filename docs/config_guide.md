# egghouse.config 사용 가이드

ML/DL 프로젝트를 위한 설정 관리 유틸리티.

---

## 개요

BaseConfig는 dataclass 기반의 설정 관리 클래스로 다양한 소스에서 설정을 로드할 수 있습니다:
- YAML 파일
- JSON 파일
- 환경 변수
- CLI 인자

---

## 기본 사용법

### 설정 클래스 정의

```python
from dataclasses import dataclass
from typing import Optional
from egghouse.config import BaseConfig

@dataclass
class TrainConfig(BaseConfig):
    """학습 설정"""
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    model_name: str = "resnet50"
    use_amp: bool = True
    checkpoint_path: Optional[str] = None
```

### 기본값으로 생성

```python
config = TrainConfig()
print(config.lr)        # 0.001
print(config.epochs)    # 100
```

---

## YAML 파일

### 로드 (from_yaml)

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

### 저장 (to_yaml)

```python
config = TrainConfig(lr=0.0005, epochs=150)
config.to_yaml('output_config.yaml')
```

생성된 파일:
```yaml
lr: 0.0005
epochs: 150
batch_size: 32
model_name: resnet50
use_amp: true
checkpoint_path: null
```

---

## JSON 파일

### 로드 (from_json)

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

### 저장 (to_json)

```python
config = TrainConfig(lr=0.0005)
config.to_json('output_config.json', indent=4)
```

---

## 환경 변수

### 로드 (from_env)

환경 변수 이름: `PREFIX` + `FIELD_NAME` (대문자)

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

### 타입 변환

환경 변수는 문자열이지만 필드 타입에 맞게 자동 변환됩니다:

| 타입 | 변환 규칙 |
|------|----------|
| `int` | `int(value)` |
| `float` | `float(value)` |
| `bool` | `true`, `1`, `yes`, `on` → True |
| `str` | 그대로 사용 |
| `Optional[T]` | T 타입으로 변환 |

---

## CLI 인자

### 로드 (from_args)

```bash
# 기본값 사용
python train.py

# CLI로 값 오버라이드
python train.py --lr 0.0001 --epochs 200

# YAML 베이스 + CLI 오버라이드
python train.py --config base.yaml --lr 0.0001

# JSON 베이스 + CLI 오버라이드
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

### Boolean 필드

```bash
# True로 설정
python train.py --use_amp

# False로 설정
python train.py --no-use_amp
```

### 자동 생성되는 --help

```bash
python train.py --help
```

```
usage: train.py [-h] [--config CONFIG] [--lr LR] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--model_name MODEL_NAME]
                [--use_amp] [--no-use_amp]

학습 설정

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

## 우선순위

`from_args()`를 사용할 때:

1. **CLI 인자** (최우선)
2. **--config 파일** (YAML/JSON)
3. **dataclass 기본값** (최하위)

```bash
# base.yaml에 lr=0.001, CLI에 lr=0.0001 지정
python train.py --config base.yaml --lr 0.0001
# → config.lr == 0.0001 (CLI 우선)
```

---

## 중첩 설정

복잡한 설정을 위한 중첩 dataclass:

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

YAML 파일:
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

**주의**: 중첩 dataclass는 YAML/JSON에서만 동작하며, CLI와 환경 변수에서는 평면 구조만 지원됩니다.

---

## 실전 예시

### ML 학습 스크립트

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

    # 설정 저장 (재현성)
    config.to_yaml(f'runs/{config.model}_config.yaml')

    # 학습 로직...

if __name__ == '__main__':
    main()
```

실행:
```bash
# 기본 설정
python train.py

# 커스텀 설정 파일 + 오버라이드
python train.py --config experiments/exp1.yaml --lr 0.0001 --epochs 200

# 환경 변수 사용 (예: Docker)
export CONFIG_LR=0.0001
export CONFIG_EPOCHS=200
python train.py  # from_env() 사용 시
```

### 설정 검증

```python
from dataclasses import dataclass, field
from egghouse.config import BaseConfig

@dataclass
class Config(BaseConfig):
    lr: float = 0.001
    epochs: int = 100

    def __post_init__(self):
        """설정 로드 후 검증"""
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
```

---

## API 요약

### BaseConfig 클래스 메서드

| 메서드 | 설명 |
|--------|------|
| `from_yaml(path)` | YAML 파일에서 로드 |
| `from_json(path)` | JSON 파일에서 로드 |
| `from_env(prefix="")` | 환경 변수에서 로드 |
| `from_args(args=None)` | CLI 인자에서 로드 |
| `to_yaml(path)` | YAML 파일로 저장 |
| `to_json(path, indent=2)` | JSON 파일로 저장 |

---

## 의존성

- **pyyaml**: YAML 지원

설치:
```bash
pip install pyyaml
```

---

## 모범 사례

1. **기본값 설정**: 모든 필드에 합리적인 기본값 지정
2. **타입 힌트**: 모든 필드에 타입 힌트 추가
3. **문서화**: docstring으로 설정 설명
4. **검증**: `__post_init__`에서 값 검증
5. **재현성**: 학습 시작 시 설정을 파일로 저장
