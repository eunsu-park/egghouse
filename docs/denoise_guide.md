# egghouse.denoise 사용 가이드

채널 무관 고전(비-DL) 영상 denoiser 모음. 각 모듈은 함수형
`denoise(image, ...)`와 파라미터형 `*Denoiser` 클래스를 함께 제공하며,
둘 다 `Callable[[np.ndarray], np.ndarray]` 프로토콜을 만족한다 (입력 2-D
float 배열, 출력 동일 shape). v0.9+.

> **설치:** 무거운 의존성은 `denoise` extra로 분리되어 있다.
> `pip install egghouse[denoise]` (scikit-image, PyWavelets, bm3d).
> 서브모듈은 필요할 때만 import되므로, `anscombe`/`wiener`만 쓰면
> scikit-image/bm3d가 없어도 동작한다.

---

## 모듈 구조

```
egghouse/denoise/
├── __init__.py   # 경량 (서브모듈 eager import 안 함)
├── wavelet.py    # WaveletDenoiser  (skimage BayesShrink)
├── bm3d.py       # BM3DDenoiser     (Block-Matching 3D)
├── nlm.py        # NLMDenoiser      (skimage non-local means)
├── tv.py         # TVDenoiser       (Total Variation, Chambolle)
├── wiener.py     # WienerDenoiser   (scipy.signal.wiener)
└── anscombe.py   # AnscombeDenoiser (Poisson 분산 안정화 래퍼)
```

---

## 함수 목록

| 모듈 | 함수형 | 클래스 | 비고 |
|------|--------|--------|------|
| `wavelet` | `denoise(image, sigma=None, ...)` | `WaveletDenoiser` | BayesShrink; sigma 자동 추정 |
| `bm3d` | `denoise(image, sigma=None)` | `BM3DDenoiser` | 강력하나 K-corona 등 자기유사 patch 적은 장면에서 신호 손상 주의 |
| `nlm` | `denoise(image, sigma=None, ...)` | `NLMDenoiser` | non-local means |
| `tv` | `denoise(image, weight=0.1)` | `TVDenoiser` | piecewise-flat prior |
| `wiener` | `denoise(image, mysize=...)` | `WienerDenoiser` | 고주파 공격적 억제 |
| `anscombe` | `forward(x)` / `inverse(z)` / `denoise(image, inner, ...)` | `AnscombeDenoiser` | inner Gaussian denoiser 감쌈 |

---

## 기본 사용

```python
import numpy as np
from egghouse.denoise.wavelet import WaveletDenoiser

noisy = ...                       # 2-D float
denoise = WaveletDenoiser()       # sigma 생략 시 자동 추정
clean = denoise(noisy)            # np.ndarray -> np.ndarray

# 함수형도 동일
from egghouse.denoise import wavelet
clean = wavelet.denoise(noisy, sigma=0.3)
```

평가는 `egghouse.image.metrics`로:

```python
from egghouse.image import psnr, ssim
print(psnr(clean, reference, data_range=4.0), ssim(clean, reference, data_range=4.0))
```

---

## Anscombe (Poisson 데이터)

광자 계수처럼 Poisson 노이즈가 지배하는 데이터는 Anscombe 분산-안정화
변환 후 Gaussian denoiser를 적용하면 좋다. 래퍼는 임의의 inner denoiser를
감싼다.

```python
from egghouse.denoise import anscombe
from egghouse.denoise.bm3d import BM3DDenoiser

clean = anscombe.denoise(noisy_counts, BM3DDenoiser(sigma=1.0))
```

> **가드:** `sqrt(x + 3/8)`는 Poisson 카운트(거의 양수)에만 유효하다.
> 배경-차감/차분 등 **0-중심**(음수 비율이 큰) 입력에서는 변환이 깨지므로,
> 래퍼가 이를 감지해 변환을 bypass하고 inner denoiser를 직접 적용하며
> `UserWarning`을 낸다. 진단용으로 강제하려면 `bypass_on_negative=False`.

---

## 참고

- 함수형 `denoise()`와 클래스형 `*Denoiser()(x)`는 동일 결과 (BM3D는
  내부 비결정성으로 ~1e-3 수준 차이 가능).
- 전체 시그니처: [API_REFERENCE.md](../API_REFERENCE.md) `egghouse.denoise` 절.
