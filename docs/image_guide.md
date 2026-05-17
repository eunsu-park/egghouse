# egghouse.image 사용 가이드

범용 이미지 처리 유틸리티. scipy.ndimage 기반으로 dtype을 보존하며 처리.

---

## 모듈 구조

```
egghouse/image/
├── __init__.py   # 모든 함수 export
├── core.py       # 기본 변환 (resize, rotate, bytescale)
├── masking.py    # 마스킹 (circle_mask, annulus_mask)
├── spatial.py    # 공간 변환 (pad, crop_or_pad, flip, roll)
├── filters.py    # 필터링 (gaussian, median, laplacian, sobel, unsharp)
└── stats.py      # 통계/분석 (normalize, histogram_eq, percentile_scale, find_center)
```

---

## 함수 목록

### Core (기본 변환)
| 함수 | 설명 | Alias |
|------|------|-------|
| `resize_image` | 이미지 크기 변경 | `resize` |
| `rotate_image` | 이미지 회전 | `rotate` |
| `bytescale_image` | uint8로 스케일링 | `bytescale` |

### Masking (마스킹)
| 함수 | 설명 |
|------|------|
| `circle_mask` | 원형 마스크 생성 |
| `annulus_mask` | 고리형 마스크 생성 |

### Spatial (공간 변환)
| 함수 | 설명 | Alias |
|------|------|-------|
| `pad_image` | 패딩 추가 | `pad` |
| `crop_or_pad` | 정확한 크기로 조정 | - |
| `flip_image` | 이미지 뒤집기 | - |
| `roll_image` | 순환 이동 | - |

### Filters (필터링)
| 함수 | 설명 |
|------|------|
| `gaussian_smooth` | 가우시안 평활화 |
| `median_denoise` | 미디언 노이즈 제거 |
| `laplacian_edge` | 라플라시안 엣지 검출 |
| `sobel_edge` | 소벨 엣지 검출 |
| `unsharp_mask` | 언샵 마스크 샤프닝 |

### Stats (통계/분석)
| 함수 | 설명 |
|------|------|
| `normalize_image` | z-score 정규화 |
| `get_image_stats` | 이미지 통계 계산 |
| `histogram_equalization` | 히스토그램 균등화 |
| `percentile_scale` | percentile 기반 스케일링 |
| `find_disk_center` | 디스크 중심 찾기 |
| `adaptive_threshold` | 적응형 이진화 |

---

## Core 함수

### resize_image

이미지 크기를 변경. dtype을 보존.

```python
from egghouse.image import resize_image

# 기본 사용 (bilinear 보간)
resized = resize_image(image, (512, 512))

# 보간 방법 선택
resized = resize_image(image, (256, 256), order=0)  # nearest
resized = resize_image(image, (256, 256), order=1)  # bilinear (기본)
resized = resize_image(image, (256, 256), order=3)  # bicubic

# 3D 이미지 (H, W, C) 지원
rgb_resized = resize_image(rgb_image, (256, 256))
```

**order 값:**
- 0: nearest-neighbor (가장 빠름, 계단 현상)
- 1: bilinear (기본값, 균형)
- 2: bi-quadratic
- 3: bi-cubic (가장 부드러움, 느림)

---

### rotate_image

이미지를 회전. 양수 각도는 반시계 방향.

```python
from egghouse.image import rotate_image

# 기본 회전 (원본 크기 유지)
rotated = rotate_image(image, angle=45)

# 전체 이미지가 보이도록 캔버스 확장
rotated = rotate_image(image, angle=45, reshape=True)

# 빈 영역 채우기 값 지정
rotated = rotate_image(image, angle=30, cval=np.nan)
```

**파라미터:**
- `angle`: 회전 각도 (도 단위)
- `reshape`: True면 전체 이미지가 보이도록 출력 크기 조정
- `cval`: 경계 밖 영역 채우기 값

---

### bytescale_image

데이터를 uint8 범위 [0, 255]로 스케일링. 시각화용.

```python
from egghouse.image import bytescale_image

# 자동 범위 감지
display = bytescale_image(data)

# 범위 지정
display = bytescale_image(data, imin=0, imax=5000)

# percentile 기반 contrast stretch
p1, p99 = np.percentile(data, [1, 99])
display = bytescale_image(data, imin=p1, imax=p99)

# 출력 범위 변경
display = bytescale_image(data, omin=10, omax=245)  # 여백 확보
```

---

## Masking 함수

### circle_mask

원형 boolean 마스크 생성. 태양 디스크 마스킹에 유용.

```python
from egghouse.image import circle_mask

# 태양 디스크 마스크 (4096x4096, 반경 1600 픽셀)
disk_mask = circle_mask(4096, radius=1600)

# 디스크 내부만 추출
masked = np.where(disk_mask, image, 0)

# 디스크 외부 마스크 (코로나 분석용)
corona_mask = circle_mask(4096, radius=1600, mask_type='outer')

# 사각형 이미지, 중심 지정
mask = circle_mask((512, 1024), radius=200, center=(256, 600))
```

**mask_type:**
- `'inner'`: 원 내부가 True (기본값)
- `'outer'`: 원 외부가 True

---

### annulus_mask

고리형 마스크 생성. 특정 반경 범위 분석용.

```python
from egghouse.image import annulus_mask

# 1.0 ~ 1.3 태양 반경 영역
solar_radius = 1600
corona_ring = annulus_mask(4096,
                           inner_radius=solar_radius,
                           outer_radius=solar_radius * 1.3)

# 마스크 적용
corona_data = image[corona_ring]
mean_intensity = corona_data.mean()
```

---

## Spatial 함수

### pad_image

이미지에 패딩 추가.

```python
from egghouse.image import pad_image

# 중앙 정렬 패딩 (기본)
padded = pad_image(image, (1024, 1024), pad_value=0)

# 좌상단 정렬
padded = pad_image(image, (1024, 1024), center=False)

# NaN으로 패딩 (off-disk 영역)
padded = pad_image(image, (5000, 5000), pad_value=np.nan)
```

---

### crop_or_pad

크기에 맞게 자동으로 crop 또는 pad.

```python
from egghouse.image import crop_or_pad

# 다양한 크기의 이미지를 동일하게 맞춤
img1 = np.random.rand(400, 600)   # 작은 이미지 → pad
img2 = np.random.rand(800, 500)   # 큰 이미지 → crop

normalized1 = crop_or_pad(img1, (512, 512))
normalized2 = crop_or_pad(img2, (512, 512))
# 둘 다 (512, 512) 크기
```

---

### flip_image

이미지 뒤집기.

```python
from egghouse.image import flip_image

# 상하 반전 (기본)
flipped = flip_image(image, axis='vertical')

# 좌우 반전
flipped = flip_image(image, axis='horizontal')

# 180도 회전 (양쪽 반전)
flipped = flip_image(image, axis='both')
```

**axis:**
- `'vertical'`: 상하 반전 (기본값)
- `'horizontal'`: 좌우 반전
- `'both'`: 둘 다 (180도 회전)

---

### roll_image

순환 이동 (cyclic shift). 경계를 넘는 픽셀이 반대편에 나타남.

```python
from egghouse.image import roll_image

# 아래로 10픽셀, 오른쪽으로 5픽셀 이동
rolled = roll_image(image, shift_y=10, shift_x=5)

# 이미지 정렬에 활용
for i, img in enumerate(images):
    aligned = roll_image(img, shift_y=0, shift_x=shifts[i])
```

---

## Filters 함수

### gaussian_smooth

가우시안 필터로 노이즈 감소. 자연스러운 평활화.

```python
from egghouse.image import gaussian_smooth

# 기본 평활화 (sigma=1.0)
smoothed = gaussian_smooth(image, sigma=1.5)

# 축별 다른 sigma
smoothed = gaussian_smooth(image, sigma=(2.0, 1.0))

# dtype 보존하지 않기
smoothed = gaussian_smooth(image, sigma=1.0, preserve_range=False)
```

---

### median_denoise

미디언 필터로 노이즈 제거. 엣지 보존에 우수. salt-and-pepper 노이즈, cosmic ray 제거에 효과적.

```python
from egghouse.image import median_denoise

# 기본 (3x3 윈도우)
denoised = median_denoise(image, size=3)

# 더 강한 노이즈용
denoised = median_denoise(noisy_image, size=5)

# 비정방 윈도우
denoised = median_denoise(image, size=(3, 5))
```

---

### laplacian_edge

라플라시안 엣지 검출. 2차 미분으로 급격한 변화 감지.

```python
from egghouse.image import laplacian_edge

# 기본 엣지 검출
edges = laplacian_edge(image)

# 가우시안 전처리 후 적용 (LoG)
smoothed = gaussian_smooth(image, sigma=1.0)
edges = laplacian_edge(smoothed)
```

---

### sobel_edge

소벨 엣지 검출. 1차 미분(그래디언트) 기반.

```python
from egghouse.image import sobel_edge

# 그래디언트 크기 (모든 엣지)
edges = sobel_edge(image)

# 수직 엣지만 (y 방향 그래디언트)
edges_y = sobel_edge(image, axis=0)

# 수평 엣지만 (x 방향 그래디언트)
edges_x = sobel_edge(image, axis=1)
```

---

### unsharp_mask

언샵 마스크 샤프닝. 블러 이미지를 빼서 엣지 강조.

```python
from egghouse.image import unsharp_mask

# 기본 샤프닝
sharp = unsharp_mask(image, sigma=1.0, amount=1.0)

# 더 강한 샤프닝
sharp = unsharp_mask(image, sigma=2.0, amount=2.0)
```

**파라미터:**
- `sigma`: 블러 강도. 높을수록 넓은 엣지 강조
- `amount`: 샤프닝 강도. 1.0 이상이면 더 강하게

---

## Stats 함수

### normalize_image

z-score 정규화. 평균 0, 표준편차 1로 변환.

```python
from egghouse.image import normalize_image

# 자동 계산
normalized = normalize_image(image)
# mean ≈ 0, std ≈ 1

# 사전 계산된 통계 사용 (학습 세트 기준)
normalized = normalize_image(image, mean=127.5, std=64.0)
```

---

### get_image_stats

이미지 통계 계산. 마스크 지원.

```python
from egghouse.image import get_image_stats

# 전체 이미지 통계
stats = get_image_stats(image)
print(f"Mean: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")
print(f"Min: {stats['min']}, Max: {stats['max']}")
print(f"p1={stats['p1']}, p99={stats['p99']}")

# 태양 디스크 내부만
disk_mask = circle_mask(4096, radius=1600)
stats = get_image_stats(image, mask=disk_mask)

# 커스텀 percentiles
stats = get_image_stats(image, percentiles=(5, 50, 95))
```

**반환값:**
- `mean`, `std`, `min`, `max`, `median`, `count`
- `p1`, `p5`, `p25`, `p50`, `p75`, `p95`, `p99` (또는 커스텀)

---

### histogram_equalization

히스토그램 균등화. 좁은 명암 분포를 균일하게.

```python
from egghouse.image import histogram_equalization

# 저대비 이미지 개선
enhanced = histogram_equalization(image)

# 비교 시각화
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image, cmap='gray')
ax1.set_title('Original')
ax2.imshow(enhanced, cmap='gray')
ax2.set_title('Equalized')
```

---

### percentile_scale

percentile 기반 스케일링. outlier에 강인함.

```python
from egghouse.image import percentile_scale

# 기본 (1%, 99%)
scaled = percentile_scale(image)

# 더 공격적인 클리핑
scaled = percentile_scale(image, low_percentile=5, high_percentile=95)

# 커스텀 출력 범위
scaled = percentile_scale(image, omin=10, omax=245)
```

bytescale_image와 유사하지만, 자동으로 percentile 기반 범위 사용.

---

### find_disk_center

밝은 디스크(태양 등)의 중심 좌표 찾기.

```python
from egghouse.image import find_disk_center

# 태양 디스크 중심 찾기
cy, cx = find_disk_center(aia_image)
print(f"Center: ({cy:.1f}, {cx:.1f})")

# 커스텀 threshold
cy, cx = find_disk_center(image, threshold=100)

# 기하학적 중심 (intensity 무시)
cy, cx = find_disk_center(image, method='geometric')

# 찾은 중심으로 마스크 생성
mask = circle_mask(image.shape, radius=1600, center=(cy, cx))
```

**method:**
- `'centroid'`: 밝기 가중 중심 (기본값)
- `'geometric'`: 기하학적 중심

---

### adaptive_threshold

적응형 이진화. 불균일한 조명 처리.

```python
from egghouse.image import adaptive_threshold

# 기본 적응형 threshold
binary = adaptive_threshold(image)

# 더 민감하게 (더 많은 foreground)
binary = adaptive_threshold(image, offset=-5)

# 더 작은 블록 (세부 사항 보존)
binary = adaptive_threshold(image, block_size=15)
```

**파라미터:**
- `block_size`: 로컬 평균 계산 윈도우 크기 (홀수)
- `offset`: 평균에서 뺄 값. 양수면 foreground 감소

---

## 워크플로우 예시

### 태양 이미지 처리

```python
from astropy.io import fits  # egghouse.io는 v0.6.0에서 제거됨
from egghouse.image import (
    resize_image, circle_mask, bytescale_image, crop_or_pad,
    gaussian_smooth, find_disk_center, get_image_stats
)

# 1. FITS 파일 읽기
data, header = fits.getdata('aia_171.fits', header=True)

# 2. 크기 정규화
data = crop_or_pad(data, (4096, 4096))

# 3. 디스크 중심 찾기 및 마스크 적용
cy, cx = find_disk_center(data)
disk_mask = circle_mask(4096, radius=1600, center=(cy, cx))
data = np.where(disk_mask, data, 0)

# 4. 통계 확인
stats = get_image_stats(data, mask=disk_mask)
print(f"Disk mean: {stats['mean']:.1f}")

# 5. 노이즈 제거
data = gaussian_smooth(data, sigma=1.0)

# 6. 리사이즈 (ML 입력용)
resized = resize_image(data, (512, 512), order=1)

# 7. 시각화용 스케일링
display = bytescale_image(resized, imin=0, imax=5000)
```

### 엣지 검출

```python
from egghouse.image import gaussian_smooth, sobel_edge, laplacian_edge

# 노이즈 제거 후 엣지 검출
smoothed = gaussian_smooth(image, sigma=1.5)

# 소벨: 그래디언트 크기
edges_sobel = sobel_edge(smoothed)

# 라플라시안: 2차 미분
edges_laplacian = laplacian_edge(smoothed)
```

---

## dtype 보존

모든 함수는 `preserve_range=True`(기본값)로 원본 dtype을 보존:

```python
import numpy as np

# uint16 입력
img_uint16 = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

# 처리 후에도 uint16 유지
resized = resize_image(img_uint16, (50, 50))
print(resized.dtype)  # uint16

smoothed = gaussian_smooth(img_uint16, sigma=1.0)
print(smoothed.dtype)  # uint16
```

`preserve_range=False`로 설정하면 float64로 반환:

```python
resized = resize_image(img_uint16, (50, 50), preserve_range=False)
print(resized.dtype)  # float64
```

---

## 의존성

- numpy
- scipy (ndimage)

설치: `pip install numpy scipy`
