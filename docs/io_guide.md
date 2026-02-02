# egghouse.io 사용 가이드

과학 데이터 포맷 파일 입출력 유틸리티.

---

## 지원 포맷

| 포맷 | 읽기 | 쓰기 | 의존성 |
|------|------|------|--------|
| FITS | `read_fits`, `read_fits_simple` | `write_fits`, `append_fits` | astropy (선택) |
| BMP | `read_bmp` | `write_bmp` | 없음 |

---

## FITS (Flexible Image Transport System)

천문학 표준 데이터 포맷. HDU(Header Data Unit) 구조로 헤더와 데이터를 저장.

### 구조

```
┌─────────────────────────────────────┐
│ Header Block (2880 bytes × N)       │
│ - 80-char cards: KEYWORD = value    │
│ - END card로 종료                    │
├─────────────────────────────────────┤
│ Data Block                          │
│ - Big-endian byte order             │
│ - BITPIX: 데이터 타입 결정           │
└─────────────────────────────────────┘
```

**BITPIX 값:**
- `8`: uint8
- `16`: int16
- `32`: int32
- `-32`: float32
- `-64`: float64

### astropy 기반 함수

astropy 설치 필요: `pip install astropy`

```python
from egghouse.io import read_fits, write_fits, read_fits_header, append_fits

# 읽기
data, header = read_fits('image.fits')
data, header = read_fits('multi.fits', hdu_index=1)  # Extension HDU

# 헤더만 읽기 (메모리 효율적)
header = read_fits_header('image.fits')

# 쓰기
write_fits('output.fits', data, header={'OBJECT': 'Sun'}, overwrite=True)

# Extension HDU 추가
append_fits('output.fits', ext_data, header={'EXTNAME': 'EXT1'})
```

### Pure NumPy 함수 (의존성 없음)

astropy 없이 FITS 파일을 읽을 때 사용.

```python
from egghouse.io import read_fits_simple, read_fits_header_simple

# Primary HDU 읽기
data, header = read_fits_simple('image.fits')

# Extension HDU 읽기
ext_data, ext_header = read_fits_simple('multi.fits', hdu_index=1)

# BSCALE/BZERO 스케일링 비활성화 (raw 데이터)
raw_data, header = read_fits_simple('image.fits', apply_scaling=False)

# 헤더만 읽기
header = read_fits_header_simple('image.fits')
```

**제한사항:**
- Binary Table 미지원 (astropy 필요)
- Compressed FITS (.fz) 미지원 (astropy 필요)
- 쓰기 기능 없음

### 함수 비교

| 함수 | 의존성 | Binary Table | Compressed | 쓰기 |
|------|--------|--------------|------------|------|
| `read_fits` | astropy | O | O | - |
| `write_fits` | astropy | - | - | O |
| `read_fits_simple` | 없음 | X | X | - |

---

## BMP (Windows Bitmap)

비압축 이미지 포맷. 외부 의존성 없이 numpy만으로 처리.

### 구조

```
┌──────────────────────────────────────┐
│ File Header (14 bytes)               │
│ - Signature: 'BM'                    │
│ - File size, Data offset             │
├──────────────────────────────────────┤
│ DIB Header (40 bytes, BITMAPINFOHEADER) │
│ - Width, Height, Bit depth           │
├──────────────────────────────────────┤
│ Color Table (8-bit only)             │
│ - 256 × 4 bytes (BGRA)               │
├──────────────────────────────────────┤
│ Pixel Data                           │
│ - Bottom-up, 4-byte row padding      │
└──────────────────────────────────────┘
```

### 사용 예시

```python
from egghouse.io import read_bmp, write_bmp, read_bmp_header

# 읽기 (항상 RGB로 반환)
data, info = read_bmp('image.bmp')  # (H, W, 3) uint8

print(info)
# {'width': 1024, 'height': 768, 'bit_depth': 24, ...}

# 쓰기
write_bmp('output.bmp', rgb_data)           # 24-bit RGB
write_bmp('gray.bmp', gray_uint8)           # 8-bit grayscale
write_bmp('output.bmp', data, overwrite=True)

# 헤더만 읽기
info = read_bmp_header('image.bmp')
```

**지원 형식:**
- 읽기: 8-bit (grayscale), 24-bit (RGB)
- 쓰기: 8-bit grayscale `(H, W)`, 24-bit RGB `(H, W, 3)`

---

## 의존성 확인

```python
from egghouse.io import HAS_ASTROPY

if HAS_ASTROPY:
    from egghouse.io import read_fits
else:
    from egghouse.io import read_fits_simple as read_fits
```

---

## 주의사항

1. **FITS 엔디언**: FITS는 항상 big-endian. `read_fits_simple`은 자동 변환.
2. **BMP 행 순서**: BMP는 bottom-up 저장. 읽을 때 자동으로 top-down 변환.
3. **BSCALE/BZERO**: `read_fits_simple`에서 `apply_scaling=True`(기본값)이면 물리값으로 변환.
4. **메모리**: 대용량 FITS 파일은 `read_fits_header` 또는 `read_fits_header_simple`로 헤더만 먼저 확인.
