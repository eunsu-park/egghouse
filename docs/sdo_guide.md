# egghouse.sdo 사용 가이드

SDO (Solar Dynamics Observatory) AIA 및 HMI 데이터 처리 유틸리티.

> **v0.4+ 신규 기능**: JSOC drms export(`jsoc_export`, `aia_euv_query`,
> v0.4), AIA Level 1→1.5 prep 단계(`aia_update_pointing`, `aia_respike`,
> `aia_correct_degradation`, `aia_deconvolve`, `mask_out_of_disk`,
> 캐시 헬퍼, v0.5)는 아래 **JSOC export** 및 **AIA Level 1 → 1.5 prep
> 단계** 절을 참조하세요. 함수 시그니처는 `API_REFERENCE.md`, 변경
> 이력은 `CHANGELOG.md`에도 정리되어 있습니다.

---

## 개요

SDO 모듈은 태양 관측 데이터 처리를 위한 전문 도구를 제공합니다:
- **AIA**: 다파장 EUV/UV 이미지 강도 스케일링
- **HMI**: 자기장 데이터 스케일링 및 벡터장 처리
- **Level 1.5**: Level 1.0 → 1.5 전처리 (north-up, centered)
- **Stacking**: 태양 자전 보정 이미지 스태킹
- **Quality**: QUALITY 키워드 해석 및 데이터 품질 검증
- **DEM**: 다파장 AIA 관측에서 온도 분포 역산 (SITES 알고리즘)

---

## AIA 처리

### aia_intscale

파장별 최적화된 강도 스케일링. 시각화용.

```python
from egghouse.sdo import aia_intscale
from astropy.io import fits

# FITS 파일 읽기 (egghouse.io는 v0.6.0에서 제거됨 — astropy 직접 사용)
data, header = fits.getdata('aia_171.fits', header=True)
exptime = header['EXPTIME']
wavelnth = header['WAVELNTH']

# 스케일링 적용 (uint8 반환)
scaled = aia_intscale(data, exptime, wavelnth)

# float 반환 (추가 처리용)
scaled_float = aia_intscale(data, exptime, wavelnth, to_bytescale=False)
```

### 지원 파장 및 스케일링 방법

| 파장 (Å) | 스케일 | 설명 |
|---------|-------|------|
| 94 | sqrt | Fe XVIII (플레어) |
| 131 | log | Fe VIII/XXI |
| 171 | sqrt | Fe IX (코로나) |
| 193 | log | Fe XII/XXIV |
| 211 | log | Fe XIV |
| 304 | log | He II (천이층) |
| 335 | log | Fe XVI |
| 1600 | linear | C IV + continuum |
| 1700 | linear | Continuum |
| 4500 | linear | Photosphere |

### get_aia_calibration

파장별 보정 파라미터 조회.

```python
from egghouse.sdo import get_aia_calibration

cal = get_aia_calibration(171)
print(f"정규화 노출시간: {cal['norm_exptime']}")
print(f"최소/최대: {cal['vmin']}, {cal['vmax']}")
print(f"스케일 방법: {cal['scale']}")
```

---

## HMI 처리

### hmi_intscale

자기장 데이터를 uint8로 스케일링.

```python
from egghouse.sdo import hmi_intscale
from astropy.io import fits

data, header = fits.getdata('hmi_m.fits', header=True)

# 기본 범위 [-100, 100] Gauss (quiet sun)
scaled = hmi_intscale(data)

# 활동 영역용 넓은 범위
scaled_ar = hmi_intscale(data, vmin=-500, vmax=500)

# 강한 자기장 영역
scaled_strong = hmi_intscale(data, vmin=-2000, vmax=2000)
```

### hmi_field_strength

벡터 자기장 성분에서 총 자기장 강도 계산.

```python
from egghouse.sdo import hmi_field_strength
from astropy.io import fits

# 벡터 자기장 성분 읽기
bx, _ = fits.getdata('hmi_bx.fits', header=True)
by, _ = fits.getdata('hmi_by.fits', header=True)
bz, _ = fits.getdata('hmi_bz.fits', header=True)

# 총 자기장 강도: |B| = sqrt(Bx² + By² + Bz²)
b_total = hmi_field_strength(bx, by, bz)
print(f"최대 자기장: {b_total.max():.1f} Gauss")
```

---

## Level 1.5 변환

Level 1.0 데이터를 표준화된 Level 1.5 포맷으로 변환:
- North-up 정렬 (CROTA2 = 0)
- 태양 중심 정렬 (CRPIX at center)
- 표준 plate scale (0.6 arcsec/px)
- 4096×4096 고정 크기

### to_level15

단일 파일 변환.

```python
from egghouse.sdo import to_level15

# 자동 instrument 감지
m = to_level15('aia_171_lev1.fits')

# 명시적 instrument 지정
m_hmi = to_level15('hmi_m_lev1.fits', instrument='HMI')

# 결과 확인
print(f"Shape: {m.data.shape}")        # (4096, 4096)
print(f"CROTA2: {m.meta['CROTA2']}")   # 0.0
print(f"LVL_NUM: {m.meta['LVL_NUM']}") # 1.5

# FITS 저장
m.save('aia_171_lev15.fits', overwrite=True)
```

### 파라미터

```python
m = to_level15(
    'aia_171.fits',
    instrument=None,           # 'AIA' or 'HMI', None=auto
    target_plate_scale=None,   # arcsec/px (기본 0.6)
    target_size=4096,          # 출력 크기
    order=3,                   # 보간 차수 (0-5)
    missing=0.0                # 패딩 값
)
```

### batch_to_level15

여러 파일 일괄 변환.

```python
from egghouse.sdo import batch_to_level15
import glob

fits_files = glob.glob('/data/aia_171_*.fits')

# 진행 콜백
def progress(current, total, msg):
    print(f"[{current}/{total}] {msg}")

output_files = batch_to_level15(
    fits_files,
    output_dir='/output/level15/',
    instrument='AIA',
    overwrite=False,
    progress_callback=progress
)

print(f"변환 완료: {len(output_files)} 파일")
```

### get_level_info

파일의 처리 레벨 정보 조회.

```python
from egghouse.sdo import get_level_info

info = get_level_info('aia_171.fits')

print(f"Level: {info['level']}")
print(f"CROTA2: {info['crota2']}")
print(f"Is Level 1.5: {info['is_level15']}")

if not info['is_level15']:
    m = to_level15('aia_171.fits')
```

---

## JSOC export (drms 기반, v0.4+)

JSOC(jsoc.stanford.edu)의 DRMS export 시스템을 이용해 SDO/AIA 데이터를
직접 스테이징하고 URL을 받아옵니다. `egghouse.sdo.jsoc` 모듈은 `drms`
패키지를 **soft dependency**로 사용합니다 — 함수 본문 안에서 lazy import
하므로 모듈을 단순 import 하는 것만으로는 `drms`가 필요하지 않습니다
(쿼리 조합 로직은 네트워크 없이 테스트 가능). 받은 URL은
`egghouse.transfer.download_parallel`에 넘겨 재시도/원자적 쓰기를
활용해 내려받는 흐름을 권장합니다.

### aia_euv_query

여러 타임스탬프에 대한 AIA EUV 레코드를 선택하는 DRMS record-set
문자열을 조합합니다. 각 시각마다 `tolerance` 구간을 스캔하며, 여러
시각은 하나의 export 요청으로 연결됩니다.

```python
from datetime import datetime, timedelta
from egghouse.sdo import aia_euv_query

times = [
    datetime(2014, 1, 1, 12, 0, 0),
    datetime(2014, 1, 1, 13, 0, 0),
    datetime(2014, 1, 1, 14, 0, 0),
]

# 기본 6개 DEM 채널, aia.lev1_euv_12s 시리즈
query = aia_euv_query(times)
print(query)
# aia.lev1_euv_12s[2014.01.01_12:00:00_TAI/12s][...]/12s][? WAVELNTH=94 OR ... ?]

# 특정 채널만, tolerance 확장
query_171 = aia_euv_query(
    times,
    wavelengths=[171, 193],
    tolerance=timedelta(seconds=24),
)
```

#### 파라미터

```python
query = aia_euv_query(
    times,                              # datetime 시퀀스 (non-empty)
    wavelengths=AIA_DEM_WAVELENGTHS,    # 기본 6개 DEM 채널 (Å)
    series=AIA_LEV1_EUV_SERIES,         # 'aia.lev1_euv_12s'
    tolerance=timedelta(seconds=12),    # 각 시각 이후 스캔 폭 (>0)
)
```

- naive `datetime`은 JSOC가 기대하는 TAI로 해석됩니다.
- `times` 또는 `wavelengths`가 비어 있거나 `tolerance`가 0 이하면
  `ValueError`가 발생합니다.

### jsoc_export

DRMS export 요청을 제출하고 스테이징이 끝날 때까지 블록한 뒤
스테이징된 파일 URL 리스트를 반환합니다.

```python
from egghouse.sdo import jsoc_export

urls = jsoc_export(
    query,
    email='you@example.com',   # jsoc.stanford.edu에 등록된 이메일
)
print(f"스테이징된 파일: {len(urls)}개")
```

#### 파라미터

```python
urls = jsoc_export(
    query,                  # aia_euv_query 등이 만든 record-set 문자열
    email='you@example.com',# JSOC export 등록 이메일 (필수, 키워드)
    method='url',           # 'url'은 스테이징 완료까지 서버에서 블록
    protocol='fits',        # 스테이징 파일 프로토콜
    client=None,            # 재사용할 drms.Client (None이면 새로 생성)
)
```

- `method='url'`은 데이터셋이 스테이징될 때까지 서버 측에서 대기 후
  구체적인 URL을 반환합니다.
- export가 성공하지 못하면 `RuntimeError`가 발생합니다. 매칭 레코드가
  없으면 빈 리스트가 반환될 수 있습니다.

### cached_correction_table / cached_pointing_table

배치 작업에서 매 레코드마다 JSOC에서 다시 가져오면 느린 aiapy
보정 테이블을 디스크에 pickle 캐시합니다. 최초 호출 시 fetch 후 기록,
이후 호출은 디스크에서 역직렬화하여 네트워크 왕복을 건너뜁니다.

```python
from datetime import datetime
from egghouse.sdo import cached_correction_table, cached_pointing_table

# degradation 보정 테이블 (aiapy.calibrate.util.get_correction_table)
corr = cached_correction_table('cache/aia_correction.pkl')

# 시간 구간별 pointing 테이블 (end는 aiapy 관례상 exclusive)
pointing = cached_pointing_table(
    'cache/aia_pointing.pkl',
    start=datetime(2014, 1, 1),
    end=datetime(2014, 1, 2),
)
```

- `cached_pointing_table`의 `start`/`end`는 캐시를 **새로 채울 때만**
  사용됩니다. 오래된 캐시 파일은 그대로 재사용되므로, 다른 시간
  창이 필요하면 파일을 먼저 삭제하세요.
- 두 함수 모두 부모 디렉터리를 필요 시 생성합니다.

### AIA_LEV1_EUV_SERIES 상수

```python
from egghouse.sdo import AIA_LEV1_EUV_SERIES

print(AIA_LEV1_EUV_SERIES)  # 'aia.lev1_euv_12s'
```

### End-to-end: query → export → 병렬 다운로드

```python
from datetime import datetime
from egghouse.sdo import aia_euv_query, jsoc_export
from egghouse.transfer import download_parallel

# 1. 여러 타임스탬프에 대한 record-set 조합
times = [
    datetime(2014, 1, 1, 12, 0, 0),
    datetime(2014, 1, 1, 13, 0, 0),
    datetime(2014, 1, 1, 14, 0, 0),
]
query = aia_euv_query(times, wavelengths=[94, 131, 171, 193, 211, 335])

# 2. JSOC export → 스테이징된 URL 리스트
urls = jsoc_export(query, email='you@example.com')

# 3. (url, dest) 태스크로 변환 후 병렬 다운로드
import os
os.makedirs('data', exist_ok=True)
tasks = [(u, os.path.join('data', u.rsplit('/', 1)[-1])) for u in urls]

result = download_parallel(tasks, parallel=4, max_retries=3)
print(f"다운로드: {result['downloaded']}, 실패: {result['failed']}")
```

---

## AIA Level 1 → 1.5 prep 단계 (v0.5+)

`to_level15`(위 절)는 AIA/HMI 공통으로 회전·리샘플링·패딩을
수행하지만, aiapy 기반의 추가 prep 단계는 다루지 않습니다.
`egghouse.sdo.prep` 모듈은 그 빈자리를 채우는 aiapy 래퍼들을
제공합니다. `sunpy`/`aiapy`/`astropy`는 모두 함수 본문 안에서 lazy
import 되므로, 모듈 import 자체는 가볍고 aiapy를 강제하지 않습니다.

**aiapy 정식 처리 순서.** AIA Level 1 → 1.5 변환의 표준(canonical)
순서는 다음과 같습니다:

```
update_pointing → respike → correct_degradation → deconvolve
                → register / to_level15 (회전·리샘플)
```

`register`/`to_level15`(L1.5 등록)는 **마지막**에 적용합니다.
`aia_correct_degradation`과 `aia_deconvolve`는 보정/PSF가 정의된
AIA 채널(EUV + 304 Å: 94, 131, 171, 193, 211, 304, 335 Å) 외의
파장에 대해서는 입력 맵을 **그대로 반환**하므로, 이종 채널이 섞인
배치에 안전하게 일괄 적용할 수 있습니다.

### aia_update_pointing

JSOC master pointing 테이블로 AIA WCS 키워드를 갱신합니다
(`aiapy.calibrate.update_pointing` 래퍼). 배치에서는
`cached_pointing_table`로 미리 받아둔 테이블을 넘겨 매 레코드
재요청을 피하세요.

```python
from sunpy.map import Map
from egghouse.sdo import aia_update_pointing, cached_pointing_table
from datetime import datetime

pointing = cached_pointing_table(
    'cache/pointing.pkl',
    start=datetime(2014, 1, 1), end=datetime(2014, 1, 2),
)

m = Map('aia_171_lev1.fits')
m = aia_update_pointing(m, pointing_table=pointing)
# pointing_table=None이면 aiapy가 매번 JSOC에서 fetch
```

### aia_respike

Level 1 파이프라인이 제거한 spike 픽셀을 다시 주입합니다
(`aiapy.calibrate.respike` 래퍼). `spikes`를 주지 않으면
`aiapy.calibrate.fetch_spikes`로 가져오는데, 레코드당 JSOC 왕복이
발생하므로 배치에서는 미리 fetch 해 넘기는 것이 좋습니다.

```python
from egghouse.sdo import aia_respike

m = aia_respike(m)                # spikes=None → 레코드별 fetch
# m = aia_respike(m, spikes=prefetched_spikes)
```

### aia_correct_degradation

시간 의존 유효면적(effective-area) 보정을 적용합니다
(`aiapy.calibrate.correct_degradation` 래퍼). 보정이 정의되지 않은
파장이면 맵을 그대로 반환합니다. `cached_correction_table` 결과를
넘겨 JSOC fetch를 배치 전체로 분할상환하세요.

```python
from egghouse.sdo import aia_correct_degradation, cached_correction_table

corr = cached_correction_table('cache/correction.pkl')
m = aia_correct_degradation(m, correction_table=corr)
```

### aia_deconvolve

AIA 맵을 PSF deconvolution 합니다 (`aiapy.psf.deconvolve` 래퍼).
PSF 계산(`aiapy.psf.psf(...)`)이 채널당 수 분으로 가장 비싼
부분이므로, `cached_aia_psfs`로 `{파장: PSF}` dict를 미리 만들어
넘기세요. PSF가 제공되지 않는 파장이면 맵을 그대로 반환합니다.

```python
from egghouse.sdo import aia_deconvolve, cached_aia_psfs

psfs = cached_aia_psfs('cache/aia_psfs.pkl')   # 표준 7개 채널
m = aia_deconvolve(m, psfs=psfs)
# psfs=None이면 호출 시 해당 채널 PSF를 직접 계산 (수 분)
```

### cached_aia_psfs

AIA PSF를 채널별로 계산해 `{파장(int): PSF 배열}` dict로 pickle
캐시합니다. 캐시 파일이 없거나 요청한 파장을 **모두** 포함하지
않으면 전체 세트로 재생성합니다.

```python
from egghouse.sdo import cached_aia_psfs

# 기본: 7개 보정 채널 (94, 131, 171, 193, 211, 304, 335)
psfs = cached_aia_psfs('cache/aia_psfs.pkl')

# 일부 채널만
psfs_subset = cached_aia_psfs('cache/aia_psfs.pkl', wavelengths=[171, 193])
```

### mask_out_of_disk

태양 림 바깥 픽셀을 sentinel 값으로 채운 `sunpy.Map` 복사본을
반환합니다 (입력은 변경하지 않음). 림 반경은 헤더의 `R_SUN`(픽셀
단위)이 있으면 그것을, 없으면 `RSUN_OBS / CDELT1`을 사용하고, 둘
다 없으면 `KeyError`를 던집니다. 다운스트림(예: DEM 모델 학습
루프)이 off-disk 영역을 무시하도록 표시할 때 유용합니다.

```python
from egghouse.sdo import mask_out_of_disk

masked = mask_out_of_disk(m, fill_value=-5000.0)  # 기본 sentinel -5000.0
```

### 배치 워크플로우: 캐시/PSF 1회 선취 후 일괄 prep

```python
from datetime import datetime
import glob
from sunpy.map import Map
from egghouse.sdo import (
    cached_pointing_table,
    cached_correction_table,
    cached_aia_psfs,
    aia_update_pointing,
    aia_respike,
    aia_correct_degradation,
    aia_deconvolve,
    to_level15,
    mask_out_of_disk,
)

# 1. 느린 테이블/PSF를 배치 시작 전 1회만 선취 (디스크 캐시)
pointing = cached_pointing_table(
    'cache/pointing.pkl',
    start=datetime(2014, 1, 1), end=datetime(2014, 1, 2),
)
corr = cached_correction_table('cache/correction.pkl')
psfs = cached_aia_psfs('cache/aia_psfs.pkl')

# 2. 각 파일에 aiapy 정식 순서로 prep 적용
for path in sorted(glob.glob('/data/aia_*_lev1.fits')):
    m = Map(path)
    m = aia_update_pointing(m, pointing_table=pointing)
    m = aia_respike(m)
    m = aia_correct_degradation(m, correction_table=corr)
    m = aia_deconvolve(m, psfs=psfs)
    # 3. 마지막에 L1.5 등록 (회전·리샘플·패딩)
    m = to_level15(m)
    # 4. (선택) off-disk 마스킹
    m = mask_out_of_disk(m)
    m.save(path.replace('_lev1', '_lev15'), overwrite=True)
```

비-AIA 또는 보정 미정의 파장이 섞여 있어도
`aia_correct_degradation`/`aia_deconvolve`가 해당 맵을 그대로
통과시키므로 위 루프를 그대로 사용할 수 있습니다.

---

## 태양 자전 보정 스태킹

시계열 이미지에서 태양 자전으로 인한 이동을 보정하여 스태킹.

### Stacking 클래스

```python
from egghouse.sdo import Stacking
import glob

fits_files = sorted(glob.glob('/data/hmi_m_*.fits'))

# 기본 사용 (리스트 반환)
stacker = Stacking(nb_stack=21)
aligned_list = stacker.run(fits_files)

# mean 결합
stacker = Stacking(nb_stack=21, method='mean')
mean_image = stacker.run(fits_files)

# median 결합
stacker = Stacking(nb_stack=21, method='median')
median_image = stacker.run(fits_files)

# sigma-clipped mean
stacker = Stacking(nb_stack=21, method='sigma_clipped', sigma_lower=3, sigma_upper=3)
clipped_image = stacker.run(fits_files)
```

### Stacking 파라미터

```python
stacker = Stacking(
    nb_stack=21,                    # 스태킹할 이미지 수
    solar_rot_period=None,          # 자전 주기 (일), None=Snodgrass 차등자전
    crop_size=512,                  # 크롭 영역 크기
    cadence_seconds=None,           # 촬영 간격 (초), None=자동 감지
    latitude_deg=0.0,               # 차등자전 적용 위도
    method='list',                  # 결합 방법
    sigma_lower=3.0,                # sigma clipping 하한
    sigma_upper=3.0                 # sigma clipping 상한
)
```

### 차등 자전 (Snodgrass Model)

태양은 위도에 따라 자전 속도가 다릅니다 (적도가 가장 빠름).

```python
from egghouse.sdo import snodgrass_rotation_rate

# 적도 (0°): 14.71 deg/day
rate_eq = snodgrass_rotation_rate(0)

# 위도 30°: ~13.8 deg/day
rate_30 = snodgrass_rotation_rate(30)

# 위도 60°: ~12.0 deg/day
rate_60 = snodgrass_rotation_rate(60)

print(f"적도: {rate_eq:.2f} deg/day")
print(f"30°: {rate_30:.2f} deg/day")
print(f"60°: {rate_60:.2f} deg/day")
```

**Snodgrass 공식:**
```
ω(B) = A + B·sin²(B) + C·sin⁴(B)
```
- A = 14.71 (적도 회전률)
- B = -2.39
- C = -1.78

### solar_rotation_shift

시간 경과에 따른 픽셀 이동량 계산.

```python
from egghouse.sdo import solar_rotation_shift

rsun_pixels = 1600  # 태양 반경 (픽셀)
time_hours = 1.0    # 시간 오프셋

# 적도에서의 이동
shift_eq = solar_rotation_shift(rsun_pixels, time_hours, latitude_deg=0)
print(f"적도 1시간 이동: {shift_eq:.2f} pixels")

# 고위도에서의 이동 (느림)
shift_60 = solar_rotation_shift(rsun_pixels, time_hours, latitude_deg=60)
print(f"60° 1시간 이동: {shift_60:.2f} pixels")

# 고정 자전 주기 사용 (Carrington)
shift_carr = solar_rotation_shift(rsun_pixels, time_hours, rotation_period_days=25.38)
```

### cross_correlate_shift

FFT 기반 위상 상관으로 서브픽셀 정렬.

```python
from egghouse.sdo import cross_correlate_shift
from scipy.ndimage import shift as ndimage_shift

# 두 이미지 간 이동량 계산
dy, dx = cross_correlate_shift(reference_image, target_image)
print(f"이동량: dy={dy:.2f}, dx={dx:.2f} pixels")

# 이동 보정 적용
aligned = ndimage_shift(target_image, (dy, dx), order=3)
```

### StreamingStackAccumulator

메모리 효율적인 스트리밍 스태킹 (Welford 알고리즘).

```python
from egghouse.sdo import StreamingStackAccumulator
from astropy.io import fits

# 대용량 파일 처리
acc = StreamingStackAccumulator(shape=(4096, 4096))

for fits_file in large_file_list:
    data, _ = fits.getdata(fits_file, header=True)
    acc.add(data)

# 결과 추출
mean_image = acc.get_mean()
std_image = acc.get_std()
variance = acc.get_variance()

print(f"처리된 이미지 수: {acc.count}")
```

---

## 코어 유틸리티

### parse_fits_header

SDO FITS 헤더에서 주요 키워드 추출.

```python
from egghouse.sdo import parse_fits_header

header = parse_fits_header('aia_171.fits')

print(f"관측 시간: {header['date_obs']}")
print(f"파장: {header['wavelnth']} Å")
print(f"노출 시간: {header['exptime']} s")
print(f"Plate scale: {header['cdelt1']} arcsec/px")
print(f"CROTA2: {header['crota2']}°")
print(f"태양 반경: {header['rsun_obs']} arcsec")
```

### validate_sdo_image

이미지 유효성 검증.

```python
from egghouse.sdo import validate_sdo_image

# 표준 SDO 크기 검증
validate_sdo_image(data)  # 4096x4096

# 크기 검증 건너뛰기
validate_sdo_image(data, expected_shape=None)

# 커스텀 크기 검증
validate_sdo_image(resized_data, expected_shape=(1024, 1024))
```

### get_solar_disk_params

헤더에서 태양 디스크 파라미터 계산.

```python
from egghouse.sdo import parse_fits_header, get_solar_disk_params

header = parse_fits_header('aia_171.fits')
params = get_solar_disk_params(header)

print(f"태양 중심: ({params['center_x']:.1f}, {params['center_y']:.1f})")
print(f"태양 반경: {params['radius_pixels']:.1f} pixels")
print(f"Plate scale: {params['plate_scale']:.3f} arcsec/px")
```

---

## 상수

```python
from egghouse.sdo import (
    AIA_PLATE_SCALE,       # 0.6 arcsec/px
    HMI_PLATE_SCALE,       # 0.6 arcsec/px
    SDO_IMAGE_SIZE,        # 4096 pixels
    SOLAR_ROTATION_PERIOD, # 25.38 days (Carrington)
    SNODGRASS_A,           # 14.71 deg/day
    SNODGRASS_B,           # -2.39
    SNODGRASS_C,           # -1.78
    HMI_CADENCE_45S,       # 45.0 seconds
    HMI_CADENCE_720S,      # 720.0 seconds
    AIA_CALIBRATION,       # 파장별 보정 테이블
)
```

---

## 전체 워크플로우 예시

### AIA 데이터 처리

```python
from astropy.io import fits, write_fits
from egghouse.sdo import to_level15, aia_intscale
from egghouse.image import circle_mask, resize_image

# 1. Level 1.5 변환
m = to_level15('aia_171_lev1.fits')

# 2. 강도 스케일링
scaled = aia_intscale(m.data, m.meta['EXPTIME'], 171)

# 3. 태양 디스크 마스킹
mask = circle_mask(4096, radius=1600)
masked = np.where(mask, scaled, 0)

# 4. 리사이즈
resized = resize_image(masked, (512, 512))
```

### HMI 스태킹 워크플로우

```python
from egghouse.sdo import Stacking, to_level15, hmi_intscale
import glob

# 1. Level 1.5 변환
fits_files = sorted(glob.glob('/data/hmi_m_*.fits'))

# 2. 스태킹 (21장 mean)
stacker = Stacking(
    nb_stack=21,
    method='mean',
    latitude_deg=15,  # 대상 위도
    crop_size=512
)
stacked = stacker.run(fits_files)

# 3. 시각화용 스케일링
display = hmi_intscale(stacked, vmin=-200, vmax=200)
```

---

## 의존성

| 기능 | 필수 | 선택 |
|------|------|------|
| AIA/HMI 스케일링 | numpy | - |
| Level 1.5 변환 | - | sunpy, astropy |
| Stacking | numpy, scipy | sunpy, astropy |
| DEM 분석 | numpy | aiapy (정확한 응답 함수) |
| FITS 헤더 파싱 | - | astropy |

설치:
```bash
# 기본 (스케일링만)
pip install numpy scipy

# SDO 전체 기능
pip install "egghouse[sdo]"

# DEM 분석 포함
pip install "egghouse[dem]"

# 모든 기능
pip install "egghouse[all]"
```

---

## QUALITY 키워드 해석

SDO FITS 헤더의 QUALITY 키워드는 데이터 품질 문제를 나타내는 32비트 정수입니다.
각 비트는 특정 품질 이슈를 나타냅니다.

### decode_quality

QUALITY 값을 사람이 읽을 수 있는 형태로 디코딩.

```python
from egghouse.sdo import decode_quality

# 정상 데이터
result = decode_quality(0)
print(result)
# [{'bit': -1, 'hex': '0x0', 'description': 'nominal', 'severity': 'ok'}]

# ISS 루프 열림 (비트 17)
result = decode_quality(0x20000)
print(result)
# [{'bit': 17, 'hex': '0x20000', 'description': 'ISS loop open', 'severity': 'caution'}]

# 여러 문제 (HMI 데이터)
result = decode_quality(0x30000, instrument="HMI")
for issue in result:
    print(f"[Bit {issue['bit']}] {issue['description']} ({issue['severity']})")
```

### format_quality

포맷된 문자열로 출력.

```python
from egghouse.sdo import format_quality

# 상세 출력
print(format_quality(0x30000, instrument="HMI"))
# QUALITY = 0x30000 (196608)
# Status: 2 issue(s) detected
#   [Bit 16] 0x10000: Dark image (info)
#   [Bit 17] 0x20000: ISS loop open (caution)

# 간략 출력
print(format_quality(0x30000, verbose=False))
```

### is_quality_ok

데이터가 분석에 사용 가능한지 확인.

```python
from egghouse.sdo import is_quality_ok

# 기본 모드: minor 이슈는 무시
is_quality_ok(0)         # True - 정상
is_quality_ok(0x1)       # True - Flatfield 미적용 (minor)
is_quality_ok(0x2000)    # False - 일식 중 (severe)

# strict 모드: 모든 이슈 체크
is_quality_ok(0x1, strict=True)  # False

# 특정 비트 무시
is_quality_ok(0x2000, ignore_bits=[13])  # True - 일식 비트 무시
```

### get_quality_summary

품질 정보 요약 딕셔너리.

```python
from egghouse.sdo import get_quality_summary

summary = get_quality_summary(0x30000)
print(f"정상 여부: {summary['is_nominal']}")      # False
print(f"사용 가능: {summary['is_usable']}")       # False
print(f"심각도 카운트: {summary['severity_counts']}")  # {'info': 1, 'caution': 1}
print(f"활성 비트: {summary['active_bits']}")     # [16, 17]
```

### print_all_quality_bits

모든 비트 정의 출력 (참조용).

```python
from egghouse.sdo import print_all_quality_bits

# AIA 비트 정의 출력
print_all_quality_bits("AIA")

# HMI 비트 정의 출력
print_all_quality_bits("HMI")
```

### 주요 QUALITY 비트

| 비트 | Hex | 설명 | 심각도 |
|-----|-----|------|--------|
| 0 | 0x1 | Flatfield 데이터 없음 | minor |
| 8 | 0x100 | 일부 픽셀 누락 | warning |
| 9 | 0x200 | 1% 이상 픽셀 누락 | warning |
| 10 | 0x400 | 5% 이상 픽셀 누락 | caution |
| 11 | 0x800 | 25% 이상 픽셀 누락 | severe |
| 13 | 0x2000 | 일식 중 | severe |
| 16 | 0x10000 | 다크 이미지 | info |
| 17 | 0x20000 | ISS 루프 열림 | caution |
| 18 | 0x40000 | 보정용 이미지 | info |
| 30 | 0x40000000 | Quicklook 이미지 | info |
| 31 | 0x80000000 | 이미지 없음 | severe |

### 품질 필터링 워크플로우

```python
from egghouse.sdo import is_quality_ok, get_quality_summary
from astropy.io import fits  # egghouse.io는 v0.6.0에서 제거됨
import glob

fits_files = glob.glob('/data/aia_171_*.fits')

# 품질 좋은 파일만 필터링
good_files = []
for f in fits_files:
    header = fits.getheader(f)
    quality = header.get('QUALITY', 0)

    if is_quality_ok(quality):
        good_files.append(f)
    else:
        summary = get_quality_summary(quality)
        print(f"제외: {f} - {summary['severity_counts']}")

print(f"사용 가능: {len(good_files)}/{len(fits_files)} 파일")
```

---

## DEM (Differential Emission Measure) 분석

다파장 AIA 관측으로부터 온도별 방출 측도(DEM)를 역산합니다.
SITES (Simple Iterative Temperature Emission Solver) 알고리즘 사용.

### 기본 개념

DEM은 코로나의 온도 분포를 나타냅니다:
```
I(λ) = ∫ K(T,λ) × DEM(T) × dT
```
- `I(λ)`: 관측 강도 (DN/s)
- `K(T,λ)`: 온도 응답 함수
- `DEM(T)`: Differential Emission Measure (cm⁻⁵ K⁻¹)

6개의 AIA EUV 채널 (94, 131, 171, 193, 211, 335 Å)을 사용하여
역산 문제를 풀어 온도 분포를 추정합니다.

### 온도 응답 함수

```python
from egghouse.dem import get_default_temperatures
from egghouse.sdo import get_temperature_response

# 기본 온도 그리드 (10^5.5 ~ 10^7.5 K)
temps = get_default_temperatures(n_bins=100)

# 온도 응답 함수 획득 (v0.3.0+)
# aiapy 0.12에서 Channel.temperature_response가 제거되어,
# 정식 CHIANTI 기반 응답은 SSW aia_get_response .npz 테이블에서 로드합니다
# (예: demregpy에 동봉된 response_matrix.npz). ssw_table_path 없이 호출하면
# aiapy 설치 환경에서는 NotImplementedError가 발생합니다.
response = get_temperature_response(
    temperatures=temps,
    ssw_table_path='response_matrix.npz',
)
print(f"Response shape: {response.shape}")  # (100, 6)

# 또는 SSW 로더를 직접 호출:
import numpy as np
from egghouse.dem import load_ssw_temperature_response
response = load_ssw_temperature_response(
    'response_matrix.npz',
    log_temperatures=np.log10(temps),
)
```

### 단일 픽셀 DEM 역산

```python
from egghouse.dem import dem_sites_pixel
import numpy as np

# 6채널 강도 (DN/s)
intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
errors = intensities * 0.1  # 10% 오차

# SITES 알고리즘으로 DEM 역산
dem, info = dem_sites_pixel(
    intensities,
    errors,
    response,
    temps,
    max_iter=100,
    tol=1e-4
)

print(f"수렴: {info['converged']}")
print(f"반복 횟수: {info['iterations']}")
print(f"Chi-squared: {info['chi2']:.2f}")
print(f"DEM peak: {dem.max():.2e} cm^-5 K^-1")
```

### 전체 맵 DEM 처리

```python
from egghouse.dem import dem_map

# image_cube: (height, width, 6) 형태
# error_cube: 동일 형태

dem_cube, info = dem_map(
    image_cube,
    error_cube,
    response,
    temps,
    chunk_size=512,  # 메모리 효율을 위한 청크 크기
    max_iter=100,
)

print(f"DEM cube shape: {dem_cube.shape}")  # (height, width, n_temps)
print(f"처리된 픽셀: {info['n_pixels']}")
```

### 파생량 계산

```python
from egghouse.dem import get_emission_measure, get_mean_temperature

# 총 방출 측도: EM = ∫ DEM × dT
em = get_emission_measure(dem, temps)
print(f"Emission Measure: {em:.2e} cm^-5")

# 특정 온도 범위만 적분
em_hot = get_emission_measure(dem, temps, t_min=1e6, t_max=1e7)

# DEM 가중 평균 온도
t_mean = get_mean_temperature(dem, temps)
print(f"Mean Temperature: {t_mean/1e6:.2f} MK")

# 맵 전체에 적용
em_map = get_emission_measure(dem_cube, temps)
t_map = get_mean_temperature(dem_cube, temps)
```

### 오차 추정 (Monte Carlo)

```python
from egghouse.dem.utils import compute_dem_errors

# Monte Carlo 방법으로 DEM 불확도 추정
dem_errors = compute_dem_errors(
    dem,
    intensities,
    errors,
    response,
    temps,
    n_monte_carlo=100
)
print(f"DEM error range: {dem_errors.min():.2e} - {dem_errors.max():.2e}")
```

### 실제 데이터 워크플로우

```python
from datetime import datetime
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
import astropy.units as u
from egghouse.dem import get_default_temperatures, dem_map, get_emission_measure, get_mean_temperature
from egghouse.sdo import get_temperature_response

# 1. AIA 데이터 다운로드
obs_time = datetime(2024, 1, 15, 12, 0, 0)
wavelengths = [94, 131, 171, 193, 211, 335]

files = {}
for wave in wavelengths:
    result = Fido.search(
        a.Time(obs_time, obs_time),
        a.Instrument("AIA"),
        a.Wavelength(wave * u.angstrom),
    )
    downloaded = Fido.fetch(result, path='./data/')
    files[wave] = downloaded[0]

# 2. 이미지 큐브 생성
maps = [Map(files[w]) for w in wavelengths]
image_cube = np.stack([
    m.data / m.exposure_time.to(u.s).value
    for m in maps
], axis=-1)
error_cube = image_cube * 0.1

# 3. 온도 응답 함수 (v0.3.0+: SSW 테이블 경로 필요 — 위 설명 참조)
temps = get_default_temperatures(n_bins=100)
response = get_temperature_response(
    wavelengths=wavelengths,
    temperatures=temps,
    ssw_table_path='response_matrix.npz',
)

# 4. DEM 계산
dem_cube, info = dem_map(
    image_cube,
    error_cube,
    response,
    temps,
    chunk_size=512,
)

# 5. 파생량
em = get_emission_measure(dem_cube, temps)
t_mean = get_mean_temperature(dem_cube, temps)

print(f"EM range: {em.min():.2e} - {em.max():.2e} cm^-5")
print(f"T_mean range: {t_mean.min()/1e6:.2f} - {t_mean.max()/1e6:.2f} MK")
```

### DEM 상수

```python
from egghouse.sdo import HAS_AIAPY
from egghouse.sdo import AIA_DEM_WAVELENGTHS

print(f"aiapy 설치: {HAS_AIAPY}")
print(f"DEM 파장: {AIA_DEM_WAVELENGTHS} Å")  # [94, 131, 171, 193, 211, 335]
```

### 주의사항

1. **aiapy 설치 권장**: 정확한 온도 응답 함수를 위해 aiapy 설치 필요
   ```bash
   pip install aiapy
   ```

2. **메모리 사용량**: 4096×4096 맵의 경우 출력 DEM 큐브가 ~6.5 GB
   - `chunk_size` 파라미터로 조절 가능
   - `mask` 파라미터로 필요한 영역만 처리

3. **양수 제약**: SITES 알고리즘은 DEM ≥ 0 제약 적용

4. **수렴 확인**: `info['converged']`로 수렴 여부 확인

---

## 참고문헌

- Boerner, P. et al. 2012, Solar Physics, 275, 41 (AIA calibration)
- Snodgrass, H.B. 1983, ApJ, 270, 288 (Differential rotation)
- Lemen, J.R. et al. 2012, Solar Physics, 275, 17 (AIA instrument)
- Schou, J. et al. 2012, Solar Physics, 275, 229 (HMI instrument)
- Morgan, H. & Pickering, J. 2019, Solar Physics, 294, 135 (SITES algorithm)
- Hannah, I.G. & Kontar, E.P. 2012, A&A, 539, A146 (DEM regularization)
- JSOC QUALITY documentation: http://jsoc.stanford.edu/jsocwiki/Lev1qualBits
