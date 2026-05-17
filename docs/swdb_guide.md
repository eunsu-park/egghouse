# egghouse.swdb 사용 가이드

태양/우주기상 영상 데이터를 위한 DB 도메인 레이어. FITS 파일을 검증된 메타데이터와
DB 레코드로 변환하고, 디렉터리를 멱등하게 등록합니다.

---

## 개요

`egghouse.swdb`는 범용 `egghouse.database` (v0.7) 인프라 위에 올라가는
**도메인 레이어**입니다 (v0.8에서 추가). 역할을 분리해서 보면:

- **참조 스키마** (`SDO_SCHEMA`, `LASCO_SCHEMA`, `SECCHI_SCHEMA`) — 기기별
  테이블의 *모양*만 담은 선언형 dict. setup-sw-db 레퍼런스 프로젝트의
  `schema_config` 블록과 **바이트 단위로 동일**(검증 완료)하므로, 이 상수만으로
  setup-sw-db 호환 `solar_images` DB를 그대로 만들 수 있습니다. DB 이름,
  자격 증명, 데이터 루트 같은 *정책*은 소비 프로젝트의 config에 남습니다.
- **`FitsHandler` ABC + `AiaFitsHandler`** — FITS 파일 한 개를 검증된
  `ValidationResult`, 평탄한 DB 행, 아카이브 경로로 변환합니다.
- **`register_fits_dir`** — 디렉터리 트리를 스캔 → 검증 → 멱등 upsert →
  (선택) 아카이브 이동.

여기서는 **AIA만 출하**됩니다 (undine가 필요로 하는 범위). LASCO, SECCHI,
HMI 등 다른 기기는 각 프로젝트에서 `FitsHandler`를 서브클래싱해 구현합니다.
`astropy`는 `extract_metadata` 안에서 **지연 임포트**되므로 이 서브패키지를
import 하는 것 자체는 가볍고 의존성이 없습니다.

```python
from egghouse.swdb import (
    SDO_SCHEMA, LASCO_SCHEMA, SECCHI_SCHEMA,
    ValidationResult,
    FitsHandler, AiaFitsHandler, AIA_EUV_WAVELENGTHS,
    scan_fits, register_fits_dir, RegisterReport,
)
```

---

## 참조 스키마

각 상수는 `{컬럼: SQL 타입}`과 메타 키(`_primary_key`, `_unique`,
`_indexes`)로 구성된 선언형 dict입니다. 그대로
`egghouse.database.create_tables_from_schema` 또는
`egghouse.database.initialize_database`에 넘길 수 있습니다.

### SDO_SCHEMA

SDO/AIA 영상 테이블. `(telescope, channel, datetime)`을 기본키로,
`file_path`에 UNIQUE 제약을 둡니다. setup-sw-db의 `sdo` 테이블과
바이트 단위로 동일합니다.

```python
SDO_SCHEMA = {
    "telescope": "VARCHAR(10) NOT NULL",
    "channel": "VARCHAR(20) NOT NULL",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "quality": "INTEGER",
    "wavelength": "INTEGER",
    "exposure_time": "REAL",
    "_primary_key": ["telescope", "channel", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"], ["telescope"], ["quality"]],
}
```

### LASCO_SCHEMA

SOHO/LASCO 코로나그래프 테이블. `(camera, datetime)` 기본키.

```python
LASCO_SCHEMA = {
    "camera": "VARCHAR(4) NOT NULL",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "exposure_time": "REAL",
    "filter": "VARCHAR(20)",
    "_primary_key": ["camera", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"]],
}
```

### SECCHI_SCHEMA

STEREO/SECCHI 테이블. `(datatype, spacecraft, instrument, channel, datetime)`
복합 기본키.

```python
SECCHI_SCHEMA = {
    "datatype": "VARCHAR(10) NOT NULL",
    "spacecraft": "VARCHAR(10) NOT NULL",
    "instrument": "VARCHAR(10) NOT NULL",
    "channel": "VARCHAR(20)",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "exposure_time": "REAL",
    "filter": "VARCHAR(20)",
    "wavelength": "INTEGER",
    "_primary_key": ["datatype", "spacecraft", "instrument", "channel", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"], ["spacecraft"], ["instrument"]],
}
```

### setup-sw-db 호환 sdo 테이블 만들기

`egghouse.database.initialize_database`는 DB를 멱등하게 생성하고
스키마의 모든 테이블을 만듭니다. `{테이블명: 스키마}` 형태로 넘깁니다.

```python
from egghouse.database import initialize_database
from egghouse.swdb import SDO_SCHEMA

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "solar_images",
    "user": "username",
    "password": "password",
}

result = initialize_database(db_config, {"sdo": SDO_SCHEMA})
# result -> {"sdo": "created" | "recreated" | "skipped"}
```

생성되는 `sdo` 테이블의 DDL은 setup-sw-db 레퍼런스의 `schema_config`와
바이트 단위로 동일하므로, egghouse를 git-pin 해 쓰는 setup-sw-db 측에
영향을 주지 않습니다.

---

## ValidationResult

FITS 검증 + 메타데이터 추출 결과를 담는 dataclass입니다. 호출 측은
`isinstance(result, str)` 같은 분기 대신 `result.success`를 확인합니다
(setup-sw-db `core/result.py`에서 포팅).

### 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `success` | `bool` | 검증 성공 여부 |
| `metadata` | `Optional[Dict[str, Any]]` | 성공 시 추출된 메타데이터, 실패 시 `None` |
| `error` | `Optional[str]` | 실패 시 에러 분류, 성공 시 `None` |
| `file_path` | `Optional[str]` | 결과가 가리키는 파일 경로 |

에러 분류(`error`) 값: `invalid_file`(파일을 열 수 없음),
`invalid_header`(필수 헤더 누락 또는 비-AIA), `invalid_data`(픽셀 배열
없음/전부 NaN), `non_zero_quality`(`QUALITY != 0`).

### 생성자 메서드

`ValidationResult`는 직접 생성하기보다 두 클래스 메서드로 만듭니다.

```python
from egghouse.swdb import ValidationResult

ok = ValidationResult.ok({"datetime": ..., "telescope": "aia"}, "/data/a.fits")
# ok.success == True, ok.metadata == {...}, ok.error is None

bad = ValidationResult.fail("invalid_header", "/data/b.fits")
# bad.success == False, bad.error == "invalid_header", bad.metadata is None
```

#### 파라미터

- `ValidationResult.ok(metadata, file_path=None)` — 검증 성공.
- `ValidationResult.fail(error, file_path=None)` — 검증 실패. `error`는
  위 에러 분류 문자열.

---

## FitsHandler (ABC)

기기별 FITS 처리 인터페이스. 세 개의 추상 메서드를 구현해야 합니다.

| 메서드 | 시그니처 | 역할 |
|--------|----------|------|
| `extract_metadata` | `(file_path: str) -> ValidationResult` | FITS를 열어 검증하고 결과 반환 |
| `to_db_record` | `(file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]` | 검증된 메타데이터를 DB 행 dict로 평탄화 |
| `target_dir` | `(root: str, metadata: Dict[str, Any]) -> Path` | 검증된 파일의 아카이브 디렉터리(`root` 하위) |

LASCO, SECCHI, HMI 등은 각 프로젝트에서 이 ABC를 서브클래싱하면
`register_fits_dir`를 그대로 재사용할 수 있습니다.

```python
from egghouse.swdb import FitsHandler

class MyInstrumentHandler(FitsHandler):
    def extract_metadata(self, file_path): ...
    def to_db_record(self, file_path, metadata): ...
    def target_dir(self, root, metadata): ...
```

---

## AiaFitsHandler

SDO/AIA EUV Level-1 핸들러. `FitsHandler`의 구체 구현이며, AIA에서만
출하됩니다.

```python
from egghouse.swdb import AiaFitsHandler

handler = AiaFitsHandler(check_data=False, require_quality_zero=False)
result = handler.extract_metadata("/data/aia.lev1_171a.fits")
if result.success:
    row = handler.to_db_record("/data/aia.lev1_171a.fits", result.metadata)
```

### 파라미터

생성자는 키워드 전용 인자만 받습니다:
`AiaFitsHandler(*, check_data=False, require_quality_zero=False)`.

- `check_data` (기본 `False`) — `True`이면 픽셀 배열이 없거나 전부 NaN일 때
  `invalid_data`로 실패시킵니다. 데이터 읽기를 강제하므로 느립니다. 기본은
  헤더 전용 검증으로, 대량 배치에 빠릅니다.
- `require_quality_zero` (기본 `False`) — `True`이면 `QUALITY != 0`일 때
  `non_zero_quality`로 실패시킵니다. 기본값은 모두 등록하고 `quality`를
  저장해 두어 나중에 필터링하는 방식입니다.

### 타임스탬프 정책 (의도적, 문서화된 분기)

`AiaFitsHandler`는 헤더의 **`T_OBS`(UTC)** 를 키로 사용합니다. undine의
관측 그룹핑과 일치하기 때문입니다. setup-sw-db 레퍼런스의 SDO 검증기는
`T_REC`(JSOC slotted record time)를 사용하지만, AIA EUV에서는 `T_OBS`가
자연스러운 관측 시각이며 TAI 변환이 필요 없습니다.

이는 **의도적이고 문서화된 분기**입니다. `sdo` *테이블 모양*은 여전히
setup-sw-db와 호환되며, `datetime` 컬럼의 의미적 출처만 프로젝트별로
다릅니다. `T_OBS`는 `YYYY-MM-DDTHH:MM:SS.sssZ` 형식으로 파싱하며, 후행
`Z`와 주변 공백을 허용합니다. 원본 문자열은 메타데이터에 `t_obs_raw`로
보존됩니다.

압축된 AIA lev1 파일은 이미지가 HDU 1에 있을 수 있어, HDU 1에 `T_OBS`가
있으면 그 헤더/데이터를, 아니면 HDU 0을 사용합니다.

### to_db_record가 만드는 행

`SDO_SCHEMA`와 정확히 맞는 평탄한 dict를 반환합니다:

```python
{
    "telescope": "aia",
    "channel": "171",        # str(int(WAVELNTH))
    "datetime": <datetime>,  # T_OBS (UTC)
    "file_path": "/archive/aia/2026/20260514/aia.lev1_171a.fits",
    "quality": 0,
    "wavelength": 171,
    "exposure_time": 2.9,
}
```

`target_dir(root, metadata)`는
`<root>/aia/<YYYY>/<YYYYMMDD>/` 형태의 아카이브 경로를 돌려줍니다.

---

## scan_fits

디렉터리 트리를 재귀적으로 훑어 FITS 파일 목록(`List[Path]`)을 반환합니다.

```python
from egghouse.swdb import scan_fits

files = scan_fits("/data/incoming", pattern="*.fits", exclude_substrings=("spike",))
```

### 파라미터

`scan_fits(scan_dir, *, pattern='*.fits', exclude_substrings=('spike',))`

- `scan_dir` — 재귀 스캔할 디렉터리. 존재하지 않으면 빈 리스트 반환.
- `pattern` (기본 `'*.fits'`) — `rglob`에 쓰이는 글롭 패턴.
- `exclude_substrings` (기본 `('spike',)`) — 파일 *이름*에 이 부분 문자열이
  (대소문자 무시) 들어가면 제외. 기본값은 AIA `spike` 아티팩트 파일을
  걸러냅니다.

결과는 정렬되어 반환됩니다.

---

## register_fits_dir

FITS 트리를 스캔 → 검증 → 멱등 upsert → (선택) 아카이브 이동까지 한 번에
수행하고 `RegisterReport`를 반환합니다.

```python
from egghouse.swdb import register_fits_dir, AiaFitsHandler

report = register_fits_dir(
    "/data/incoming",
    handler=AiaFitsHandler(),
    table="sdo",
    db_config=db_config,
    conflict_columns=["telescope", "channel", "datetime"],
    move_root="/archive",
    error_dirs={"invalid_header": "_bad/header"},
    parallel=8,
    batch_size=1000,
    verbose=True,
)
print(report.summary())
```

### 파라미터

`register_fits_dir(scan_dir, *, handler, table, db_config, conflict_columns,
move_root=None, error_dirs=None, pattern='*.fits',
exclude_substrings=('spike',), parallel=1, batch_size=1000, verbose=False)`

- `scan_dir` — 재귀 스캔할 디렉터리.
- `handler` — `FitsHandler` 인스턴스 (예: `AiaFitsHandler()`).
- `table` — 대상 DB 테이블명.
- `db_config` — `PostgresManager`에 넘길 kwargs dict.
- `conflict_columns` — 멱등 upsert의 복합 충돌 대상 (예:
  `["telescope", "channel", "datetime"]`).
- `move_root` (기본 `None`) — 설정하면 검증된 파일을
  `handler.target_dir(move_root, metadata)` 아래로 이동합니다. 대상이 이미
  존재하면 `skipped_existing`로 세고 파일을 제자리에 둡니다. `None`이면
  파일을 있는 자리 그대로 등록합니다.
- `error_dirs` (기본 `None`) — `{에러분류: 하위디렉터리}`. `move_root` 하위로
  무효 파일을 옮깁니다. `move_root`가 `None`이면 무시됩니다.
- `pattern` / `exclude_substrings` — `scan_fits`로 전달.
- `parallel` (기본 `1`) — 헤더 검증용 스레드 워커 수. 헤더 전용 검증은
  I/O 바운드이므로 스레드 풀을 씁니다(피클링 제약 없음, 핸들러가 상태를
  가져도 됨).
- `batch_size` (기본 `1000`) — upsert 배치 크기.
- `verbose` (기본 `False`) — 파일별 처리 결과를 출력.

### 멱등성

DB 쓰기는 `egghouse.database.upsert_dataframe`에 위임되며,
`ON CONFLICT DO NOTHING`을 씁니다. `SDO_SCHEMA`의 복합 PK와 별도의
`UNIQUE(file_path)` 제약 양쪽 모두에서 충돌은 에러가 아니라 스킵으로
처리됩니다. 따라서 이미 등록된 트리 위에서 다시 실행하면 아무것도
삽입되지 않습니다. 파일 이동은 해당 배치의 DB 쓰기가 성공한 *뒤에만*
일어납니다.

### 리포트 정합성

`RegisterReport`의 카운트는 다음 항등식을 만족합니다:

```
scanned == valid + sum(errors.values()) + skipped_existing
```

스캔된 모든 파일은 정확히 한 군데(유효 / 에러 분류 / 대상-이미-존재)로
귀속됩니다.

---

## RegisterReport

`register_fits_dir`가 반환하는 dataclass입니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `scanned` | `int` | 스캔된 파일 수 |
| `valid` | `int` | 검증을 통과해 DB 행으로 만들어진 수 |
| `inserted` | `int` | 실제 DB에 삽입된 행 수 (스킵 제외) |
| `skipped_existing` | `int` | 이동 대상이 이미 존재해 건너뛴 수 |
| `errors` | `Dict[str, int]` | 에러 분류별 카운트 |

`inserted`는 멱등 스킵을 제외한 *실제* 삽입 수이므로, 재실행 시
`valid`는 그대로여도 `inserted`는 0일 수 있습니다.

### .summary()

사람이 읽기 좋은 멀티라인 문자열을 반환합니다.

```python
print(report.summary())
# scanned files          : 1240
# valid                  : 1198
# inserted (DB)          : 1198
# skipped (target exists): 0
# errors:
#   invalid_header: 42
```

---

## 엔드투엔드 예제

`SDO_SCHEMA`로 DB를 초기화하고, AIA FITS 디렉터리를 `AiaFitsHandler`로
등록한 뒤 리포트를 출력하는 전체 흐름입니다.

```python
from egghouse.database import initialize_database
from egghouse.swdb import SDO_SCHEMA, AiaFitsHandler, register_fits_dir

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "solar_images",
    "user": "username",
    "password": "password",
}

# 1) setup-sw-db 호환 sdo 테이블 생성 (멱등)
initialize_database(db_config, {"sdo": SDO_SCHEMA})

# 2) AIA FITS 디렉터리 스캔 → 검증 → 멱등 upsert → 아카이브 이동
report = register_fits_dir(
    "/data/incoming/aia",
    handler=AiaFitsHandler(check_data=False, require_quality_zero=False),
    table="sdo",
    db_config=db_config,
    conflict_columns=["telescope", "channel", "datetime"],
    move_root="/archive",
    error_dirs={
        "invalid_file": "_bad/file",
        "invalid_header": "_bad/header",
    },
    parallel=8,
    batch_size=1000,
)

# 3) 결과 요약
print(report.summary())
```

같은 명령을 다시 돌리면 `register_fits_dir`의 멱등성 덕분에 새로 삽입되는
행이 없고(`inserted == 0`), 리포트 카운트는 그대로 정합합니다.
