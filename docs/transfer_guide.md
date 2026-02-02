# egghouse.transfer 사용 가이드

HTTP 파일 다운로드 유틸리티.

---

## 개요

웹 서버에서 파일을 다운로드하기 위한 유틸리티:
- 단일 파일 다운로드 (재시도 로직 포함)
- 디렉토리 목록에서 파일 링크 스크래핑
- 병렬 다운로드 (ThreadPoolExecutor)

---

## 설치

```bash
pip install requests beautifulsoup4
```

---

## 단일 파일 다운로드

### download_single_file

```python
from egghouse.transfer import download_single_file

# 기본 다운로드
success = download_single_file(
    'https://example.com/data.fits',
    '/local/path/data.fits'
)

if success:
    print("Download complete!")
```

### 파라미터

```python
success = download_single_file(
    source_url='https://example.com/file.fits',
    destination='/local/path/file.fits',
    overwrite=False,       # True면 기존 파일 덮어쓰기
    max_retries=3,         # 최대 재시도 횟수
    timeout=30,            # 요청 타임아웃 (초)
    verify_ssl=True        # SSL 인증서 검증
)
```

### 재시도 로직

다운로드 실패 시 지수 백오프로 재시도:
- 1차 시도 실패 → 1초 대기 → 2차 시도
- 2차 시도 실패 → 2초 대기 → 3차 시도
- 3차 시도 실패 → 4초 대기 → 4차 시도 (max_retries=3일 때)

```python
# 불안정한 연결용 설정
success = download_single_file(
    url, destination,
    max_retries=5,   # 더 많은 재시도
    timeout=60       # 더 긴 타임아웃
)
```

### SSL 인증서 무시

내부 서버나 자체 서명 인증서 사용 시:

```python
# 주의: 신뢰할 수 있는 서버에서만 사용
success = download_single_file(
    'https://internal-server/data.fits',
    '/local/data.fits',
    verify_ssl=False
)
```

---

## 파일 목록 가져오기

### get_file_list

웹 디렉토리 목록(Apache/Nginx index)에서 파일 링크를 추출합니다.

```python
from egghouse.transfer import get_file_list

# FITS 파일 목록
files = get_file_list(
    'https://data.server.com/2024/01/',
    extensions=['fits']
)
print(files)  # ['aia_171_001.fits', 'aia_171_002.fits', ...]

# 여러 확장자
files = get_file_list(
    'https://data.server.com/archive/',
    extensions=['fits', 'fits.gz', 'csv']
)
```

### 파라미터

```python
files = get_file_list(
    base_url='https://example.com/data/',
    extensions=['fits', 'csv'],  # 필터링할 확장자
    timeout=30,                  # 요청 타임아웃
    verify_ssl=True              # SSL 검증
)
```

### 반환값

- 디렉토리 목록의 `<a href>` 태그에서 파일명 추출
- 지정한 확장자로 끝나는 파일만 필터링
- 'parent', '..', 'index', 'readme' 등 탐색 링크 제외

---

## 병렬 다운로드

### download_parallel

여러 파일을 동시에 다운로드합니다.

```python
from egghouse.transfer import download_parallel

# 다운로드 작업 목록 (URL, 저장 경로)
tasks = [
    ('https://example.com/file1.fits', '/data/file1.fits'),
    ('https://example.com/file2.fits', '/data/file2.fits'),
    ('https://example.com/file3.fits', '/data/file3.fits'),
]

# 4개 스레드로 병렬 다운로드
result = download_parallel(tasks, parallel=4)

print(f"성공: {result['downloaded']}")
print(f"실패: {result['failed']}")
```

### 파라미터

```python
result = download_parallel(
    download_tasks=tasks,      # [(url, path), ...] 튜플 리스트
    overwrite=False,           # 기존 파일 덮어쓰기
    max_retries=3,             # 파일별 재시도 횟수
    parallel=4,                # 동시 다운로드 수 (1=순차)
    timeout=30,                # 요청 타임아웃
    verify_ssl=True            # SSL 검증
)
```

### 반환값

```python
{
    'downloaded': 10,  # 성공한 파일 수
    'failed': 2        # 실패한 파일 수
}
```

---

## 전체 워크플로우 예시

### 데이터 아카이브에서 다운로드

```python
from egghouse.transfer import get_file_list, download_parallel
import os

# 1. 파일 목록 가져오기
base_url = 'https://data.archive.org/solar/2024/01/'
files = get_file_list(base_url, extensions=['fits'])

print(f"Found {len(files)} files")

# 2. 다운로드 작업 생성
output_dir = '/local/data/2024/01/'
os.makedirs(output_dir, exist_ok=True)

tasks = [
    (f"{base_url}{filename}", f"{output_dir}{filename}")
    for filename in files
]

# 3. 병렬 다운로드
result = download_parallel(tasks, parallel=4, max_retries=5)

print(f"Downloaded: {result['downloaded']}")
print(f"Failed: {result['failed']}")
```

### 날짜 범위 다운로드

```python
from datetime import datetime, timedelta
from egghouse.transfer import get_file_list, download_parallel
import os

base_url = 'https://data.archive.org/solar'
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 31)

all_tasks = []
current = start_date

while current <= end_date:
    date_str = current.strftime('%Y/%m/%d')
    url = f"{base_url}/{date_str}/"

    files = get_file_list(url, extensions=['fits'])

    for f in files:
        src = f"{url}{f}"
        dst = f"/local/data/{date_str}/{f}"
        all_tasks.append((src, dst))

    current += timedelta(days=1)

print(f"Total files to download: {len(all_tasks)}")

# 8개 스레드로 다운로드
result = download_parallel(all_tasks, parallel=8)
```

### 재시도 및 resume

```python
from egghouse.transfer import download_single_file
import os

def download_with_resume(url, path, max_retries=3):
    """이미 있는 파일은 건너뛰고 다운로드"""
    if os.path.exists(path):
        print(f"Skip (exists): {path}")
        return True

    return download_single_file(url, path, max_retries=max_retries)

# 사용
tasks = [...]
for url, path in tasks:
    download_with_resume(url, path)
```

---

## API 요약

| 함수 | 설명 |
|------|------|
| `download_single_file(url, dest, ...)` | 단일 파일 다운로드 (재시도 포함) |
| `get_file_list(url, extensions, ...)` | 디렉토리 목록에서 파일 링크 스크래핑 |
| `download_parallel(tasks, ...)` | 병렬 다운로드 |

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| requests | HTTP 요청 |
| beautifulsoup4 | HTML 파싱 |

설치:
```bash
pip install requests beautifulsoup4
```

---

## 주의사항

1. **서버 부하**: 병렬 다운로드 수(`parallel`)를 너무 높게 설정하면 서버에 부하를 줄 수 있습니다.
2. **SSL 검증**: `verify_ssl=False`는 신뢰할 수 있는 내부 서버에서만 사용하세요.
3. **타임아웃**: 대용량 파일은 `timeout` 값을 늘려야 할 수 있습니다.
4. **디스크 공간**: 다운로드 전 충분한 디스크 공간이 있는지 확인하세요.
