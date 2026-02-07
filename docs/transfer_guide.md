# egghouse.transfer 사용 가이드

파일 전송 유틸리티 (HTTP, FTP, SFTP 지원).

---

## 개요

다양한 프로토콜로 파일을 다운로드/업로드하기 위한 유틸리티:

**HTTP**
- 단일 파일 다운로드 (재시도 로직 포함)
- 디렉토리 목록에서 파일 링크 스크래핑
- 병렬 다운로드 (ThreadPoolExecutor)

**FTP** (추가 의존성 없음)
- FTP 서버 연결 (context manager)
- 파일 다운로드/업로드
- 디렉토리 파일 목록 조회
- 병렬 다운로드/업로드

**SFTP** (paramiko 필요)
- SSH 기반 보안 파일 전송
- 비밀번호 또는 키 파일 인증
- 파일 다운로드/업로드
- 병렬 다운로드/업로드

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

## FTP 다운로드/업로드

Python 표준 라이브러리 `ftplib`를 사용하며, 추가 의존성이 필요 없습니다.

### FTP 연결

```python
from egghouse.transfer import ftp_connection, ftp_list_files, ftp_download_file

# Context manager로 연결 (자동 연결 해제)
with ftp_connection('ftp.example.com', user='anonymous', password='') as ftp:
    # 파일 목록 조회
    files = ftp_list_files(ftp, '/data/', extensions=['fits'])
    print(f"Found {len(files)} files")

    # 단일 파일 다운로드
    ftp_download_file(ftp, '/data/file.fits', 'local_file.fits')
```

### FTP 연결 파라미터

```python
with ftp_connection(
    host='ftp.example.com',
    port=21,                # FTP 포트 (기본: 21)
    user='anonymous',       # 사용자명 (기본: anonymous)
    password='',            # 비밀번호
    timeout=30,             # 연결 타임아웃 (초)
    passive=True            # 패시브 모드 (기본: True)
) as ftp:
    ...
```

### FTP 파일 업로드

```python
from egghouse.transfer import ftp_connection, ftp_upload_file

with ftp_connection('ftp.example.com', user='admin', password='secret') as ftp:
    # 단일 파일 업로드
    success = ftp_upload_file(ftp, 'local_file.fits', '/remote/path/file.fits')

    if success:
        print("Upload complete!")
```

### FTP 병렬 다운로드

```python
from egghouse.transfer import ftp_download_parallel

# 다운로드 작업 목록 (원격 경로, 로컬 경로)
tasks = [
    ('/data/file1.fits', '/local/file1.fits'),
    ('/data/file2.fits', '/local/file2.fits'),
    ('/data/file3.fits', '/local/file3.fits'),
]

# 4개 연결로 병렬 다운로드
result = ftp_download_parallel(
    host='ftp.example.com',
    download_tasks=tasks,
    user='anonymous',
    parallel=4,
    max_retries=3
)

print(f"성공: {result['downloaded']}")
print(f"실패: {result['failed']}")
```

### FTP 병렬 업로드

```python
from egghouse.transfer import ftp_upload_parallel

# 업로드 작업 목록 (로컬 경로, 원격 경로)
tasks = [
    ('/local/file1.fits', '/upload/file1.fits'),
    ('/local/file2.fits', '/upload/file2.fits'),
]

result = ftp_upload_parallel(
    host='ftp.example.com',
    upload_tasks=tasks,
    user='admin',
    password='secret',
    parallel=4
)

print(f"성공: {result['uploaded']}")
print(f"실패: {result['failed']}")
```

---

## SFTP 다운로드/업로드

SSH 기반 보안 파일 전송. `paramiko` 라이브러리가 필요합니다.

### SFTP 설치

```bash
pip install paramiko
```

### SFTP 의존성 확인

```python
from egghouse.transfer import HAS_PARAMIKO

if HAS_PARAMIKO:
    print("SFTP 사용 가능")
else:
    print("paramiko 설치 필요: pip install paramiko")
```

### SFTP 연결 (비밀번호)

```python
from egghouse.transfer import sftp_connection, sftp_download_file, sftp_list_files

with sftp_connection(
    host='sftp.example.com',
    user='admin',
    password='secret'
) as sftp:
    # 파일 목록 조회
    files = sftp_list_files(sftp, '/data/', extensions=['fits'])

    # 파일 다운로드
    sftp_download_file(sftp, '/data/file.fits', 'local_file.fits')
```

### SFTP 연결 (SSH 키)

```python
from egghouse.transfer import sftp_connection, sftp_upload_file

with sftp_connection(
    host='sftp.example.com',
    port=22,
    user='admin',
    key_file='~/.ssh/id_rsa'  # SSH 개인키 경로
) as sftp:
    # 파일 업로드
    sftp_upload_file(sftp, 'local_file.fits', '/remote/path/file.fits')
```

### SFTP 연결 파라미터

```python
with sftp_connection(
    host='sftp.example.com',
    port=22,                    # SSH 포트 (기본: 22)
    user='username',            # 사용자명
    password='secret',          # 비밀번호 (키 인증 시 생략 가능)
    key_file='~/.ssh/id_rsa',   # SSH 개인키 경로 (선택)
    timeout=30                  # 연결 타임아웃 (초)
) as sftp:
    ...
```

### SFTP 병렬 다운로드

```python
from egghouse.transfer import sftp_download_parallel

tasks = [
    ('/data/file1.fits', '/local/file1.fits'),
    ('/data/file2.fits', '/local/file2.fits'),
]

result = sftp_download_parallel(
    host='sftp.example.com',
    download_tasks=tasks,
    user='admin',
    key_file='~/.ssh/id_rsa',
    parallel=4,
    max_retries=3
)

print(f"성공: {result['downloaded']}")
print(f"실패: {result['failed']}")
```

### SFTP 병렬 업로드

```python
from egghouse.transfer import sftp_upload_parallel

tasks = [
    ('/local/file1.fits', '/upload/file1.fits'),
    ('/local/file2.fits', '/upload/file2.fits'),
]

result = sftp_upload_parallel(
    host='sftp.example.com',
    upload_tasks=tasks,
    user='admin',
    key_file='~/.ssh/id_rsa',
    parallel=4
)

print(f"성공: {result['uploaded']}")
print(f"실패: {result['failed']}")
```

---

## API 요약

### HTTP

| 함수 | 설명 |
|------|------|
| `download_single_file(url, dest, ...)` | 단일 파일 다운로드 (재시도 포함) |
| `get_file_list(url, extensions, ...)` | 디렉토리 목록에서 파일 링크 스크래핑 |
| `download_parallel(tasks, ...)` | 병렬 다운로드 |

### FTP

| 함수 | 설명 |
|------|------|
| `ftp_connection(host, ...)` | FTP 연결 context manager |
| `ftp_download_file(ftp, remote, local, ...)` | 단일 파일 다운로드 |
| `ftp_upload_file(ftp, local, remote, ...)` | 단일 파일 업로드 |
| `ftp_list_files(ftp, dir, ...)` | 디렉토리 파일 목록 |
| `ftp_download_parallel(host, tasks, ...)` | 병렬 다운로드 |
| `ftp_upload_parallel(host, tasks, ...)` | 병렬 업로드 |

### SFTP

| 함수 | 설명 |
|------|------|
| `sftp_connection(host, ...)` | SFTP 연결 context manager |
| `sftp_download_file(sftp, remote, local, ...)` | 단일 파일 다운로드 |
| `sftp_upload_file(sftp, local, remote, ...)` | 단일 파일 업로드 |
| `sftp_list_files(sftp, dir, ...)` | 디렉토리 파일 목록 |
| `sftp_download_parallel(host, tasks, ...)` | 병렬 다운로드 |
| `sftp_upload_parallel(host, tasks, ...)` | 병렬 업로드 |

---

## 의존성

| 패키지 | 용도 | 필수 |
|--------|------|------|
| requests | HTTP 요청 | HTTP 사용 시 |
| beautifulsoup4 | HTML 파싱 | HTTP 사용 시 |
| paramiko | SFTP 전송 | SFTP 사용 시 |

설치:
```bash
# HTTP만 사용
pip install requests beautifulsoup4

# SFTP 추가
pip install paramiko

# 전체 설치
pip install egghouse[transfer,sftp]
```

---

## 주의사항

1. **서버 부하**: 병렬 다운로드 수(`parallel`)를 너무 높게 설정하면 서버에 부하를 줄 수 있습니다.
2. **SSL 검증**: `verify_ssl=False`는 신뢰할 수 있는 내부 서버에서만 사용하세요.
3. **타임아웃**: 대용량 파일은 `timeout` 값을 늘려야 할 수 있습니다.
4. **디스크 공간**: 다운로드 전 충분한 디스크 공간이 있는지 확인하세요.
5. **FTP 패시브 모드**: 방화벽 뒤에 있으면 `passive=True` (기본값)를 사용하세요.
6. **SSH 키 보안**: SFTP 키 파일은 적절한 권한(600)으로 보호하세요.
