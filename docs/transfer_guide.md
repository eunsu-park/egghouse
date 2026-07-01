# egghouse.transfer Usage Guide

File transfer utility (supports HTTP, FTP, SFTP).

---

## Overview

A utility for downloading/uploading files over various protocols:

**HTTP**
- Single file download (with retry logic)
- Scraping file links from a directory listing
- Parallel download (ThreadPoolExecutor)

**FTP** (no additional dependencies)
- Connect to an FTP server (context manager)
- Download/upload files
- List files in a directory
- Parallel download/upload

**SFTP** (requires paramiko)
- SSH-based secure file transfer
- Password or key file authentication
- Download/upload files
- Parallel download/upload

---

## Installation

```bash
pip install requests beautifulsoup4
```

---

## Single File Download

### download_single_file

```python
from egghouse.transfer import download_single_file

# Basic download
success = download_single_file(
    'https://example.com/data.fits',
    '/local/path/data.fits'
)

if success:
    print("Download complete!")
```

### Parameters

```python
success = download_single_file(
    source_url='https://example.com/file.fits',
    destination='/local/path/file.fits',
    overwrite=False,       # If True, overwrite the existing file
    max_retries=3,         # Maximum number of retries
    timeout=30,            # Request timeout (seconds)
    verify_ssl=True        # Verify SSL certificate
)
```

### Retry Logic

On download failure, retries with exponential backoff:
- 1st attempt fails → wait 1 second → 2nd attempt
- 2nd attempt fails → wait 2 seconds → 3rd attempt
- 3rd attempt fails → wait 4 seconds → 4th attempt (when max_retries=3)

```python
# Configuration for unstable connections
success = download_single_file(
    url, destination,
    max_retries=5,   # More retries
    timeout=60       # Longer timeout
)
```

### Ignoring SSL Certificates

When using internal servers or self-signed certificates:

```python
# Caution: use only with trusted servers
success = download_single_file(
    'https://internal-server/data.fits',
    '/local/data.fits',
    verify_ssl=False
)
```

---

## Getting the File List

### get_file_list

Extracts file links from a web directory listing (Apache/Nginx index).

```python
from egghouse.transfer import get_file_list

# List of FITS files
files = get_file_list(
    'https://data.server.com/2024/01/',
    extensions=['fits']
)
print(files)  # ['aia_171_001.fits', 'aia_171_002.fits', ...]

# Multiple extensions
files = get_file_list(
    'https://data.server.com/archive/',
    extensions=['fits', 'fits.gz', 'csv']
)
```

### Parameters

```python
files = get_file_list(
    base_url='https://example.com/data/',
    extensions=['fits', 'csv'],  # Extensions to filter by
    timeout=30,                  # Request timeout
    verify_ssl=True              # SSL verification
)
```

### Return Value

- Extracts filenames from `<a href>` tags in the directory listing
- Filters to only files ending with the specified extensions
- Excludes navigation links such as 'parent', '..', 'index', 'readme'

---

## Parallel Download

### download_parallel

Downloads multiple files concurrently.

```python
from egghouse.transfer import download_parallel

# List of download tasks (URL, save path)
tasks = [
    ('https://example.com/file1.fits', '/data/file1.fits'),
    ('https://example.com/file2.fits', '/data/file2.fits'),
    ('https://example.com/file3.fits', '/data/file3.fits'),
]

# Parallel download with 4 threads
result = download_parallel(tasks, parallel=4)

print(f"Success: {result['downloaded']}")
print(f"Failed: {result['failed']}")
```

### Parameters

```python
result = download_parallel(
    download_tasks=tasks,      # List of (url, path) tuples
    overwrite=False,           # Overwrite existing files
    max_retries=3,             # Retries per file
    parallel=4,                # Number of concurrent downloads (1=sequential)
    timeout=30,                # Request timeout
    verify_ssl=True            # SSL verification
)
```

### Return Value

```python
{
    'downloaded': 10,  # Number of successful files
    'failed': 2        # Number of failed files
}
```

---

## Full Workflow Examples

### Downloading from a Data Archive

```python
from egghouse.transfer import get_file_list, download_parallel
import os

# 1. Get the file list
base_url = 'https://data.archive.org/solar/2024/01/'
files = get_file_list(base_url, extensions=['fits'])

print(f"Found {len(files)} files")

# 2. Create download tasks
output_dir = '/local/data/2024/01/'
os.makedirs(output_dir, exist_ok=True)

tasks = [
    (f"{base_url}{filename}", f"{output_dir}{filename}")
    for filename in files
]

# 3. Parallel download
result = download_parallel(tasks, parallel=4, max_retries=5)

print(f"Downloaded: {result['downloaded']}")
print(f"Failed: {result['failed']}")
```

### Downloading a Date Range

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

# Download with 8 threads
result = download_parallel(all_tasks, parallel=8)
```

### Retry and Resume

```python
from egghouse.transfer import download_single_file
import os

def download_with_resume(url, path, max_retries=3):
    """Skip files that already exist and download the rest"""
    if os.path.exists(path):
        print(f"Skip (exists): {path}")
        return True

    return download_single_file(url, path, max_retries=max_retries)

# Usage
tasks = [...]
for url, path in tasks:
    download_with_resume(url, path)
```

---

## FTP Download/Upload

Uses the Python standard library `ftplib`, so no additional dependencies are required.

### FTP Connection

```python
from egghouse.transfer import ftp_connection, ftp_list_files, ftp_download_file

# Connect via context manager (automatic disconnection)
with ftp_connection('ftp.example.com', user='anonymous', password='') as ftp:
    # List files
    files = ftp_list_files(ftp, '/data/', extensions=['fits'])
    print(f"Found {len(files)} files")

    # Download a single file
    ftp_download_file(ftp, '/data/file.fits', 'local_file.fits')
```

### FTP Connection Parameters

```python
with ftp_connection(
    host='ftp.example.com',
    port=21,                # FTP port (default: 21)
    user='anonymous',       # Username (default: anonymous)
    password='',            # Password
    timeout=30,             # Connection timeout (seconds)
    passive=True            # Passive mode (default: True)
) as ftp:
    ...
```

### FTP File Upload

```python
from egghouse.transfer import ftp_connection, ftp_upload_file

with ftp_connection('ftp.example.com', user='admin', password='secret') as ftp:
    # Upload a single file
    success = ftp_upload_file(ftp, 'local_file.fits', '/remote/path/file.fits')

    if success:
        print("Upload complete!")
```

### FTP Parallel Download

```python
from egghouse.transfer import ftp_download_parallel

# List of download tasks (remote path, local path)
tasks = [
    ('/data/file1.fits', '/local/file1.fits'),
    ('/data/file2.fits', '/local/file2.fits'),
    ('/data/file3.fits', '/local/file3.fits'),
]

# Parallel download with 4 connections
result = ftp_download_parallel(
    host='ftp.example.com',
    download_tasks=tasks,
    user='anonymous',
    parallel=4,
    max_retries=3
)

print(f"Success: {result['downloaded']}")
print(f"Failed: {result['failed']}")
```

### FTP Parallel Upload

```python
from egghouse.transfer import ftp_upload_parallel

# List of upload tasks (local path, remote path)
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

print(f"Success: {result['uploaded']}")
print(f"Failed: {result['failed']}")
```

---

## SFTP Download/Upload

SSH-based secure file transfer. Requires the `paramiko` library.

### SFTP Installation

```bash
pip install paramiko
```

### Checking the SFTP Dependency

```python
from egghouse.transfer import HAS_PARAMIKO

if HAS_PARAMIKO:
    print("SFTP available")
else:
    print("paramiko installation required: pip install paramiko")
```

### SFTP Connection (Password)

```python
from egghouse.transfer import sftp_connection, sftp_download_file, sftp_list_files

with sftp_connection(
    host='sftp.example.com',
    user='admin',
    password='secret'
) as sftp:
    # List files
    files = sftp_list_files(sftp, '/data/', extensions=['fits'])

    # Download a file
    sftp_download_file(sftp, '/data/file.fits', 'local_file.fits')
```

### SFTP Connection (SSH Key)

```python
from egghouse.transfer import sftp_connection, sftp_upload_file

with sftp_connection(
    host='sftp.example.com',
    port=22,
    user='admin',
    key_file='~/.ssh/id_rsa'  # Path to SSH private key
) as sftp:
    # Upload a file
    sftp_upload_file(sftp, 'local_file.fits', '/remote/path/file.fits')
```

### SFTP Connection Parameters

```python
with sftp_connection(
    host='sftp.example.com',
    port=22,                    # SSH port (default: 22)
    user='username',            # Username
    password='secret',          # Password (can be omitted for key authentication)
    key_file='~/.ssh/id_rsa',   # Path to SSH private key (optional)
    timeout=30                  # Connection timeout (seconds)
) as sftp:
    ...
```

### SFTP Parallel Download

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

print(f"Success: {result['downloaded']}")
print(f"Failed: {result['failed']}")
```

### SFTP Parallel Upload

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

print(f"Success: {result['uploaded']}")
print(f"Failed: {result['failed']}")
```

---

## API Summary

### HTTP

| Function | Description |
|------|------|
| `download_single_file(url, dest, ...)` | Single file download (with retry) |
| `get_file_list(url, extensions, ...)` | Scrape file links from a directory listing |
| `download_parallel(tasks, ...)` | Parallel download |

### FTP

| Function | Description |
|------|------|
| `ftp_connection(host, ...)` | FTP connection context manager |
| `ftp_download_file(ftp, remote, local, ...)` | Single file download |
| `ftp_upload_file(ftp, local, remote, ...)` | Single file upload |
| `ftp_list_files(ftp, dir, ...)` | List files in a directory |
| `ftp_download_parallel(host, tasks, ...)` | Parallel download |
| `ftp_upload_parallel(host, tasks, ...)` | Parallel upload |

### SFTP

| Function | Description |
|------|------|
| `sftp_connection(host, ...)` | SFTP connection context manager |
| `sftp_download_file(sftp, remote, local, ...)` | Single file download |
| `sftp_upload_file(sftp, local, remote, ...)` | Single file upload |
| `sftp_list_files(sftp, dir, ...)` | List files in a directory |
| `sftp_download_parallel(host, tasks, ...)` | Parallel download |
| `sftp_upload_parallel(host, tasks, ...)` | Parallel upload |

---

## Dependencies

| Package | Purpose | Required |
|--------|------|------|
| requests | HTTP requests | When using HTTP |
| beautifulsoup4 | HTML parsing | When using HTTP |
| paramiko | SFTP transfer | When using SFTP |

Installation:
```bash
# HTTP only
pip install requests beautifulsoup4

# Add SFTP
pip install paramiko

# Full installation
pip install egghouse[transfer,sftp]
```

---

## Notes

1. **Server load**: Setting the number of parallel downloads (`parallel`) too high can put load on the server.
2. **SSL verification**: Use `verify_ssl=False` only with trusted internal servers.
3. **Timeout**: Large files may require increasing the `timeout` value.
4. **Disk space**: Verify that there is enough disk space before downloading.
5. **FTP passive mode**: If you are behind a firewall, use `passive=True` (the default).
6. **SSH key security**: Protect SFTP key files with appropriate permissions (600).
