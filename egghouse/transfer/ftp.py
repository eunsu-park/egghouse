"""
FTP file transfer utilities.

Provides functions for connecting to FTP servers, uploading/downloading files,
and listing remote directories. Uses Python's standard library ftplib.
"""

import time
import ftplib
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Generator


@contextmanager
def ftp_connection(
    host: str,
    port: int = 21,
    user: str = 'anonymous',
    password: str = '',
    timeout: int = 30,
    passive: bool = True
) -> Generator[ftplib.FTP, None, None]:
    """
    Context manager for FTP connection.

    Args:
        host: FTP server hostname.
        port: FTP port. Defaults to 21.
        user: Username. Defaults to 'anonymous'.
        password: Password. Empty for anonymous.
        timeout: Connection timeout in seconds. Defaults to 30.
        passive: Use passive mode. Defaults to True.

    Yields:
        ftplib.FTP connection object.

    Example:
        >>> with ftp_connection('ftp.example.com') as ftp:
        ...     files = ftp_list_files(ftp, '/data/')
        ...     for f in files:
        ...         print(f)
    """
    ftp = ftplib.FTP()
    try:
        ftp.connect(host, port, timeout=timeout)
        ftp.login(user, password)
        ftp.set_pasv(passive)
        yield ftp
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()


def ftp_download_file(
    ftp: ftplib.FTP,
    remote_path: str,
    local_path: str,
    overwrite: bool = False
) -> bool:
    """
    Download a single file via FTP.

    Args:
        ftp: Active FTP connection.
        remote_path: Remote file path on server.
        local_path: Local destination path.
        overwrite: If True, overwrite existing file. Defaults to False.

    Returns:
        True if download succeeded, False otherwise.
    """
    if Path(local_path).exists() and not overwrite:
        return True

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f'RETR {remote_path}', f.write)
        return True
    except ftplib.all_errors as e:
        print(f"Failed to download {remote_path}: {e}")
        # Clean up partial download
        if Path(local_path).exists():
            Path(local_path).unlink()
        return False


def ftp_upload_file(
    ftp: ftplib.FTP,
    local_path: str,
    remote_path: str,
    overwrite: bool = False
) -> bool:
    """
    Upload a single file via FTP.

    Args:
        ftp: Active FTP connection.
        local_path: Local file path.
        remote_path: Remote destination path.
        overwrite: If True, overwrite existing file. Defaults to False.

    Returns:
        True if upload succeeded, False otherwise.
    """
    if not Path(local_path).exists():
        print(f"Local file not found: {local_path}")
        return False

    # Check if remote file exists
    if not overwrite:
        try:
            ftp.size(remote_path)
            return True  # File exists, skip
        except ftplib.error_perm:
            pass  # File doesn't exist, proceed

    try:
        with open(local_path, 'rb') as f:
            ftp.storbinary(f'STOR {remote_path}', f)
        return True
    except ftplib.all_errors as e:
        print(f"Failed to upload {local_path}: {e}")
        return False


def ftp_list_files(
    ftp: ftplib.FTP,
    remote_dir: str = '.',
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    List files in a remote FTP directory.

    Args:
        ftp: Active FTP connection.
        remote_dir: Remote directory path. Defaults to current directory.
        extensions: Filter by extensions (e.g., ['fits', 'csv']).
            If None, returns all files.

    Returns:
        List of filenames in the directory.
    """
    try:
        files = ftp.nlst(remote_dir)

        # Extract just the filename from full paths
        files = [Path(f).name for f in files]

        # Filter by extension if specified
        if extensions:
            extensions_lower = [ext.lower().lstrip('.') for ext in extensions]
            files = [
                f for f in files
                if any(f.lower().endswith(f'.{ext}') for ext in extensions_lower)
            ]

        return files

    except ftplib.all_errors as e:
        print(f"Failed to list directory {remote_dir}: {e}")
        return []


def ftp_download_parallel(
    host: str,
    download_tasks: List[Tuple[str, str]],
    port: int = 21,
    user: str = 'anonymous',
    password: str = '',
    overwrite: bool = False,
    max_retries: int = 3,
    parallel: int = 1,
    timeout: int = 30,
    passive: bool = True
) -> Dict[str, int]:
    """
    Download multiple files via FTP with optional parallelization.

    Each parallel worker creates its own FTP connection.

    Args:
        host: FTP server hostname.
        download_tasks: List of (remote_path, local_path) tuples.
        port: FTP port. Defaults to 21.
        user: Username. Defaults to 'anonymous'.
        password: Password.
        overwrite: If True, overwrite existing files. Defaults to False.
        max_retries: Maximum retry attempts per file. Defaults to 3.
        parallel: Number of parallel connections. Defaults to 1.
        timeout: Connection timeout in seconds. Defaults to 30.
        passive: Use passive mode. Defaults to True.

    Returns:
        Dictionary with 'downloaded' and 'failed' counts.

    Example:
        >>> tasks = [
        ...     ('/data/file1.fits', '/local/file1.fits'),
        ...     ('/data/file2.fits', '/local/file2.fits'),
        ... ]
        >>> result = ftp_download_parallel('ftp.example.com', tasks, parallel=4)
        >>> print(f"Downloaded: {result['downloaded']}, Failed: {result['failed']}")
    """
    if not download_tasks:
        return {"downloaded": 0, "failed": 0}

    def _download_with_retry(task: Tuple[str, str]) -> bool:
        remote_path, local_path = task

        for attempt in range(max_retries + 1):
            try:
                with ftp_connection(host, port, user, password, timeout, passive) as ftp:
                    if ftp_download_file(ftp, remote_path, local_path, overwrite):
                        return True
            except Exception as e:
                if attempt == max_retries:
                    print(f"Failed to download {remote_path} after {max_retries + 1} attempts: {e}")
                    return False
                time.sleep(2 ** attempt)  # Exponential backoff

        return False

    successful = 0
    failed = 0

    if parallel < 2:
        # Sequential processing
        for task in download_tasks:
            if _download_with_retry(task):
                successful += 1
            else:
                failed += 1
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            results = executor.map(_download_with_retry, download_tasks)
            for result in results:
                if result:
                    successful += 1
                else:
                    failed += 1

    return {"downloaded": successful, "failed": failed}


def ftp_upload_parallel(
    host: str,
    upload_tasks: List[Tuple[str, str]],
    port: int = 21,
    user: str = 'anonymous',
    password: str = '',
    overwrite: bool = False,
    max_retries: int = 3,
    parallel: int = 1,
    timeout: int = 30,
    passive: bool = True
) -> Dict[str, int]:
    """
    Upload multiple files via FTP with optional parallelization.

    Each parallel worker creates its own FTP connection.

    Args:
        host: FTP server hostname.
        upload_tasks: List of (local_path, remote_path) tuples.
        port: FTP port. Defaults to 21.
        user: Username. Defaults to 'anonymous'.
        password: Password.
        overwrite: If True, overwrite existing files. Defaults to False.
        max_retries: Maximum retry attempts per file. Defaults to 3.
        parallel: Number of parallel connections. Defaults to 1.
        timeout: Connection timeout in seconds. Defaults to 30.
        passive: Use passive mode. Defaults to True.

    Returns:
        Dictionary with 'uploaded' and 'failed' counts.
    """
    if not upload_tasks:
        return {"uploaded": 0, "failed": 0}

    def _upload_with_retry(task: Tuple[str, str]) -> bool:
        local_path, remote_path = task

        for attempt in range(max_retries + 1):
            try:
                with ftp_connection(host, port, user, password, timeout, passive) as ftp:
                    if ftp_upload_file(ftp, local_path, remote_path, overwrite):
                        return True
            except Exception as e:
                if attempt == max_retries:
                    print(f"Failed to upload {local_path} after {max_retries + 1} attempts: {e}")
                    return False
                time.sleep(2 ** attempt)

        return False

    successful = 0
    failed = 0

    if parallel < 2:
        for task in upload_tasks:
            if _upload_with_retry(task):
                successful += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            results = executor.map(_upload_with_retry, upload_tasks)
            for result in results:
                if result:
                    successful += 1
                else:
                    failed += 1

    return {"uploaded": successful, "failed": failed}
