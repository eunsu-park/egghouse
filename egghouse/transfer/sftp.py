"""
SFTP (SSH File Transfer Protocol) utilities.

Provides functions for connecting to SFTP servers, uploading/downloading files,
and listing remote directories. Requires paramiko library.
"""

import time
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Generator, Any

try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False


def _require_paramiko() -> None:
    """Raise ImportError if paramiko is not available."""
    if not HAS_PARAMIKO:
        raise ImportError(
            "paramiko is required for SFTP. "
            "Install with: pip install paramiko"
        )


@contextmanager
def sftp_connection(
    host: str,
    port: int = 22,
    user: str = None,
    password: Optional[str] = None,
    key_file: Optional[str] = None,
    timeout: int = 30
) -> Generator[Any, None, None]:
    """
    Context manager for SFTP connection.

    Supports both password and key-based authentication.

    Args:
        host: SFTP server hostname.
        port: SSH port. Defaults to 22.
        user: Username.
        password: Password (if not using key authentication).
        key_file: Path to private key file (e.g., ~/.ssh/id_rsa).
        timeout: Connection timeout in seconds. Defaults to 30.

    Yields:
        paramiko.SFTPClient object.

    Example:
        >>> with sftp_connection('sftp.example.com', user='admin', key_file='~/.ssh/id_rsa') as sftp:
        ...     files = sftp_list_files(sftp, '/data/')
        ...     for f in files:
        ...         print(f)
    """
    _require_paramiko()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Expand ~ in key_file path
        if key_file:
            key_file = str(Path(key_file).expanduser())

        ssh.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            key_filename=key_file,
            timeout=timeout
        )
        sftp = ssh.open_sftp()
        yield sftp
    finally:
        try:
            sftp.close()
        except Exception:
            pass
        ssh.close()


def sftp_download_file(
    sftp: Any,
    remote_path: str,
    local_path: str,
    overwrite: bool = False
) -> bool:
    """
    Download a single file via SFTP.

    Args:
        sftp: Active SFTP connection (paramiko.SFTPClient).
        remote_path: Remote file path.
        local_path: Local destination path.
        overwrite: If True, overwrite existing file. Defaults to False.

    Returns:
        True if download succeeded, False otherwise.
    """
    _require_paramiko()

    if Path(local_path).exists() and not overwrite:
        return True

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        sftp.get(remote_path, local_path)
        return True
    except Exception as e:
        print(f"Failed to download {remote_path}: {e}")
        # Clean up partial download
        if Path(local_path).exists():
            Path(local_path).unlink()
        return False


def sftp_upload_file(
    sftp: Any,
    local_path: str,
    remote_path: str,
    overwrite: bool = False
) -> bool:
    """
    Upload a single file via SFTP.

    Args:
        sftp: Active SFTP connection (paramiko.SFTPClient).
        local_path: Local file path.
        remote_path: Remote destination path.
        overwrite: If True, overwrite existing file. Defaults to False.

    Returns:
        True if upload succeeded, False otherwise.
    """
    _require_paramiko()

    if not Path(local_path).exists():
        print(f"Local file not found: {local_path}")
        return False

    # Check if remote file exists
    if not overwrite:
        try:
            sftp.stat(remote_path)
            return True  # File exists, skip
        except IOError:
            pass  # File doesn't exist, proceed

    try:
        sftp.put(local_path, remote_path)
        return True
    except Exception as e:
        print(f"Failed to upload {local_path}: {e}")
        return False


def sftp_list_files(
    sftp: Any,
    remote_dir: str = '.',
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    List files in a remote SFTP directory.

    Args:
        sftp: Active SFTP connection (paramiko.SFTPClient).
        remote_dir: Remote directory path. Defaults to current directory.
        extensions: Filter by extensions (e.g., ['fits', 'csv']).
            If None, returns all files.

    Returns:
        List of filenames in the directory.
    """
    _require_paramiko()

    try:
        files = sftp.listdir(remote_dir)

        # Filter out directories (only keep files)
        file_list = []
        for f in files:
            try:
                stat = sftp.stat(f'{remote_dir}/{f}')
                # Check if it's a regular file (not a directory)
                if not stat.st_mode & 0o40000:
                    file_list.append(f)
            except Exception:
                file_list.append(f)  # Include if stat fails

        # Filter by extension if specified
        if extensions:
            extensions_lower = [ext.lower().lstrip('.') for ext in extensions]
            file_list = [
                f for f in file_list
                if any(f.lower().endswith(f'.{ext}') for ext in extensions_lower)
            ]

        return file_list

    except Exception as e:
        print(f"Failed to list directory {remote_dir}: {e}")
        return []


def sftp_download_parallel(
    host: str,
    download_tasks: List[Tuple[str, str]],
    port: int = 22,
    user: str = None,
    password: Optional[str] = None,
    key_file: Optional[str] = None,
    overwrite: bool = False,
    max_retries: int = 3,
    parallel: int = 1,
    timeout: int = 30
) -> Dict[str, int]:
    """
    Download multiple files via SFTP with optional parallelization.

    Each parallel worker creates its own SFTP connection.

    Args:
        host: SFTP server hostname.
        download_tasks: List of (remote_path, local_path) tuples.
        port: SSH port. Defaults to 22.
        user: Username.
        password: Password (if not using key authentication).
        key_file: Path to private key file.
        overwrite: If True, overwrite existing files. Defaults to False.
        max_retries: Maximum retry attempts per file. Defaults to 3.
        parallel: Number of parallel connections. Defaults to 1.
        timeout: Connection timeout. Defaults to 30.

    Returns:
        Dictionary with 'downloaded' and 'failed' counts.

    Example:
        >>> tasks = [
        ...     ('/data/file1.fits', '/local/file1.fits'),
        ...     ('/data/file2.fits', '/local/file2.fits'),
        ... ]
        >>> result = sftp_download_parallel(
        ...     'sftp.example.com', tasks,
        ...     user='admin', key_file='~/.ssh/id_rsa', parallel=4
        ... )
        >>> print(f"Downloaded: {result['downloaded']}, Failed: {result['failed']}")
    """
    _require_paramiko()

    if not download_tasks:
        return {"downloaded": 0, "failed": 0}

    def _download_with_retry(task: Tuple[str, str]) -> bool:
        remote_path, local_path = task

        for attempt in range(max_retries + 1):
            try:
                with sftp_connection(host, port, user, password, key_file, timeout) as sftp:
                    if sftp_download_file(sftp, remote_path, local_path, overwrite):
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


def sftp_upload_parallel(
    host: str,
    upload_tasks: List[Tuple[str, str]],
    port: int = 22,
    user: str = None,
    password: Optional[str] = None,
    key_file: Optional[str] = None,
    overwrite: bool = False,
    max_retries: int = 3,
    parallel: int = 1,
    timeout: int = 30
) -> Dict[str, int]:
    """
    Upload multiple files via SFTP with optional parallelization.

    Each parallel worker creates its own SFTP connection.

    Args:
        host: SFTP server hostname.
        upload_tasks: List of (local_path, remote_path) tuples.
        port: SSH port. Defaults to 22.
        user: Username.
        password: Password (if not using key authentication).
        key_file: Path to private key file.
        overwrite: If True, overwrite existing files. Defaults to False.
        max_retries: Maximum retry attempts per file. Defaults to 3.
        parallel: Number of parallel connections. Defaults to 1.
        timeout: Connection timeout. Defaults to 30.

    Returns:
        Dictionary with 'uploaded' and 'failed' counts.
    """
    _require_paramiko()

    if not upload_tasks:
        return {"uploaded": 0, "failed": 0}

    def _upload_with_retry(task: Tuple[str, str]) -> bool:
        local_path, remote_path = task

        for attempt in range(max_retries + 1):
            try:
                with sftp_connection(host, port, user, password, key_file, timeout) as sftp:
                    if sftp_upload_file(sftp, local_path, remote_path, overwrite):
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
