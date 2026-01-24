"""
HTTP file transfer utilities.

Provides functions for downloading files from web servers with retry logic
and parallel download capabilities.
"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning
import urllib3


def download_single_file(
    source_url: str,
    destination: str,
    overwrite: bool = False,
    max_retries: int = 3,
    timeout: int = 30,
    verify_ssl: bool = True
) -> bool:
    """
    Download a single file with retry logic.

    Args:
        source_url: URL to download from.
        destination: Local path to save the file.
        overwrite: If True, overwrite existing files. Defaults to False.
        max_retries: Maximum number of retry attempts. Defaults to 3.
        timeout: Request timeout in seconds. Defaults to 30.
        verify_ssl: If True, verify SSL certificates. Defaults to True.
            Set to False only for trusted internal servers.

    Returns:
        True if download succeeded, False otherwise.
    """
    if Path(destination).exists() and not overwrite:
        return True

    Path(destination).parent.mkdir(parents=True, exist_ok=True)

    # Suppress SSL warnings only if verify_ssl is False
    if not verify_ssl:
        urllib3.disable_warnings(InsecureRequestWarning)

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(source_url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()

            with open(destination, 'wb') as f:
                f.write(response.content)
            return True

        except requests.RequestException as e:
            if attempt == max_retries:
                print(f"Failed to download {source_url} after {max_retries + 1} attempts: {e}")
                return False
            time.sleep(2 ** attempt)  # Exponential backoff

    return False


def get_file_list(
    base_url: str,
    extensions: List[str],
    timeout: int = 30,
    verify_ssl: bool = True
) -> List[str]:
    """
    Get list of files from a web directory listing.

    Parses the HTML directory listing and extracts file links matching
    the specified extensions.

    Args:
        base_url: URL of the directory to list.
        extensions: List of file extensions to filter (e.g., ['fits', 'csv']).
        timeout: Request timeout in seconds. Defaults to 30.
        verify_ssl: If True, verify SSL certificates. Defaults to True.

    Returns:
        List of filenames matching the extensions.
    """
    # Suppress SSL warnings only if verify_ssl is False
    if not verify_ssl:
        urllib3.disable_warnings(InsecureRequestWarning)

    try:
        response = requests.get(f"{base_url}/", timeout=timeout, verify=verify_ssl)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        files: List[str] = []

        # Normalize extensions to lowercase
        extensions_lower = [ext.lower() for ext in extensions]

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if (href and
                any(href.lower().endswith(f".{ext}") for ext in extensions_lower) and
                not href.startswith('/') and '?' not in href):
                files.append(href)

        # Filter out navigation links
        skip_keywords = ['parent', '..', 'index', 'readme']
        return [f for f in files if not any(skip in f.lower() for skip in skip_keywords)]

    except requests.RequestException as e:
        print(f"Error fetching file list from {base_url}: {e}")
        return []


def download_parallel(
    download_tasks: List[Tuple[str, str]],
    overwrite: bool = False,
    max_retries: int = 3,
    parallel: int = 1,
    timeout: int = 30,
    verify_ssl: bool = True
) -> Dict[str, int]:
    """
    Download multiple files with optional parallelization.

    Args:
        download_tasks: List of (source_url, destination_path) tuples.
        overwrite: If True, overwrite existing files. Defaults to False.
        max_retries: Maximum retry attempts per file. Defaults to 3.
        parallel: Number of parallel downloads. Defaults to 1 (sequential).
        timeout: Request timeout in seconds. Defaults to 30.
        verify_ssl: If True, verify SSL certificates. Defaults to True.

    Returns:
        Dictionary with 'downloaded' and 'failed' counts.

    Example:
        >>> tasks = [
        ...     ('https://example.com/file1.fits', '/data/file1.fits'),
        ...     ('https://example.com/file2.fits', '/data/file2.fits'),
        ... ]
        >>> result = download_parallel(tasks, parallel=4)
        >>> print(f"Downloaded: {result['downloaded']}, Failed: {result['failed']}")
    """
    if not download_tasks:
        return {"downloaded": 0, "failed": 0}

    successful = 0
    failed = 0

    def _download_task(task: Tuple[str, str]) -> bool:
        source, dest = task
        return download_single_file(
            source, dest, overwrite, max_retries, timeout, verify_ssl
        )

    if parallel < 2:
        # Sequential processing
        for task in download_tasks:
            if _download_task(task):
                successful += 1
            else:
                failed += 1
    else:
        # Parallel processing using ThreadPoolExecutor (I/O-bound task)
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            results = executor.map(_download_task, download_tasks)

            for result in results:
                if result:
                    successful += 1
                else:
                    failed += 1

    return {"downloaded": successful, "failed": failed}
