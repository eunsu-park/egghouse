"""HTTP file transfer utilities.

Provides functions for downloading files from web servers with retry logic
and parallel download capabilities.

Error handling:

- **Transient errors** (connection failures, DNS errors, timeouts, 5xx HTTP
  responses, 408 Request Timeout) are retried with exponential backoff
  (`2 ** attempt` seconds, capped at 60s).
- **Terminal errors** (404 Not Found and other 4xx responses) are NOT
  retried. `get_file_list` treats 404 as "no data here" and returns an
  empty list. `download_single_file` returns False immediately.
- After exhausting retries on a transient error, `get_file_list` re-raises
  the underlying `requests.RequestException` so the caller can distinguish
  "couldn't reach the server" from "no data here". `download_single_file`
  keeps returning False after exhausting retries.
- `download_single_file` streams the body to a temporary `.part` file and
  verifies the written byte count against the ``Content-Length`` header
  when the server provides one; a short/truncated transfer that raised no
  exception is treated as a transient error and retried.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning


# Maximum backoff sleep between retries, in seconds.
_MAX_BACKOFF = 60.0


def _is_transient_error(exc: BaseException) -> bool:
    """Returns True when the exception is a candidate for retry.

    Connection failures, DNS errors, timeouts, chunked-encoding errors,
    5xx HTTP responses, and 408 Request Timeout are treated as transient.
    Other 4xx HTTP responses are treated as terminal.
    """
    if isinstance(exc, (requests.ConnectionError, requests.Timeout,
                        requests.exceptions.ChunkedEncodingError)):
        return True
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is None:
            return True
        code = response.status_code
        if code == 408:
            return True
        if 500 <= code < 600:
            return True
        return False
    if isinstance(exc, requests.RequestException):
        # Unknown subclass — retry by default to avoid silently dropping.
        return True
    return False


def _backoff_sleep(attempt: int, base: float = 2.0) -> None:
    """Sleeps for `base ** attempt` seconds, capped at `_MAX_BACKOFF`."""
    delay = min(base ** attempt, _MAX_BACKOFF)
    time.sleep(delay)


def download_single_file(
    source_url: str,
    destination: str,
    overwrite: bool = False,
    max_retries: int = 3,
    timeout: int = 30,
    verify_ssl: bool = True,
) -> bool:
    """
    Download a single file with retry logic.

    The body is streamed to a temporary ``<destination>.part`` file in
    1 MiB chunks (constant memory regardless of file size). When the
    server sends a ``Content-Length`` header, the written byte count is
    verified against it; a mismatch (a truncated transfer that raised no
    exception) is treated as a transient error and retried. On success
    the temp file is atomically ``os.replace``d onto the destination, so
    a crash mid-transfer never leaves a truncated file at the final path.

    Retries on transient errors (see module docstring) with exponential
    backoff. Terminal errors (e.g., 404 Not Found) return False immediately
    without retrying. After exhausting retries on transient errors, returns
    False.

    Args:
        source_url: URL to download from.
        destination: Local path to save the file.
        overwrite: If True, overwrite existing files. Defaults to False.
        max_retries: Maximum number of retry attempts on transient errors.
            Defaults to 3 (i.e., up to 4 attempts total).
        timeout: Request timeout in seconds. Defaults to 30.
        verify_ssl: If True, verify SSL certificates. Defaults to True.
            Set to False only for trusted internal servers.

    Returns:
        True if download succeeded, False otherwise.
    """
    dest_path = Path(destination)
    if dest_path.exists() and not overwrite:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if not verify_ssl:
        urllib3.disable_warnings(InsecureRequestWarning)

    # Atomic-write: stream to <destination>.part, then os.replace to final.
    # Guarantees the final path is created only when the full body has been
    # written, so a SIGKILL or power loss mid-stream cannot leave a truncated
    # file at the destination. POSIX os.replace is atomic within the same
    # filesystem; cross-fs callers should ensure the tmp and final paths
    # share a mount.
    tmp_path = dest_path.with_name(dest_path.name + ".part")

    def _cleanup_tmp() -> None:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass  # Best-effort; leftover .part is non-fatal.

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        incomplete = False
        try:
            response = requests.get(
                source_url, timeout=timeout, verify=verify_ssl, stream=True
            )
            response.raise_for_status()

            expected = response.headers.get("Content-Length")
            expected_size = int(expected) if expected else None

            written = 0
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)

            if expected_size is not None and written != expected_size:
                # Truncated transfer with no exception raised: retry it.
                _cleanup_tmp()
                incomplete = True
            else:
                os.replace(tmp_path, dest_path)
                return True

        except requests.RequestException as e:
            last_exc = e
            _cleanup_tmp()
            if not _is_transient_error(e):
                print(
                    f"Failed to download {source_url}: {e} "
                    "(terminal error, no retry)"
                )
                return False

        if attempt < max_retries:
            _backoff_sleep(attempt)
        elif incomplete:
            print(
                f"Failed to download {source_url} after {max_retries + 1} "
                f"attempts (incomplete: {written}/{expected_size} bytes)"
            )
            _cleanup_tmp()
            return False
        else:
            print(
                f"Failed to download {source_url} after {max_retries + 1} "
                f"attempts (transient error): {last_exc}"
            )
            _cleanup_tmp()
            return False

    _cleanup_tmp()
    return False


def get_file_list(
    base_url: str,
    extensions: List[str],
    timeout: int = 30,
    verify_ssl: bool = True,
    max_retries: int = 3,
) -> List[str]:
    """
    Get list of files from a web directory listing.

    Parses the HTML directory listing and extracts file links matching
    the specified extensions.

    Retries on transient errors (see module docstring). A 404 response
    is treated as "no listing available here" and returns an empty list
    immediately. After exhausting retries on a transient error, the
    underlying `requests.RequestException` is re-raised so the caller can
    distinguish "no data here" from "couldn't reach the server".

    Args:
        base_url: URL of the directory to list.
        extensions: List of file extensions to filter (e.g., ['fits', 'csv']).
        timeout: Request timeout in seconds. Defaults to 30.
        verify_ssl: If True, verify SSL certificates. Defaults to True.
        max_retries: Maximum number of retry attempts on transient errors.
            Defaults to 3 (i.e., up to 4 attempts total).

    Returns:
        List of filenames matching the extensions. Empty list when the
        URL returns 404 OR the listing has no matching files.

    Raises:
        requests.RequestException: When all retries on a transient error
            are exhausted. The caller can distinguish this from the
            "no data" empty-list return.
    """
    # Lazy import: only directory listing needs an HTML parser, so a
    # caller that only downloads files (download_single_file /
    # download_parallel) does not require beautifulsoup4 to be installed.
    from bs4 import BeautifulSoup

    if not verify_ssl:
        urllib3.disable_warnings(InsecureRequestWarning)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                f"{base_url}/", timeout=timeout, verify=verify_ssl
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            files: List[str] = []
            extensions_lower = [ext.lower() for ext in extensions]

            for link in soup.find_all("a", href=True):
                href = link.get("href")
                if (
                    href
                    and any(
                        href.lower().endswith(f".{ext}") for ext in extensions_lower
                    )
                    and not href.startswith("/")
                    and "?" not in href
                ):
                    files.append(href)

            skip_keywords = ["parent", "..", "index", "readme"]
            return [
                f for f in files
                if not any(skip in f.lower() for skip in skip_keywords)
            ]

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # 404: no data here. Terminal, return empty list.
                return []
            last_exc = e
            if not _is_transient_error(e):
                print(f"Error fetching file list from {base_url}: {e}")
                return []
        except requests.RequestException as e:
            last_exc = e
            # Connection / timeout / etc. are transient by definition.

        if attempt < max_retries:
            _backoff_sleep(attempt)
        else:
            print(
                f"Error fetching file list from {base_url} after "
                f"{max_retries + 1} attempts: {last_exc}"
            )
            assert last_exc is not None
            raise last_exc

    # Unreachable; satisfies the type checker.
    return []


def download_parallel(
    download_tasks: List[Tuple[str, str]],
    overwrite: bool = False,
    max_retries: int = 3,
    parallel: int = 1,
    timeout: int = 30,
    verify_ssl: bool = True,
) -> Dict[str, int]:
    """
    Download multiple files with optional parallelization.

    Each task is processed by `download_single_file`, which retries on
    transient errors and returns False on terminal errors or after
    exhausting retries.

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
        for task in download_tasks:
            if _download_task(task):
                successful += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            results = executor.map(_download_task, download_tasks)
            for result in results:
                if result:
                    successful += 1
                else:
                    failed += 1

    return {"downloaded": successful, "failed": failed}


def download_text(
    url: str,
    *,
    timeout: int = 30,
    max_retries: int = 3,
    verify_ssl: bool = True,
) -> Optional[str]:
    """Fetch a URL's response body as text with retry logic.

    Same transient/terminal classification and exponential backoff as
    ``download_single_file`` (see module docstring). Terminal errors
    (e.g., 404) return None immediately; after exhausting retries on
    transient errors, returns None.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on transient errors.
            Defaults to 3 (up to 4 attempts total).
        verify_ssl: If True, verify SSL certificates. Defaults to True.

    Returns:
        Response text on success, None otherwise.
    """
    if not verify_ssl:
        urllib3.disable_warnings(InsecureRequestWarning)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            last_exc = e
            if not _is_transient_error(e):
                print(f"Failed to fetch {url}: {e} (terminal error, no retry)")
                return None

        if attempt < max_retries:
            _backoff_sleep(attempt)
        else:
            print(
                f"Failed to fetch {url} after {max_retries + 1} attempts "
                f"(transient error): {last_exc}"
            )
            return None

    return None


def download_json(
    url: str,
    *,
    timeout: int = 30,
    max_retries: int = 3,
    verify_ssl: bool = True,
):
    """Fetch a URL's response body and parse it as JSON.

    Same retry policy as ``download_text``. A JSON decode failure is
    treated as terminal (the body, not transport, is the problem) and
    returns None without retrying.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on transient errors.
        verify_ssl: If True, verify SSL certificates.

    Returns:
        Parsed JSON value on success, None otherwise.
    """
    if not verify_ssl:
        urllib3.disable_warnings(InsecureRequestWarning)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError as e:
                print(f"Failed to parse JSON from {url}: {e} (no retry)")
                return None
        except requests.RequestException as e:
            last_exc = e
            if not _is_transient_error(e):
                print(f"Failed to fetch {url}: {e} (terminal error, no retry)")
                return None

        if attempt < max_retries:
            _backoff_sleep(attempt)
        else:
            print(
                f"Failed to fetch {url} after {max_retries + 1} attempts "
                f"(transient error): {last_exc}"
            )
            return None

    return None
