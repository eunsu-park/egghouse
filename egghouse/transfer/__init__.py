"""
File transfer utilities.

Supported protocols:
    - HTTP/HTTPS: Web downloads with retry and parallel support (requires requests, beautifulsoup4)
    - FTP: Standard FTP with ftplib (no external dependencies)
    - SFTP: Secure FTP via SSH (requires paramiko)

Example:
    >>> from egghouse.transfer import ftp_connection, ftp_download_file
    >>> # FTP download
    >>> with ftp_connection('ftp.example.com') as ftp:
    ...     ftp_download_file(ftp, '/data/file.fits', 'file.fits')
"""

# HTTP (optional - requires requests, beautifulsoup4)
try:
    from .http import download_single_file
    from .http import get_file_list
    from .http import download_parallel
    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False
    download_single_file = None
    get_file_list = None
    download_parallel = None

# FTP (no external dependencies)
from .ftp import (
    ftp_connection,
    ftp_download_file,
    ftp_upload_file,
    ftp_list_files,
    ftp_download_parallel,
    ftp_upload_parallel,
)

# SFTP (optional - requires paramiko)
from .sftp import (
    sftp_connection,
    sftp_download_file,
    sftp_upload_file,
    sftp_list_files,
    sftp_download_parallel,
    sftp_upload_parallel,
    HAS_PARAMIKO,
)

__all__ = [
    # HTTP
    'download_single_file',
    'get_file_list',
    'download_parallel',
    'HAS_HTTP',
    # FTP
    'ftp_connection',
    'ftp_download_file',
    'ftp_upload_file',
    'ftp_list_files',
    'ftp_download_parallel',
    'ftp_upload_parallel',
    # SFTP
    'sftp_connection',
    'sftp_download_file',
    'sftp_upload_file',
    'sftp_list_files',
    'sftp_download_parallel',
    'sftp_upload_parallel',
    'HAS_PARAMIKO',
]
