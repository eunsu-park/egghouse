"""
BMP (Bitmap) file I/O utilities.

No external dependencies required - uses only numpy and struct.

BMP Format Overview:
    BMP is a raster image format developed by Microsoft. A BMP file consists of:

    1. File Header (14 bytes):
        Offset  Size  Field
        0       2     Signature: 'BM' (0x42, 0x4D)
        2       4     File size in bytes
        6       2     Reserved1 (unused)
        8       2     Reserved2 (unused)
        10      4     Pixel data offset from start of file

    2. DIB Header (BITMAPINFOHEADER, 40 bytes):
        Offset  Size  Field
        14      4     Header size (40 for BITMAPINFOHEADER)
        18      4     Image width in pixels (signed)
        22      4     Image height in pixels (signed, positive=bottom-up)
        26      2     Color planes (must be 1)
        28      2     Bits per pixel (1, 4, 8, 16, 24, or 32)
        30      4     Compression method (0=BI_RGB uncompressed)
        34      4     Image data size (may be 0 for BI_RGB)
        38      4     Horizontal resolution (pixels/meter)
        42      4     Vertical resolution (pixels/meter)
        46      4     Number of colors in palette (0=default)
        50      4     Number of important colors (0=all)

    3. Color Table (optional):
        Present for bpp <= 8. Each entry is 4 bytes: B, G, R, 0x00.
        Number of entries: 2^bpp (or as specified in header).

    4. Pixel Data:
        - Stored bottom-up by default (first row in file = bottom of image).
          Negative height indicates top-down order.
        - Each row is padded to a multiple of 4 bytes.
        - For 24-bit: pixels stored as B, G, R (3 bytes each).
        - For 8-bit: palette indices (1 byte each).

    This module supports reading 8-bit (grayscale/palette) and 24-bit
    (RGB) uncompressed BMPs, and writing 24-bit RGB or 8-bit grayscale BMPs.

Example:
    >>> from egghouse.io import read_bmp, write_bmp
    >>> data, info = read_bmp('image.bmp')
    >>> print(data.shape, data.dtype)
    (512, 512, 3) uint8
    >>> write_bmp('output.bmp', data)
"""

import os
import struct
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np


# BMP signature
_BMP_SIGNATURE = b'BM'

# BITMAPINFOHEADER size
_DIB_HEADER_SIZE = 40

# Compression constants
_BI_RGB = 0


def _parse_file_header(f) -> Dict[str, Any]:
    """Parse BMP file header (14 bytes)."""
    raw = f.read(14)
    if len(raw) < 14:
        raise ValueError("File too small for BMP file header")

    signature = raw[0:2]
    if signature != _BMP_SIGNATURE:
        raise ValueError(
            f"Invalid BMP signature: {signature!r} (expected b'BM')"
        )

    file_size, _, _, data_offset = struct.unpack('<I HH I', raw[2:14])
    return {
        'file_size': file_size,
        'data_offset': data_offset,
    }


def _parse_dib_header(f) -> Dict[str, Any]:
    """Parse BITMAPINFOHEADER (40 bytes)."""
    raw = f.read(4)
    header_size = struct.unpack('<I', raw)[0]

    if header_size < _DIB_HEADER_SIZE:
        raise ValueError(
            f"Unsupported DIB header size: {header_size} "
            f"(expected >= {_DIB_HEADER_SIZE})"
        )

    raw = f.read(header_size - 4)
    fields = struct.unpack('<i i HH I I i i I I', raw[:36])

    info = {
        'header_size': header_size,
        'width': fields[0],
        'height': fields[1],
        'planes': fields[2],
        'bpp': fields[3],
        'compression': fields[4],
        'image_size': fields[5],
        'x_ppm': fields[6],
        'y_ppm': fields[7],
        'colors_used': fields[8],
        'colors_important': fields[9],
    }

    # Skip remaining header bytes if header is larger than 40
    if header_size > _DIB_HEADER_SIZE:
        f.read(header_size - _DIB_HEADER_SIZE)

    return info


def read_bmp_header(filepath: str) -> Dict[str, Any]:
    """
    Read BMP file and DIB headers without loading pixel data.

    Args:
        filepath: Path to BMP file.

    Returns:
        Dictionary containing:
            - file_size: Total file size in bytes
            - data_offset: Offset to pixel data
            - width: Image width in pixels
            - height: Image height (absolute value)
            - bpp: Bits per pixel
            - compression: Compression method (0=uncompressed)
            - top_down: True if image is stored top-down

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a valid BMP.

    Example:
        >>> info = read_bmp_header('image.bmp')
        >>> print(f"{info['width']}x{info['height']}, {info['bpp']}bpp")
        512x512, 24bpp
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BMP file not found: {filepath}")

    with open(filepath, 'rb') as f:
        file_header = _parse_file_header(f)
        dib_header = _parse_dib_header(f)

    top_down = dib_header['height'] < 0
    result = {**file_header, **dib_header}
    result['height'] = abs(result['height'])
    result['top_down'] = top_down
    return result


def read_bmp(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read pixel data and header from a BMP file.

    Supports uncompressed 8-bit (grayscale/palette) and 24-bit (RGB) BMPs.
    Returns pixel data as RGB uint8 array regardless of input format.

    Args:
        filepath: Path to BMP file.

    Returns:
        Tuple of (data, header_info) where data is a uint8 numpy array
        with shape (H, W, 3) for RGB, and header_info is a dictionary
        of BMP header fields.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a valid BMP or uses unsupported format.

    Example:
        >>> data, info = read_bmp('image.bmp')
        >>> print(data.shape, data.dtype)
        (512, 512, 3) uint8
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BMP file not found: {filepath}")

    with open(filepath, 'rb') as f:
        file_header = _parse_file_header(f)
        dib_header = _parse_dib_header(f)

        if dib_header['compression'] != _BI_RGB:
            raise ValueError(
                f"Unsupported compression: {dib_header['compression']} "
                f"(only uncompressed BI_RGB=0 is supported)"
            )

        bpp = dib_header['bpp']
        width = dib_header['width']
        height_raw = dib_header['height']
        top_down = height_raw < 0
        height = abs(height_raw)

        # Read color table for 8-bit images
        palette = None
        if bpp == 8:
            n_colors = dib_header['colors_used'] or 256
            palette_data = f.read(n_colors * 4)
            palette = np.frombuffer(palette_data, dtype=np.uint8).reshape(-1, 4)

        # Seek to pixel data
        f.seek(file_header['data_offset'])

        if bpp == 24:
            row_size = (width * 3 + 3) & ~3  # Padded to 4 bytes
            raw = f.read(row_size * height)
            pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_size)
            # Extract BGR, convert to RGB
            bgr = pixels[:, :width * 3].reshape(height, width, 3)
            data = bgr[:, :, ::-1].copy()  # BGR -> RGB

        elif bpp == 8:
            row_size = (width + 3) & ~3  # Padded to 4 bytes
            raw = f.read(row_size * height)
            indices = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_size)
            indices = indices[:, :width]

            if palette is not None:
                # Map palette indices to RGB
                bgr = palette[indices, :3]  # B, G, R columns
                data = bgr[:, :, ::-1].copy()  # BGR -> RGB
            else:
                # Grayscale fallback
                data = np.stack([indices] * 3, axis=-1)

        else:
            raise ValueError(
                f"Unsupported bits per pixel: {bpp} "
                f"(only 8 and 24 are supported)"
            )

    # BMP default is bottom-up; flip if not top-down
    if not top_down:
        data = np.flipud(data)

    header_info = {**file_header, **dib_header}
    header_info['height'] = height
    header_info['top_down'] = top_down
    return data, header_info


def write_bmp(
    filepath: str,
    data: np.ndarray,
    overwrite: bool = False
) -> None:
    """
    Write a numpy array to a BMP file.

    Accepts grayscale (H, W) or RGB (H, W, 3) uint8 arrays.
    Grayscale images are written as 8-bit BMP with a grayscale palette.
    RGB images are written as 24-bit BMP.

    Args:
        filepath: Output file path.
        data: Image data as uint8 numpy array.
            Shape (H, W) for grayscale or (H, W, 3) for RGB.
        overwrite: If True, overwrite existing file. Defaults to False.

    Raises:
        FileExistsError: If file exists and overwrite is False.
        ValueError: If data shape or dtype is invalid.

    Example:
        >>> import numpy as np
        >>> rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        >>> rgb[:, :, 0] = 255  # Red image
        >>> write_bmp('red.bmp', rgb)
        >>>
        >>> gray = np.zeros((256, 256), dtype=np.uint8)
        >>> write_bmp('black.bmp', gray)
    """
    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(f"File already exists: {filepath}")

    if data.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 data, got {data.dtype}. "
            f"Convert with data.astype(np.uint8)"
        )

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        _write_bmp_8bit(filepath, data)
    elif data.ndim == 3 and data.shape[2] == 3:
        _write_bmp_24bit(filepath, data)
    else:
        raise ValueError(
            f"Expected shape (H, W) or (H, W, 3), got {data.shape}"
        )


def _write_bmp_24bit(filepath: str, data: np.ndarray) -> None:
    """Write 24-bit RGB BMP."""
    height, width = data.shape[:2]
    row_size = (width * 3 + 3) & ~3
    padding = row_size - width * 3
    image_size = row_size * height
    file_size = 14 + _DIB_HEADER_SIZE + image_size

    # RGB -> BGR and flip vertically (bottom-up)
    bgr = data[::-1, :, ::-1]

    with open(filepath, 'wb') as f:
        # File header
        f.write(_BMP_SIGNATURE)
        f.write(struct.pack('<I', file_size))
        f.write(struct.pack('<HH', 0, 0))
        f.write(struct.pack('<I', 14 + _DIB_HEADER_SIZE))

        # DIB header
        f.write(struct.pack('<I', _DIB_HEADER_SIZE))
        f.write(struct.pack('<i i', width, height))
        f.write(struct.pack('<HH', 1, 24))
        f.write(struct.pack('<I I', _BI_RGB, image_size))
        f.write(struct.pack('<i i', 0, 0))
        f.write(struct.pack('<I I', 0, 0))

        # Pixel data with row padding
        pad_bytes = b'\x00' * padding
        for row in range(height):
            f.write(bgr[row].tobytes())
            if padding > 0:
                f.write(pad_bytes)


def _write_bmp_8bit(filepath: str, data: np.ndarray) -> None:
    """Write 8-bit grayscale BMP with grayscale palette."""
    height, width = data.shape
    row_size = (width + 3) & ~3
    padding = row_size - width
    palette_size = 256 * 4
    image_size = row_size * height
    data_offset = 14 + _DIB_HEADER_SIZE + palette_size
    file_size = data_offset + image_size

    # Flip vertically (bottom-up)
    flipped = data[::-1]

    with open(filepath, 'wb') as f:
        # File header
        f.write(_BMP_SIGNATURE)
        f.write(struct.pack('<I', file_size))
        f.write(struct.pack('<HH', 0, 0))
        f.write(struct.pack('<I', data_offset))

        # DIB header
        f.write(struct.pack('<I', _DIB_HEADER_SIZE))
        f.write(struct.pack('<i i', width, height))
        f.write(struct.pack('<HH', 1, 8))
        f.write(struct.pack('<I I', _BI_RGB, image_size))
        f.write(struct.pack('<i i', 0, 0))
        f.write(struct.pack('<I I', 256, 0))

        # Grayscale palette (B, G, R, 0x00)
        for i in range(256):
            f.write(struct.pack('BBBB', i, i, i, 0))

        # Pixel data with row padding
        pad_bytes = b'\x00' * padding
        for row in range(height):
            f.write(flipped[row].tobytes())
            if padding > 0:
                f.write(pad_bytes)
