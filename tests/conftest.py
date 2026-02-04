"""Pytest configuration and shared fixtures."""

import os
import tempfile

import numpy as np
import pytest


# =============================================================================
# Image fixtures
# =============================================================================


@pytest.fixture
def sample_image_2d():
    """Create a 2D grayscale test image (64x64, uint8)."""
    np.random.seed(42)
    return np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def sample_image_float():
    """Create a 2D float test image (64x64, float32)."""
    np.random.seed(42)
    return np.random.rand(64, 64).astype(np.float32) * 1000


@pytest.fixture
def sample_image_3d():
    """Create a 3D RGB test image (64x64x3, uint8)."""
    np.random.seed(42)
    return np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Create a larger test image (256x256, float64)."""
    np.random.seed(42)
    return np.random.rand(256, 256).astype(np.float64)


@pytest.fixture
def solar_disk_image():
    """Create a simulated solar disk image for testing."""
    size = 128
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 3

    disk = np.sqrt((x - center) ** 2 + (y - center) ** 2) < radius
    image = np.zeros((size, size), dtype=np.float32)
    image[disk] = 1000 + 200 * np.sin(x[disk] / 10) * np.cos(y[disk] / 10)
    return image


# =============================================================================
# File fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create a sample YAML config file."""
    config_content = """lr: 0.01
epochs: 50
name: experiment1
debug: true
"""
    path = os.path.join(temp_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(config_content)
    return path


@pytest.fixture
def sample_json_config(temp_dir):
    """Create a sample JSON config file."""
    import json

    config = {"lr": 0.02, "epochs": 200, "name": "test", "debug": False}
    path = os.path.join(temp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


# =============================================================================
# Custom markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_astropy: mark test as requiring astropy"
    )
    config.addinivalue_line(
        "markers", "requires_sunpy: mark test as requiring sunpy"
    )
    config.addinivalue_line(
        "markers", "requires_psycopg2: mark test as requiring psycopg2"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
