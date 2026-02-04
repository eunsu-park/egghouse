"""Tests for egghouse.config.base module."""

import json
import os
from dataclasses import dataclass
from typing import Optional

import pytest

# Import with fallback for when pyyaml is not installed
try:
    from egghouse.config import BaseConfig
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


@pytest.fixture
def skip_without_config():
    """Skip test if config module is not available."""
    if not HAS_CONFIG:
        pytest.skip("pyyaml not installed")


@dataclass
class SampleConfig(BaseConfig if HAS_CONFIG else object):
    """Sample config for testing."""

    lr: float = 0.001
    epochs: int = 100
    name: str = "test"
    debug: bool = False
    dropout: Optional[float] = None


@pytest.mark.skipif(not HAS_CONFIG, reason="pyyaml not installed")
class TestBaseConfigDefaults:
    """Tests for default value handling."""

    def test_default_values(self):
        """Test default value initialization."""
        config = SampleConfig()
        assert config.lr == 0.001
        assert config.epochs == 100
        assert config.name == "test"
        assert config.debug is False
        assert config.dropout is None

    def test_override_defaults(self):
        """Test overriding default values."""
        config = SampleConfig(lr=0.01, epochs=50)
        assert config.lr == 0.01
        assert config.epochs == 50
        assert config.name == "test"  # Unchanged


@pytest.mark.skipif(not HAS_CONFIG, reason="pyyaml not installed")
class TestFromYaml:
    """Tests for from_yaml method."""

    def test_from_yaml_basic(self, temp_dir):
        """Test loading from YAML file."""
        yaml_content = """
lr: 0.01
epochs: 50
name: experiment1
debug: true
"""
        path = os.path.join(temp_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(yaml_content)

        config = SampleConfig.from_yaml(path)
        assert config.lr == 0.01
        assert config.epochs == 50
        assert config.name == "experiment1"
        assert config.debug is True

    def test_from_yaml_partial(self, temp_dir):
        """Test loading partial config (uses defaults for missing)."""
        yaml_content = """
lr: 0.02
"""
        path = os.path.join(temp_dir, "partial.yaml")
        with open(path, "w") as f:
            f.write(yaml_content)

        config = SampleConfig.from_yaml(path)
        assert config.lr == 0.02
        assert config.epochs == 100  # Default
        assert config.name == "test"  # Default

    def test_from_yaml_optional_field(self, temp_dir):
        """Test loading with optional field."""
        yaml_content = """
lr: 0.01
dropout: 0.5
"""
        path = os.path.join(temp_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(yaml_content)

        config = SampleConfig.from_yaml(path)
        assert config.dropout == 0.5

    def test_from_yaml_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            SampleConfig.from_yaml("/nonexistent/path/config.yaml")


@pytest.mark.skipif(not HAS_CONFIG, reason="pyyaml not installed")
class TestFromJson:
    """Tests for from_json method."""

    def test_from_json_basic(self, temp_dir):
        """Test loading from JSON file."""
        data = {"lr": 0.02, "epochs": 200, "debug": True}
        path = os.path.join(temp_dir, "config.json")
        with open(path, "w") as f:
            json.dump(data, f)

        config = SampleConfig.from_json(path)
        assert config.lr == 0.02
        assert config.epochs == 200
        assert config.debug is True

    def test_from_json_partial(self, temp_dir):
        """Test loading partial config from JSON."""
        data = {"epochs": 300}
        path = os.path.join(temp_dir, "partial.json")
        with open(path, "w") as f:
            json.dump(data, f)

        config = SampleConfig.from_json(path)
        assert config.epochs == 300
        assert config.lr == 0.001  # Default


@pytest.mark.skipif(not HAS_CONFIG, reason="pyyaml not installed")
class TestToYaml:
    """Tests for to_yaml method."""

    def test_to_yaml_basic(self, temp_dir):
        """Test saving to YAML file."""
        config = SampleConfig(lr=0.05, epochs=300)
        path = os.path.join(temp_dir, "output.yaml")
        config.to_yaml(path)

        # Read back
        loaded = SampleConfig.from_yaml(path)
        assert loaded.lr == 0.05
        assert loaded.epochs == 300

    def test_to_yaml_roundtrip(self, temp_dir):
        """Test save and load roundtrip."""
        original = SampleConfig(
            lr=0.001, epochs=100, name="roundtrip", debug=True, dropout=0.3
        )
        path = os.path.join(temp_dir, "roundtrip.yaml")
        original.to_yaml(path)

        loaded = SampleConfig.from_yaml(path)
        assert loaded.lr == original.lr
        assert loaded.epochs == original.epochs
        assert loaded.name == original.name
        assert loaded.debug == original.debug
        assert loaded.dropout == original.dropout


@pytest.mark.skipif(not HAS_CONFIG, reason="pyyaml not installed")
class TestFromEnv:
    """Tests for from_env method."""

    def test_from_env_basic(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("LR", "0.1")
        monkeypatch.setenv("EPOCHS", "500")

        config = SampleConfig.from_env()
        assert config.lr == 0.1
        assert config.epochs == 500

    def test_from_env_with_prefix(self, monkeypatch):
        """Test loading with prefix."""
        monkeypatch.setenv("TRAIN_LR", "0.05")
        monkeypatch.setenv("TRAIN_EPOCHS", "250")

        config = SampleConfig.from_env(prefix="TRAIN_")
        assert config.lr == 0.05
        assert config.epochs == 250

    def test_from_env_boolean(self, monkeypatch):
        """Test boolean conversion from env."""
        monkeypatch.setenv("DEBUG", "true")
        config = SampleConfig.from_env()
        assert config.debug is True

        monkeypatch.setenv("DEBUG", "false")
        config = SampleConfig.from_env()
        assert config.debug is False

    def test_from_env_uses_defaults(self, monkeypatch):
        """Test that missing env vars use defaults."""
        # Don't set any env vars
        config = SampleConfig.from_env()
        assert config.lr == 0.001  # Default
        assert config.epochs == 100  # Default


@pytest.mark.skipif(not HAS_CONFIG, reason="pyyaml not installed")
class TestFromArgs:
    """Tests for from_args method."""

    def test_from_args_basic(self):
        """Test loading from CLI arguments."""
        args = ["--lr", "0.005", "--epochs", "150"]
        config = SampleConfig.from_args(args)
        assert config.lr == 0.005
        assert config.epochs == 150

    def test_from_args_boolean(self):
        """Test boolean arguments."""
        args = ["--debug"]
        config = SampleConfig.from_args(args)
        assert config.debug is True

    def test_from_args_partial(self):
        """Test partial CLI arguments."""
        args = ["--epochs", "50"]
        config = SampleConfig.from_args(args)
        assert config.epochs == 50
        assert config.lr == 0.001  # Default

    def test_from_args_string_field(self):
        """Test string field from CLI."""
        args = ["--name", "my_experiment"]
        config = SampleConfig.from_args(args)
        assert config.name == "my_experiment"
