#!/usr/bin/env python
"""
Configuration Management Example
================================

Demonstrates how to use BaseConfig for ML/DL project configuration.

Requires: pip install "egghouse[config]"

Run:
    python examples/03_config_usage.py
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

try:
    from egghouse.config import BaseConfig
except ImportError:
    print("Error: pyyaml not installed.")
    print("Install with: pip install 'egghouse[config]'")
    exit(1)


# Define your configuration as a dataclass
@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration for a neural network."""

    # Model parameters
    model_name: str = "resnet50"
    num_classes: int = 10
    pretrained: bool = True

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

    # Data parameters
    data_path: str = "./data"
    num_workers: int = 4

    # Optional parameters
    resume_from: Optional[str] = None


def main():
    print("=" * 60)
    print("egghouse - Configuration Management Example")
    print("=" * 60)

    # 1. Create config with defaults
    print("\n1. Default Configuration")
    print("-" * 40)
    config = TrainingConfig()
    print(f"Model:         {config.model_name}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size:    {config.batch_size}")
    print(f"Epochs:        {config.epochs}")

    # 2. Create config with custom values
    print("\n2. Custom Configuration")
    print("-" * 40)
    custom = TrainingConfig(
        learning_rate=0.0001,
        batch_size=64,
        epochs=200,
        model_name="vit_base"
    )
    print(f"Model:         {custom.model_name}")
    print(f"Learning rate: {custom.learning_rate}")
    print(f"Batch size:    {custom.batch_size}")
    print(f"Epochs:        {custom.epochs}")

    # 3. Save and load from YAML
    print("\n3. Save and Load from YAML")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "config.yaml")

        # Save
        custom.to_yaml(yaml_path)
        print(f"Saved to: {yaml_path}")

        # Show file content
        with open(yaml_path) as f:
            content = f.read()
        print("\nYAML content:")
        print(content)

        # Load
        loaded = TrainingConfig.from_yaml(yaml_path)
        print(f"Loaded learning_rate: {loaded.learning_rate}")
        print(f"Loaded batch_size: {loaded.batch_size}")

    # 4. Load from environment variables
    print("\n4. Load from Environment Variables")
    print("-" * 40)

    # Set some env vars
    os.environ["LEARNING_RATE"] = "0.005"
    os.environ["EPOCHS"] = "50"

    env_config = TrainingConfig.from_env()
    print(f"From env LEARNING_RATE: {env_config.learning_rate}")
    print(f"From env EPOCHS: {env_config.epochs}")

    # Clean up
    del os.environ["LEARNING_RATE"]
    del os.environ["EPOCHS"]

    # 5. Load from CLI arguments
    print("\n5. Load from CLI Arguments")
    print("-" * 40)

    # Simulate CLI arguments
    cli_args = ["--learning_rate", "0.01", "--batch_size", "128", "--epochs", "25"]
    print(f"Simulated CLI: {' '.join(cli_args)}")

    cli_config = TrainingConfig.from_args(cli_args)
    print(f"From CLI learning_rate: {cli_config.learning_rate}")
    print(f"From CLI batch_size: {cli_config.batch_size}")
    print(f"From CLI epochs: {cli_config.epochs}")

    # 6. Typical usage pattern
    print("\n6. Typical Usage Pattern")
    print("-" * 40)
    print("""
# train.py
from dataclasses import dataclass
from egghouse.config import BaseConfig

@dataclass
class Config(BaseConfig):
    lr: float = 0.001
    epochs: int = 100
    model: str = "resnet50"

# Load from YAML base, override with CLI
config = Config.from_args()

# Or with explicit YAML file:
# python train.py --config config.yaml --lr 0.01

print(f"Training with lr={config.lr}, epochs={config.epochs}")
""")

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
