"""
Configuration management for ML/DL projects.

Provides BaseConfig class with YAML, JSON, CLI, and environment variable support.

Example:
    >>> from dataclasses import dataclass
    >>> from egghouse.config import BaseConfig
    >>>
    >>> @dataclass
    >>> class TrainConfig(BaseConfig):
    ...     lr: float = 0.001
    ...     epochs: int = 100
    >>>
    >>> # Load from YAML
    >>> config = TrainConfig.from_yaml('config.yaml')
    >>>
    >>> # Load from JSON
    >>> config = TrainConfig.from_json('config.json')
    >>>
    >>> # Load from environment variables
    >>> config = TrainConfig.from_env(prefix="TRAIN_")
    >>>
    >>> # Load from CLI
    >>> config = TrainConfig.from_args()
"""

from .base import BaseConfig

__all__ = ['BaseConfig']
