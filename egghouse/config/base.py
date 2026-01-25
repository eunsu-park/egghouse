"""
Base configuration module for ML/DL projects.

This module provides a BaseConfig class that supports:
- YAML file loading/saving
- JSON file loading/saving
- CLI argument parsing with override capability
- Environment variable loading
- Type hints via dataclasses

Example:
    >>> from egghouse.config import BaseConfig
    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    >>> class TrainConfig(BaseConfig):
    ...     lr: float = 0.0002
    ...     epochs: int = 200
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
    >>> # Or from CLI with YAML base
    >>> # python train.py --config config.yaml --lr 0.001
    >>> config = TrainConfig.from_args()
"""

import argparse
import json
import os
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Optional, TypeVar, Type, List, Any, get_origin, get_args

import yaml


T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig:
    """Base configuration class with YAML, JSON, CLI, and environment variable support.

    Subclasses should be decorated with @dataclass and define
    configuration fields with type hints and default values.

    Attributes:
        All attributes are defined by subclasses.

    Example:
        >>> from dataclasses import dataclass
        >>> from egghouse.config import BaseConfig
        >>>
        >>> @dataclass
        >>> class TrainConfig(BaseConfig):
        ...     lr: float = 0.001
        ...     epochs: int = 100
        >>>
        >>> config = TrainConfig()
        >>> config.to_yaml('config.yaml')
    """

    @classmethod
    def from_yaml(cls: Type[T], path: str) -> T:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Configuration instance populated from YAML data.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where the YAML file will be saved.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_json(cls: Type[T], path: str) -> T:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Configuration instance populated from JSON data.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            json.JSONDecodeError: If the JSON file is malformed.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str, indent: int = 2) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path where the JSON file will be saved.
            indent: JSON indentation level (default: 2).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=indent)

    @classmethod
    def from_env(cls: Type[T], prefix: str = "") -> T:
        """Load configuration from environment variables.

        Environment variable names are constructed as: PREFIX + FIELD_NAME (uppercase).
        Type conversion is automatic based on field type hints.

        Args:
            prefix: Environment variable prefix (e.g., "TRAIN_" -> TRAIN_LR, TRAIN_EPOCHS)

        Returns:
            Configuration instance with values from environment variables.
            Fields not found in environment use their default values.

        Example:
            >>> # With environment: TRAIN_LR=0.001 TRAIN_EPOCHS=50
            >>> config = TrainConfig.from_env(prefix="TRAIN_")
            >>> print(config.lr)  # 0.001
        """
        config_dict = {}
        for f in fields(cls):
            env_key = f"{prefix}{f.name}".upper()
            env_value = os.environ.get(env_key)

            if env_value is not None:
                config_dict[f.name] = cls._convert_env_value(f.type, env_value)
            else:
                config_dict[f.name] = f.default

        return cls(**config_dict)

    @staticmethod
    def _convert_env_value(field_type: Any, value: str) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            field_type: The target type for conversion.
            value: String value from environment variable.

        Returns:
            Converted value matching the field type.
        """
        # Handle Optional types
        origin = get_origin(field_type)
        if origin is not None:
            type_args = get_args(field_type)
            if type(None) in type_args:
                field_type = [t for t in type_args if t is not type(None)][0]

        if field_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif field_type == int:
            return int(value)
        elif field_type == float:
            return float(value)
        else:
            return value

    @classmethod
    def from_args(cls: Type[T], args: Optional[List[str]] = None) -> T:
        """Create configuration from CLI arguments with optional YAML/JSON base.

        Supports loading a base configuration from YAML/JSON and overriding
        specific values via command-line arguments.

        Args:
            args: List of CLI arguments. If None, uses sys.argv.

        Returns:
            Configuration instance with CLI overrides applied.

        Example:
            python train.py --config base.yaml --lr 0.001 --epochs 100
        """
        parser = argparse.ArgumentParser(
            description=cls.__doc__ or 'Configuration',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            '--config', type=str, default=None,
            help='Path to YAML/JSON config file (values can be overridden by CLI args)'
        )

        # Add arguments for each dataclass field
        for f in fields(cls):
            arg_name = f'--{f.name}'
            field_type = f.type

            # Handle Optional types
            origin = get_origin(field_type)
            if origin is not None:
                type_args = get_args(field_type)
                if type(None) in type_args:
                    # Optional type - get the non-None type
                    field_type = [t for t in type_args if t is not type(None)][0]

            if field_type == bool:
                # Boolean fields use store_true/store_false
                parser.add_argument(
                    arg_name, action='store_true', default=None,
                    help=f'{f.name} (default: {f.default})'
                )
                parser.add_argument(
                    f'--no-{f.name}', action='store_false', dest=f.name,
                    help=f'Disable {f.name}'
                )
            else:
                parser.add_argument(
                    arg_name, type=field_type, default=None,
                    help=f'{f.name} (default: {f.default})'
                )

        parsed = parser.parse_args(args)

        # Start with defaults or config file
        if parsed.config:
            # Detect format by extension
            ext = os.path.splitext(parsed.config)[1].lower()
            if ext == '.json':
                config = cls.from_json(parsed.config)
            else:
                config = cls.from_yaml(parsed.config)
            config_dict = asdict(config)
        else:
            # Use dataclass defaults
            config_dict = {f.name: f.default for f in fields(cls)}

        # Override with CLI arguments (only if explicitly provided)
        for f in fields(cls):
            cli_value = getattr(parsed, f.name, None)
            if cli_value is not None:
                config_dict[f.name] = cli_value

        return cls(**config_dict)

    def __str__(self) -> str:
        """Return a formatted string representation of the config."""
        lines = [f'{self.__class__.__name__}:']
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f'  {f.name}: {value}')
        return '\n'.join(lines)
