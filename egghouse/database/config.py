"""
Configuration management for PostgreSQL database.

Supports multiple configuration methods:
1. YAML/JSON files
2. Environment variables
3. Direct dictionary
"""

import json
import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load database configuration from file or environment.

    Priority (highest to lowest):
    1. Environment variables (DB_*)
    2. Config file (YAML/JSON)
    3. Default values

    Args:
        config_path: Path to config file (YAML or JSON).

    Returns:
        Configuration dictionary with 'database' key.

    Example:
        >>> config = load_config('config.yaml')
        >>> db = PostgresManager(**config['database'])

        >>> # Using environment variables:
        >>> # DB_HOST=myserver DB_PORT=5432 DB_NAME=mydb DB_USER=user DB_PASSWORD=pass
        >>> config = load_config()
        >>> db = PostgresManager(**config['database'])
    """
    # Load base config from file
    file_config: Dict[str, Any] = {}
    if config_path:
        file_config = _load_file(config_path)

    # Get database section or use root
    db_config = file_config.get('database', file_config)

    # Load from environment variables
    env_config = _load_env()

    # Merge: env overrides file
    merged = _merge(db_config, env_config)

    # Apply defaults
    defaults = {
        'host': 'localhost',
        'port': 5432,
        'log_queries': False
    }
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value

    return {'database': merged}


def _load_file(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is not supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Determine format by extension
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.yaml', '.yml'):
        return yaml.safe_load(content) or {}
    elif ext == '.json':
        return json.loads(content)
    else:
        # Try YAML first, then JSON
        try:
            return yaml.safe_load(content) or {}
        except yaml.YAMLError:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(f"Unsupported config format: {ext}")


def _load_env() -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Environment variables are matched by DB_ prefix and converted to
    lowercase keys. For example:
    - DB_HOST -> {'host': value}
    - DB_PORT -> {'port': value}

    Returns:
        Configuration dictionary from environment variables.
    """
    config: Dict[str, Any] = {}
    env_prefix = "DB_"

    type_hints = {
        'port': int,
        'log_queries': bool
    }

    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(env_prefix):].lower()

            # Handle special case for 'name' vs 'database'
            if config_key == 'name':
                config_key = 'database'

            # Convert type if hint provided
            config[config_key] = _convert_type(config_key, value, type_hints)

    return config


def _convert_type(key: str, value: str, type_hints: Dict[str, type]) -> Any:
    """
    Convert string value to appropriate type based on type hints.

    Args:
        key: Configuration key.
        value: String value from environment or file.
        type_hints: Dict mapping keys to types.

    Returns:
        Converted value.
    """
    target_type = type_hints.get(key)

    if target_type is None:
        return value

    if target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif target_type == list:
        return [v.strip() for v in value.split(',')]
    else:
        return value


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge override dictionary into base dictionary.

    Args:
        base: Base configuration dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result and
            isinstance(result[key], dict) and
            isinstance(value, dict)
        ):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value

    return result


def from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration from dictionary.

    Args:
        config_dict: Configuration dictionary.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If required keys are missing.

    Example:
        >>> config = from_dict({
        ...     'host': 'localhost',
        ...     'database': 'mydb',
        ...     'user': 'user',
        ...     'password': 'pass'
        ... })
        >>> db = PostgresManager(**config)
    """
    required_keys = ['host', 'database', 'user', 'password']
    missing_keys = [key for key in required_keys if key not in config_dict]

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    return config_dict


def create_example_config(output_path: str = 'config.example.yaml') -> None:
    """
    Create an example configuration file.

    Args:
        output_path: Path to save example config.
    """
    example_config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'your_database_name',
            'user': 'your_username',
            'password': 'your_password'
        },
        'logging': {
            'log_queries': True,
            'log_level': 'INFO',
            'log_file': None  # None for console only, or provide filepath
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    print(f"Example config created: {output_path}")
