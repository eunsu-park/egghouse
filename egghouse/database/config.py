"""
Configuration management for PostgreSQL database.

Supports multiple configuration methods:
1. YAML/JSON files
2. Environment variables
3. Direct dictionary
"""

import os
import json
from typing import Dict, Any, Optional


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load database configuration from file or environment.
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file (YAML/JSON)
    3. Default values
    
    Args:
        config_path: Path to config file (YAML or JSON)
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_config('config.yaml')
        >>> db = PostgresManager(**config['database'])
    """
    config = {}
    
    # 1. Load from file if provided
    if config_path:
        config = _load_from_file(config_path)
    
    # 2. Override with environment variables
    env_config = _load_from_env()
    if env_config:
        if 'database' not in config:
            config['database'] = {}
        config['database'].update(env_config)
    
    # 3. Add defaults
    config = _add_defaults(config)
    
    return config


def _load_from_file(file_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install it with: pip install pyyaml"
                )
        elif file_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path}")


def _load_from_env() -> Dict[str, Any]:
    """Load database configuration from environment variables."""
    env_config = {}
    
    # Database connection
    if os.getenv('DB_HOST'):
        env_config['host'] = os.getenv('DB_HOST')
    if os.getenv('DB_PORT'):
        env_config['port'] = int(os.getenv('DB_PORT'))
    if os.getenv('DB_NAME') or os.getenv('DB_DATABASE'):
        env_config['database'] = os.getenv('DB_NAME') or os.getenv('DB_DATABASE')
    if os.getenv('DB_USER'):
        env_config['user'] = os.getenv('DB_USER')
    if os.getenv('DB_PASSWORD'):
        env_config['password'] = os.getenv('DB_PASSWORD')
    
    # Logging
    if os.getenv('DB_LOG_QUERIES'):
        env_config['log_queries'] = os.getenv('DB_LOG_QUERIES').lower() == 'true'
    
    return env_config


def _add_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add default values to configuration."""
    if 'database' not in config:
        config['database'] = {}
    
    defaults = {
        'host': 'localhost',
        'port': 5432,
        'log_queries': False
    }
    
    for key, value in defaults.items():
        if key not in config['database']:
            config['database'][key] = value
    
    return config


def from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated configuration
        
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


def create_example_config(output_path: str = 'config.example.yaml'):
    """
    Create an example configuration file.
    
    Args:
        output_path: Path to save example config
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
    
    try:
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
        print(f"Example config created: {output_path}")
    except ImportError:
        # Fallback to JSON if YAML not available
        output_path = output_path.replace('.yaml', '.json').replace('.yml', '.json')
        with open(output_path, 'w') as f:
            json.dump(example_config, f, indent=2)
        print(f"Example config created: {output_path}")