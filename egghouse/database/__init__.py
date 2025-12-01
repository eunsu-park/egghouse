"""
Database utilities for egghouse.

This package provides simple database management tools for research purposes.
"""

from .postgres import PostgresManager, to_dataframe
from .config import load_config, from_dict, create_example_config

__all__ = ['PostgresManager', 'to_dataframe', 'load_config', 'from_dict', 'create_example_config']
__version__ = '0.1.0'
