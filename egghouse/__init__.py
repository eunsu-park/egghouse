"""
egghouse - A personal arsenal of reusable code for solar physics research

This package provides utilities for solar physics data analysis and processing.
"""

__version__ = "0.1.0"
__author__ = "Eunsu Park"

# Import main modules for easier access
from . import database

__all__ = ["database", "sdo", "image"]
