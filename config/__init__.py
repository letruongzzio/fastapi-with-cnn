"""
Configuration package initialization.
"""

from . import catdog_cfg
from . import logging_cfg

# Access the imported modules to avoid linting errors
_ = catdog_cfg
_ = logging_cfg
