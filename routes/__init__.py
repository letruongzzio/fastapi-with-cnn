"""
This module is used to import all the routes in the application.
"""

from . import catdog_route
from . import base

__all__ = ['catdog_route', 'base']

_ = catdog_route
_ = base
