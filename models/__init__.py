"""
This module is used to import the model and predictor classes for the catdog model.
"""

from . import catdog_model
from . import catdog_predictor

__all__ = ['catdog_model', 'catdog_predictor']

_ = catdog_model
_ = catdog_predictor
