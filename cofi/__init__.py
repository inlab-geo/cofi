__doc__ = """
cofi
-----------------

Common Framework for Inference
"""

from .base_solver import BaseSolver, OptimiserMixin

from . import cofi_objective
from . import utils

from . import linear_reg
from . import optimisers
from . import samplers


__all__ = [
    "BaseSolver",
    "OptimiserMixin",
]
