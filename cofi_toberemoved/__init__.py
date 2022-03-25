"""The cofi package contains base classes for inversion solvers, forward operators and
   objective classes.
"""

__all__ = []

from .model_params import Model
from .base_forward import BaseForward, LinearForward, PolynomialForward
from .base_objective import BaseObjective, LeastSquareObjective, LinearObjective
from .base_solver import BaseSolver, OptimiserMixin

from . import cofi_objective
from . import utils

from . import linear_reg
from . import optimisers
from . import samplers

try:
    from . import _version

    __version__ = _version.__version__
except ImportError:
    pass

__all__ = [
    "BaseSolver",
    "OptimiserMixin",
    "Model",
    "BaseObjective",
    "LeastSquareObjective",
    "LinearObjective",
    "BaseForward",
    "LinearForward",
    "PolynomialForward",
]
