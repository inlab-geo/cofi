from .base_solver import BaseSolver, OptimiserMixin
from .base_forward import BaseForward, LinearFittingFwd, PolynomialFittingFwd
from .base_objective import BaseObjective, LeastSquareObjective, LinearFittingObjective
from .model_params import Model

from . import cofi_objective
from . import utils

from . import linear_reg
from . import optimisers
from . import samplers

try:
    from . import _version
    __version__ = _version.__version__
except:
    pass


__all__ = [
    # Solver
    "BaseSolver",
    "OptimiserMixin",
    # Objective
    "Model",
    "BaseObjective",
    "LeastSquareObjective",
    "LinearFittingObjective",
    "BaseForward",
    "LinearFittingFwd",
    "PolynomialFittingFwd",
]
