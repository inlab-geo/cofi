"""The cofi package contains base classes for inversion solvers, forward operators and
   objective classes.
"""

__all__ = []

from .model_params import Model
__all__.append("Model")

from .base_forward import BaseForward, LinearForward, PolynomialForward
__all__.append("BaseForward")
__all__.append("LinearForward")
__all__.append("PolynomialForward")

from .base_objective import BaseObjective, LeastSquareObjective, LinearObjective
__all__.append("BaseObjective")
__all__.append("LeastSquareObjective")
__all__.append("LinearObjective")

from .base_solver import BaseSolver, OptimiserMixin
__all__.append("BaseSolver")
__all__.append("OptimiserMixin")


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
