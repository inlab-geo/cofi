from .base_solver import BaseSolver

from .scipy_opt import ScipyOptMinSolver
from .numpy_lstsq import NumpyLstSqSolver


__all__ = [
    "BaseSolver",           # public API, for advanced usage (own solver)
    "ScipyOptMinSolver",
    "NumpyLstSqSolver",
]


# dispatch solver: {inv_options.method + inv_options.tool -> BaseSolver}
solver_dispatch_table = {
    "optimisation": {
        "scipy.optimize.minimize": ScipyOptMinSolver
    },
    "least square": {
        "numpy.linalg.lstsq": NumpyLstSqSolver
    }
}
