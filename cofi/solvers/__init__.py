from .base_solver import BaseSolver

from .scipy_opt_min import ScipyOptMinSolver
from .scipy_opt_lstsq import ScipyOptLstSqSolver
from .numpy_lstsq import NumpyLstSqSolver


__all__ = [
    "BaseSolver",           # public API, for advanced usage (own solver)
    "ScipyOptMinSolver",
    "NumpyLstSqSolver",
]


# solvers table grouped by method: {inv_options.method -> {inv_options.tool -> BaseSolver}}
solvers_table = {
    "optimisation": {
        "scipy.optimize.minimize": ScipyOptMinSolver,
        "scipy.optimize.least_squares": ScipyOptLstSqSolver,
    },
    "least square": {
        "numpy.linalg.lstsq": NumpyLstSqSolver,
    }
}

# solvers suggest table grouped by method: {inv_options.method -> inv_options.tool}
# e.g. {'optimisation': ['scipy.optimize.minimize'], 'least square': ['numpy.linalg.lstsq']}
solver_suggest_table = {k:list(val.keys()) for k,val in solvers_table.items()}

# solvers dispatch table grouped by tool: {inv_options.tool -> BaseSolver}
# e.g. {'scipy.optimize.minimize': <class 'cofi.solvers.scipy_opt_min.ScipyOptMinSolver'>, 'numpy.linalg.lstsq': <class 'cofi.solvers.numpy_lstsq.NumpyLstSqSolver'>}
solver_dispatch_table = {k:val for values in solvers_table.values() for k,val in values.items()}

# all solving methods: {inv_options.method}
# e.g. {'optimisation', 'least square'}
solver_methods = set(solvers_table.keys())
