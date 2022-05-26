from .base_solver import BaseSolver

from .scipy_opt_min import ScipyOptMinSolver
from .scipy_opt_lstsq import ScipyOptLstSqSolver
from .scipy_lstsq import ScipyLstSqSolver
from .emcee import EmceeSolver


__all__ = [
    "BaseSolver",  # public API, for advanced usage (own solver)
    "ScipyOptMinSolver",
    "ScipyOptLstSqSolver",
    "ScipyLstSqSolver",
    "EmceeSolver",
]


# solvers table grouped by method: {inv_options.method -> {inv_options.tool -> BaseSolver}}
solvers_table = {
    "optimisation": {
        "scipy.optimize.minimize": ScipyOptMinSolver,
        "scipy.optimize.least_squares": ScipyOptLstSqSolver,
    },
    "linear least square": {
        "scipy.linalg.lstsq": ScipyLstSqSolver,
    },
    "sampling": {"emcee": EmceeSolver},
}

# solvers suggest table grouped by method: {inv_options.method -> inv_options.tool}
# NOTE: the default backend solver is from this table, set the first one manually when necessary
# e.g. {'optimisation': ['scipy.optimize.minimize'], 'linear least square': ['scipy.linalg.lstsq']}
solver_suggest_table = {k: list(val.keys()) for k, val in solvers_table.items()}

# solvers dispatch table grouped by tool: {inv_options.tool -> BaseSolver}
# e.g. {'scipy.optimize.minimize':
#           <class 'cofi.solvers.scipy_opt_min.ScipyOptMinSolver'>,
#       'scipy.linalg.lstsq':
#           <class 'cofi.solvers.scipy_lstsq.ScipyLstSqSolver'>}
solver_dispatch_table = {
    k: val for values in solvers_table.values() for k, val in values.items()
}

# all solving methods: {inv_options.method}
# e.g. {'optimisation', 'linear least square'}
solver_methods = set(solvers_table.keys())
