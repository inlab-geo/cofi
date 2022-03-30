from .base_solver import BaseSolver

from .scipy_opt import ScipyOptMinSolver
from .numpy_lstsq import NumpyLstSqSolver


__all__ = [
    "BaseSolver",           # public API, for advanced usage (own solver)
    "ScipyOptMinSolver",
    "NumpyLstSqSolver",
]


# solvers table grouped by method: {inv_options.method -> {inv_options.tool -> BaseSolver}}
solvers_table = {
    "optimisation": {
        "scipy.optimize.minimize": ScipyOptMinSolver
    },
    "least square": {
        "numpy.linalg.lstsq": NumpyLstSqSolver
    }
}
# solvers suggest table grouped by method: {inv_options.method -> inv_options.tool}
solver_suggest_table = {k:list(val.keys()) for k,val in solvers_table.items()}
# solvers dispatch table grouped by tool: {inv_options.tool -> BaseSolver}
solver_dispatch_table = {k:val for values in solvers_table.values() for k,val in values.items()}
# all solving methods: {inv_options.method}
solver_methods = solvers_table.keys()

print(solvers_table)
print(solver_suggest_table)
print(solver_dispatch_table)
print(solver_methods)
