from .base_solver import BaseSolver

from .scipy_opt import ScipyOptMinSolver


__all__ = [
    "BaseSolver",           # public API, for advanced usage (own solver)
    "ScipyOptMinSolver",
]


# dispatch solver: {inv_options.method + inv_options.tool -> BaseSolver}
solver_dispatch_table = {
    ("optimisation", "scipy.optimize.minimize") : ScipyOptMinSolver
}
