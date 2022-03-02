__doc__ = """
cofi.optimisers
---------------

This package contains wrappers over several optimisers, and we utilise existing
optimisers in our inversion context so that a "best" model is calculated.
"""

from ._scipy_optimiser import ScipyOptimiserSolver, ScipyOptimiserLSSolver
from ._petsc_tao import TAOSolver

__all__ = [
    "ScipyOptimiserSolver",
    "ScipyOptimiserLSSolver",
    "TAOSolver",
]
