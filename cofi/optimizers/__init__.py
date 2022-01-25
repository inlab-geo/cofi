__doc__ = """
cofi.optimizers
---------------

This package contains wrappers over several optimizers, and we utilise existing
optimizers in our inversion context so that a "best" model is calculated.
"""

from ._scipy_optimizer import ScipyOptimizerSolver, ScipyOptimizerLSSolver
from ._petsc_tao import TAOSolver

__all__ = [
    "ScipyOptimizerSolver",
    "ScipyOptimizerLSSolver",
    "TAOSolver",
]
