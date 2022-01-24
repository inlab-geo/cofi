__doc__ = """
cofi.cofi_solvers
-----------------

This package contains inversion solvers that interface with existing tools
or are depended on the implementation done by InLab.
"""

from .base_solver import BaseSolver

# from .rjmcmc._solver import ReversibleJumpMCMC
from . import linear_regression #, scipy
# from .scipy import *
# from . import petsc
# from .petsc._tao import TAOSolver


__all__ = [
    "BaseSolver",
    # "ReversibleJumpMCMC",
    "LRNormalEquation",
    "LRNormalEquationC",
    "LRNormalEquationCpp",
    "LRNormalEquationF77",
    "LRNormalEquationF90",
    # "TAOSolver",
    # "ScipyOptimizerSolver",
    # "ScipyOptimizerLSSolver",
]
