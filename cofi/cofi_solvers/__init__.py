__doc__ = """
cofi.cofi_solvers
-----------------

This package contains inversion solvers that interface with existing tools
or are depended on the implementation done by InLab.
"""

from .base_solver import BaseSolver

# from .rjmcmc._solver import ReversibleJumpMCMC
from .linear_regression._python_solver import SimpleLinearRegression
from .scipy._scipy_optimizer import ScipyOptimizerSolver, ScipyOptimizerLMSolver
from .petsc._tao import TAOSolver

__all__ = [
    "BaseSolver",
    # "ReversibleJumpMCMC",
    "SimpleLinearRegression",
    "TAOSolver",
    "ScipyOptimizerSolver",
    "ScipyOptimizerLMSolver",
]
