__doc__ = """
cofi.cofi_solvers
-----------------

This package contains inversion solvers that interface with existing tools
or are depended on the implementation done by InLab.
"""

from .base_solver import BaseSolver

# from .rjmcmc._solver import ReversibleJumpMCMC
from .linear_regression._c_solver import LRNormalEquationC
from .linear_regression._cpp_solver import LRNormalEquationCpp
from .linear_regression._python_solver import LRNormalEquation
from .scipy._scipy_optimizer import ScipyOptimizerSolver, ScipyOptimizerLSSolver
from .petsc._tao import TAOSolver


__all__ = [
    "BaseSolver",
    # "ReversibleJumpMCMC",
    "LRNormalEquationC",
    "LRNormalEquationCpp",
    "LRNormalEquation",
    "TAOSolver",
    "ScipyOptimizerSolver",
    "ScipyOptimizerLSSolver",
]
