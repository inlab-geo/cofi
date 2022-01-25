__doc__ = """
cofi.cofi_solvers
-----------------

This package contains inversion solvers that interface with existing tools
or are depended on the implementation done by InLab.
"""

from .base_solver import BaseSolver

# from .rjmcmc._solver import ReversibleJumpMCMC
# from . import linear_regression #, scipy
# from .scipy import *
# from . import petsc
# from .petsc._tao import TAOSolver

# from .linear_regression import *
# from .petsc import *
# from .scipy import *

from .linear_regression._python_solver import LRNormalEquation
from .linear_regression._c_solver import LRNormalEquationC
from .linear_regression._cpp_solver import LRNormalEquationCpp
from .linear_regression._f77_solver import LRNormalEquationF77
from .linear_regression._f90_solver import LRNormalEquationF90
from .petsc._tao import TAOSolver
from .scipy._scipy_optimizer import ScipyOptimizerLSSolver, ScipyOptimizerSolver


__all__ = [
    "BaseSolver",
    # "ReversibleJumpMCMC",
    "LRNormalEquation",
    "LRNormalEquationC",
    "LRNormalEquationCpp",
    "LRNormalEquationF77",
    "LRNormalEquationF90",
    "TAOSolver",
    "ScipyOptimizerSolver",
    "ScipyOptimizerLSSolver",
]
