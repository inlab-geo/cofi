__doc__='''
cofi.cofi_solvers
-----------------

This package contains inversion solvers that interface with existing tools
or are depended on the implementation done by InLab.
'''

from .base_solver import BaseSolver
# from .rjmcmc._solver import ReversibleJumpMCMC
from .linear_regression._solver import LinearRegression
from .petsc import *

__all__ = [
    "BaseSolver", 
    # "ReversibleJumpMCMC", 
    "LinearRegression", 
    "TAOSolver"
]
