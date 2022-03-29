__doc__ = """
cofi.linear_reg
---------------

This package includes linear regression solvers (using the normal equation)
that are implemented with pure Python/C/C++/Fortran77/Fortran90. It's here
mainly for testing purpose.
"""

from ._python_solver import LRNormalEquation
from ._c_solver import LRNormalEquationC
from ._cpp_solver import LRNormalEquationCpp
from ._f77_solver import LRNormalEquationF77
from ._f90_solver import LRNormalEquationF90

__all__ = [
    "LRNormalEquation",
    "LRNormalEquationC",
    "LRNormalEquationCpp",
    "LRNormalEquationF77",
    "LRNormalEquationF90",
]
