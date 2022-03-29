from cofi.solvers.base_solver import BaseSolver
import pytest


class NullSolver(BaseSolver):
    def __init__(self):
        super().__init__()


with pytest.raises(NotImplementedError):
    s = NullSolver()
    s.solve()
