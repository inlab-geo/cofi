from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import BaseObjective, Model

from petsc4py import PETSc


class TAOSolver(BaseSolver):
    def __init__(self, objective: BaseObjective):
        self.obj = objective

    def solve(self, ) -> Model:


