from typing import Type

from . import BaseProblem, InversionOptions
from .solvers import solver_dispatch_table, BaseSolver


class InversionResult:
    def __init__(self, ) -> None:
        pass

    def summary(self) -> None:
        # TODO - directly print to console
        raise NotImplementedError

    def __repr__(self) -> str:
        # TODO
        raise NotImplementedError


class InversionRunner:
    def __init__(self, inv_problem: BaseProblem, inv_options: InversionOptions) -> None:
        self.inv_problem = inv_problem
        self.inv_options = inv_options
        # dispatch inversion_solver from self.inv_options, validation is done by solver
        self.inv_solver = self._dispatch_solver()(inv_problem, inv_options)

    def run(self) -> InversionResult:
        res_dict = self.inv_solver.solve()
        return InversionResult(res_dict)

    def _dispatch_solver(self) -> Type[BaseSolver]:
        # TODO - look up solver_dispatch_table to return constructor for a BaseSolver subclass
        raise NotImplementedError
