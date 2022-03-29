from typing import Type

from . import BaseProblem, InversionOptions
from .solvers import solver_dispatch_table, BaseSolver


class InversionResult:
    def __init__(self, ) -> None:
        pass

    def summary(self) -> None:
        # TODO - directly print to console
        title = "Summary for inversion result"
        display_width = len(title)
        double_line = "=" * display_width
        single_line = "-" * display_width
        print(title)
        print(double_line)
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
        self.inv_result = InversionResult(res_dict)
        return self.inv_result

    def _dispatch_solver(self) -> Type[BaseSolver]:
        # TODO - look up solver_dispatch_table to return constructor for a BaseSolver subclass
        raise NotImplementedError

    def summary(self):      # TODO to test
        title = "Summary for inversion runner"
        subtitle_result = "Trained with the following result"
        subtitle_options = "With inversion solver defined as below"
        subtitle_problem = "For inversion problem defined as below"
        display_width = max(len(title), len(subtitle_result), len(subtitle_options), len(subtitle_problem))
        double_line = "=" * display_width
        single_line = "-" * display_width
        print(title)
        print(double_line)
        if hasattr(self, "inv_result"):
            print(subtitle_result)
            self.inv_result.summary()
            print(single_line)
        print(subtitle_options)
        self.inv_options.summary()
        print(single_line)
        print(subtitle_problem)
        self.inv_problem.summary()
