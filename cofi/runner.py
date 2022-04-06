from typing import Type
import json

from . import BaseProblem, InversionOptions
from .solvers import solver_dispatch_table, BaseSolver


class InversionResult:
    def __init__(self, res: dict) -> None:
        self.__dict__.update(res)
        self.res = res
        if not hasattr(self, "success"):
            raise ValueError(
                "inversion termination status not returned in result dictionary, "
                "fix your solver to return properly. Check CoFI documentation "
                "'Advanced Usage' section for how to plug in your own solver"
            )
        self.success_or_not = "success" if hasattr(self, "success") and self.success else "failure"

    def summary(self) -> None:
        self._summary()

    def _summary(self, display_lines=True) -> None:
        title = "Summary for inversion result"
        display_width = len(title)
        double_line = "=" * display_width
        single_line = "-" * display_width
        print(title)
        if display_lines: print(double_line)
        print(self.success_or_not.upper())
        if display_lines: print(single_line)
        for key, val in self.res.items():
            if key != "success":
                print(f"{key}: {val}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.success_or_not})"


class InversionRunner:
    def __init__(self, inv_problem: BaseProblem, inv_options: InversionOptions) -> None:
        self.inv_problem = inv_problem
        self.inv_options = inv_options
        # dispatch inversion_solver from self.inv_options, validation is done by solver
        self.inv_solve = self._dispatch_solver()(inv_problem, inv_options)

    def run(self) -> InversionResult:
        res_dict = self.inv_solve()
        self.inv_result = InversionResult(res_dict)
        return self.inv_result

    def _dispatch_solver(self) -> Type[BaseSolver]:
        tool = self.inv_options.get_tool()
        # look up solver_dispatch_table to return constructor for a BaseSolver subclass
        if isinstance(tool, str):
            return solver_dispatch_table[tool]
        else:      # self-defined BaseSolver (note that a BaseSolver object is a callable)
            return self.inv_options.tool

    def summary(self):      # TODO to test
        title = "Summary for inversion runner"
        subtitle_result = "Trained with the following result:"
        subtitle_options = "With inversion solver defined as below:"
        subtitle_problem = "For inversion problem defined as below:"
        display_width = max(len(title), len(subtitle_result), len(subtitle_options), len(subtitle_problem))
        double_line = "=" * display_width
        single_line = "-" * display_width
        print(title)
        print(double_line)
        if hasattr(self, "inv_result"):
            print(f"{subtitle_result}\n")
            self.inv_result._summary(False)
            print(single_line)
        else:
            print("Inversion hasn't started, try `runner.run()` to see result")
            print(single_line)
        print(f"{subtitle_options}\n")
        self.inv_options._summary(False)
        print(single_line)
        print(f"{subtitle_problem}\n")
        self.inv_problem._summary(False)
