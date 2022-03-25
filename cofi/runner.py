from cofi.base_problem import BaseProblem
from cofi.inversion_options import InversionOptions


class InversionRunner:
    def __init__(self, inv_problem: BaseProblem, inv_options: InversionOptions) -> None:
        self.inv_problem = inv_problem
        self.inv_options = inv_options
