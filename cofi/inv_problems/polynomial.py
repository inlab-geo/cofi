import numpy as np
# https://numpy.org/doc/stable/reference/routines.polynomials.html

from ..base_problem import BaseProblem


class PolynomialProblem(BaseProblem):
    def __init__(self, *kwargs):
        super().__init__(kwargs)

    def suggest_solvers(self):
        raise NotImplementedError()
