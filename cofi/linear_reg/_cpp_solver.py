from cofi import BaseSolver
from cofi.cofi_objective import LinearFittingObjective, Model
from ._cpp_solver_lib import solve as cpp_solve

import numpy as np
from ._utils import warn_normal_equation


class LRNormalEquationCpp(BaseSolver):
    def __init__(self, objective: LinearFittingObjective):
        self.objective = objective

    def solve(self) -> Model:
        warn_normal_equation()

        G = self.objective.design_matrix()
        Y = self.objective.data_y()

        res = cpp_solve(G.shape[1], G.shape[0], G, Y)

        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model
