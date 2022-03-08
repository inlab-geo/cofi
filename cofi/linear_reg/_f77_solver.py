import numpy as np

from .. import Model, BaseSolver, LinearObjective
from ._f77_solver_lib import solve as f77_solve
from ._utils import warn_normal_equation


class LRNormalEquationF77(BaseSolver):
    def __init__(self, objective: LinearObjective):
        self.objective = objective

    def solve(self) -> Model:
        warn_normal_equation()

        G = self.objective.basis_matrix()
        Y = self.objective.data_y()

        res = f77_solve(
            G.shape[1], G.shape[0], np.asfortranarray(G), np.asfortranarray(Y)
        )

        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model
