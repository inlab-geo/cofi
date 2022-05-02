import numpy as np

from .. import Model, BaseSolver, LinearObjective
from ._f90_solver_lib import f90_lr_mod
from ._utils import warn_normal_equation


class LRNormalEquationF90(BaseSolver):
    def __init__(self, objective: LinearObjective):
        self.objective = objective

    def solve(self) -> Model:
        warn_normal_equation()

        G = self.objective.basis_matrix()
        Y = self.objective.data_y()

        res = f90_lr_mod.solve(
            G.shape[1], G.shape[0], np.asfortranarray(G), np.asfortranarray(Y)
        )

        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model
