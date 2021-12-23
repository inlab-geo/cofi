# distutils: sources = lib/_c_solver.c
# distutils: include_dirs = lib/

# from .lib._c_solver import solve as c_solve
cimport _c_solver
from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import LeastSquareObjective, Model

import numpy as np
from warnings import warn


cdef hello_wrapper():
    hello()


class LRNormalEquationC(BaseSolver):
    def __init__(self, objective: LeastSquareObjective):
        hello_wrapper()
        self.objective = objective

    def solve(self) -> Model:
        warn(
            "You are using linear regression formula solver, please note that this is"
            " only for small scale of data"
        )

        G = self.objective.design_matrix()
        Y = self.objective.data_y()
        # res = np.zeros(G.shape[1])
        # c_solve(G, Y, res)
        res = np.linalg.inv(G.T @ G) @ (G.T @ Y)
        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model
