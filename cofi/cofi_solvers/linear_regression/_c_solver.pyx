# from .lib._c_solver import solve as c_solve
# from .lib._c_solver_lib cimport hello
from libc.math cimport sin
from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import LeastSquareObjective, Model

import numpy as np
from warnings import warn


cdef hello_wrapper():
    print(sin(3.14))


class SimpleLinearRegressionC(BaseSolver):
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
