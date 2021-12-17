from cofi.cofi_objective import LeastSquareObjective, Model
from cofi.cofi_objective.base_forward import LinearFittingFwd

import numpy as np
from typing import Callable, Union


class LinearFitting(LeastSquareObjective):
    def __init__(
        self,
        X,
        Y,
        params_count,
        design_matrix: Callable = None,
        forward: LinearFittingFwd = None,
        initial_model: Union[Model, np.ndarray, list] = None,
    ):
        self.calc_design_matrix = design_matrix
        if forward and hasattr(forward, "design_matrix"):
            self.calc_design_matrix = forward.design_matrix
        elif forward is None:
            if design_matrix is None:
                raise ValueError(
                    "Please specify at least one of design_matrix and forward"
                )
            forward = LinearFittingFwd(params_count, design_matrix)

        super().__init__(X, Y, forward, initial_model)

    def design_matrix(self):
        if hasattr(self, "_design_matrix"):
            return self._design_matrix
        else:
            self._design_matrix = self.calc_design_matrix(self.X)
            return self._design_matrix

    def gradient(self, model: Union[Model, np.ndarray]):
        return np.squeeze(self.jacobian(model).T @ self.residual(model))

    def hessian(self, model: Union[Model, np.ndarray]):
        g = self.design_matrix()
        return g.T @ g

    def jacobian(self, model: Union[Model, np.ndarray]):
        return self.design_matrix()
