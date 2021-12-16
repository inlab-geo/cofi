from cofi.cofi_objective import LeastSquareObjective
from cofi.cofi_objective.base_forward import LinearFittingFwd

from typing import Callable, Union


class LinearFitting(LeastSquareObjective):
    def __init__(self, X, Y, params_count, basis_transform: Callable = None, forward: LinearFittingFwd = None):
        self.basis_transform = basis_transform
        if forward and hasattr(forward, "basis_transform"):
            self.basis_transform = forward.basis_transform
        elif forward is None:
            if basis_transform is None:
                raise ValueError("Please specify at least one of basis_transform and forward")
            forward = LinearFittingFwd(params_count, basis_transform)

        super().__init__(X, Y, forward)

    def misfit(self):
        if self.forward:
            return 

    def data_x(self):
        return self.X

    def data_y(self):
        return self.Y

