from cofi.cofi_objective import LeastSquareObjective
from cofi.cofi_objective.base_forward import BaseForward

from typing import Callable, Union


class LinearFitting(LeastSquareObjective):
    def __init__(self, X, Y, forward: Union[BaseForward, Callable], transform: Callable = None):
        super().__init__(X, Y, forward)
        if transform:
            self.transform = transform
        elif hasattr(forward, "transform"):
            self.transform = forward.transform
        else:
            self.transform = None

    def data_x(self):
        return self.X

    def data_y(self):
        return self.Y

