from numbers import Number
import numpy as np

from .. import BaseProblem


_default_data_x = np.array([1,2,3,4,5])
_default_data_y = np.vectorize(lambda x_i: 2 + x_i + x_i**2)(_default_data_x) + np.random.randn(_default_data_x.shape[0])


class PolynomialProblem(BaseProblem):
    def __init__(self, x=_default_data_x, y=_default_data_y, degree=2, **kwargs):
        basis_matrix = np.array([x**i for i in range(degree+1)])
        self.set_dataset(basis_matrix, y)  # access by properties: self.data_x, self.data_y
        super().__init__(kwargs)

    def objective(self, model: np.ndarray) -> Number:
        raise NotImplementedError

    def gradient(self, model: np.ndarray) -> Number:
        raise NotImplementedError

    def hessian(self, model: np.ndarray) -> Number:
        raise NotImplementedError

    def residual(self, model: np.ndarray) -> Number:
        raise NotImplementedError

    def jacobian(self, model: np.ndarray) -> Number:
        raise NotImplementedError

    def data_misfit(self, model: np.ndarray) -> Number:
        raise NotImplementedError

    def forward(self, model: np.ndarray) -> Number:
        return self.data_x @ model

    def suggest_solvers(self):
        raise NotImplementedError()
