from typing import Union

import numpy as np

from .. import Model, BaseObjective


class ExpDecay(BaseObjective):
    """Defines the problem of exponential decay (sum of exponentials).

    :math:`y = f(t) = \sum_{i=0}^{M}{a_i e^{-b_i t}}` where :math:`M` is the
    number of addends, and is equal to the number of parameters divided by 2.

    Must implement the 'misfit' function.
    Depending on solvers, the following functions may also need to be provided:
    `residual(model)`, `jacobian(model)`, `gradient(model)`, `hessian(model)`

    Apart from that, `data_x()`, `data_y()`, `initial_model()` are also required for
    some solvers.

    For parallel computing, respective functions like `jacobian` need to have its
    parallel version and have `_mpi` as suffix of the function name, with additional
    range of data in the signatures (e.g. `jacobian_mpi(model, n, m)`)
    """

    def __init__(self, data_x, data_y, initial_model: Union[Model, np.ndarray]):
        self.x = np.asanyarray(data_x)
        self.y = np.asanyarray(data_y)
        if isinstance(initial_model, Model):
            initial_model = initial_model.values()
        self.m0 = np.asanyarray(initial_model)
        self.n_params = self.m0.shape[0] if len(self.m0.shape) > 0 else 1

        if self.n_params % 2 != 0:
            raise ValueError(
                "Exponential decay sums need to have an even number of parameters, but"
                f" got {self.n_params} instead"
            )

        self._last_validated_model = None

    def _forward(self, model: Union[Model, np.ndarray], ret_model=False):
        model = self._validate_model(model)

        yhat = np.zeros_like(self.x)
        for i in range(int(self.n_params / 2)):
            yhat += model[i * 2] * np.exp(-model[i * 2 + 1] * self.x)
        return (yhat, model) if ret_model else yhat

    def _forward_mpi(self, model: Union[Model, np.ndarray], n, m, ret_model=False):
        model = self._validate_model(model)

        yhat = np.zeros((m - n,))
        for i in range(int(self.n_params / 2)):
            yhat += model[i * 2] * np.exp(-model[i * 2 + 1] * self.x[n:m])
        return (yhat, model) if ret_model else yhat

    def residual(self, model: Union[Model, np.ndarray]):
        return self.residual_mpi(model, 0, np.shape(self.x)[0])

    def residual_mpi(self, model: Union[Model, np.ndarray], n, m):
        yhat = self._forward_mpi(model, n, m)
        return yhat - self.y[n:m]

    def misfit(self, model: Union[Model, np.ndarray]):
        residuals = self.residual(model)
        return residuals @ residuals

    def misfit_mpi(self, model: Union[Model, np.ndarray], n, m):
        residuals = self.residual_mpi(model, n, m)
        return residuals @ residuals

    def jacobian(self, model: Union[Model, np.ndarray]):
        return self.jacobian_mpi(model, 0, np.shape(self.x)[0])

    def jacobian_mpi(self, model: Union[Model, np.ndarray], n, m):
        model = self._validate_model(model)

        jac = np.zeros([m - n, self.n_params])
        for i in range(int(self.n_params / 2)):
            for j in range(n, m):
                jac[j - n, i * 2] = np.exp(-model[i * 2 + 1] * self.x[j])
                jac[j - n, i * 2 + 1] = (
                    -model[i * 2] * self.x[j] * np.exp(-model[i * 2 + 1] * self.x[j])
                )
        return jac

    def gradient(self, model: Union[Model, np.ndarray]):
        yhat, model = self._forward(model, True)
        jac = self.jacobian(model)
        return jac.T @ (yhat - self.y)

    def gradient_mpi(self, model: Union[Model, np.ndarray], n, m):
        yhat, model = self._forward_mpi(model, n, m, True)
        jac = self.jacobian_mpi(model, n, m)
        return jac.T @ (yhat - self.y[n:m])

    def hessian(self, model: Union[Model, np.ndarray]):
        # using the standard approximation (J^T J)
        jac = self.jacobian(model)
        hessian = jac.T @ jac
        return hessian

    def hessian_mpi(self, model: Union[Model, np.ndarray], n, m):
        jac = self.jacobian_mpi(model, n, m)
        hessian = jac.T @ jac
        return hessian

    def data_x(self):
        return self.x

    def data_y(self):
        return self.y

    def initial_model(self):
        return self.m0

    def params_size(self):
        return self.n_params

    def _validate_model(self, model: Union[Model, np.ndarray]) -> np.ndarray:
        if (
            model is self._last_validated_model
        ):  # validated already (and converted if needed)
            return model

        if isinstance(model, Model):
            n_params = model.length()
            model = np.asanyarray(model.values())
        else:
            model = np.asanyarray(model)
            n_params = model.shape[0] if len(model.shape) > 0 else 1

        if n_params != self.n_params:
            raise ValueError(
                "Model length doesn't match initialisation, expected"
                f" {self.n_params} parameters but got {n_params} instead"
            )

        self._last_validated_model = model
        return model
