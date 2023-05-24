from numbers import Number
import numpy as np

from ._reg_base import BaseRegularization


class ModelCovariance(BaseRegularization):
    r"""CoFI's utility class to calculate model prior distribution given
    
    TODO Document me
    """
    def __init__(self, model_covariance_inv, mean_model):
        self._Cminv = model_covariance_inv
        self._mu = mean_model
        self._model_shape = mean_model.shape

    def reg(self, model: np.ndarray) -> Number:
        pass

    def gradient(self, model: np.ndarray) -> np.ndarray:
        return super().gradient(model)

    def hessian(self, model: np.ndarray) -> np.ndarray:
        return super().hessian(model)

    @property
    def model_shape(self) -> tuple:
        return self._model_shape


class GaussianPrior(ModelCovariance):
    r"""CoFI's utility class to calculate the Gaussian prior, given the inverse of
    model covariance matrix and the mean model

    :math:`GaussianPrior(C_m^{-1}, \mu) = -\frac{1}{2}(m-\mu)^TC_m^{-1}(m-\mu)`

    Parameters
    ----------
    TODO
    """

    def __init__(self, model_covariance_inv, mean_model):
        self._Cminv = model_covariance_inv
        self._mu = mean_model
        self._model_shape = mean_model.shape

    def reg(self, model: np.ndarray) -> Number:
        pass

    def gradient(self, model: np.ndarray) -> np.ndarray:
        return super().gradient(model)

    def hessian(self, model: np.ndarray) -> np.ndarray:
        return super().hessian(model)

    @property
    def model_shape(self) -> tuple:
        return self._model_shape
