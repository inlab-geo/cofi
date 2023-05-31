from numbers import Number
import numpy as np

from ._reg_base import BaseRegularization
from .._exceptions import DimensionMismatchError


class ModelCovariance(BaseRegularization):
    r"""CoFI's utility abstract class to calculate model prior distribution

    See :class:`GaussianPrior` for a concrete subclass example.
    """


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
        self._mu = np.ravel(mean_model)
        self._validate_shape()
        self._model_shape = mean_model.shape

    def reg(self, model: np.ndarray) -> Number:
        flat_m = self._validate_model(model)
        diff_m = flat_m - self._mu
        return diff_m.T @ self._Cminv @ diff_m

    def gradient(self, model: np.ndarray) -> np.ndarray:
        flat_m = self._validate_model(model)
        return 2 * self._Cminv @ (flat_m - self._mu)

    def hessian(self, model: np.ndarray) -> np.ndarray:
        return 2 * self._Cminv

    @property
    def model_shape(self) -> tuple:
        return self._model_shape
    
    def _validate_shape(self):
        Cminv = self._Cminv
        mu = self._mu
        if Cminv.shape != (mu.shape[0], mu.shape[0]):
            raise ValueError(
                f"({(mu.shape[0], mu.shape[0])}) expected for the shape of "
                f"model_covariance_inv but got matrix of shape {Cminv.shape}"
            )

    def _validate_model(self, model):
        flat_m = np.ravel(model)
        if flat_m.size != self.model_size:
            raise DimensionMismatchError(
                entered_name="model",
                entered_dimension=model.shape,
                expected_source="model_size",
                expected_dimension=self.model_size,
            )
        return flat_m
