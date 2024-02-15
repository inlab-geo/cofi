from numbers import Number
from typing import Union, Tuple
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

    :math:`GaussianPrior(C_m^{-1}, \mu) = (m-\mu)^TC_m^{-1}(m-\mu)`

    With gradient of:

    :math:`2\times C_m^{-1} (m-\mu)`

    And hessian of:

    :math:`2\times C_m^{-1}`

    Optionally, the inverse model covariance matrix can be constructed given the
    correlation lengths and sigma.

    Parameters
    ----------
    model_covariance_inv: np.ndarray or (tuple, float)
        the inverse model covariance matrix, either provided by users or specified so
        can be generated, corresponding to :math:`C_m^{-1}` in the formula above. When
        this is a tuple (i.e. CoFI will construct the matrix for you), it should be in
        the form of `(corr_lengths, sigma)`
    mean_model: np.ndarray
        the prior model, corresponding to :math:`\mu` in the formula above

    Raises
    ------
    TypeError
        if arguments aren't in accepted types as described above in "Parameters"
        section
    ValueError
        if shape of model_covariance_inv and mean_model doesn't match, or if shape of
        corr_lengths and mean_model doesn't match

    Examples
    --------

    Generate a Gaussian Prior term for models of size (4,4), with correlation lengths
    of (2,2), and sigma of 0.5:

    >>> from cofi.utils import GaussianPrior
    >>> import numpy as np
    >>> my_prior_model = np.array([[1,2],[3,4]])
    >>> my_reg = GaussianPrior(model_covariance_inv=((2,2), 0.5), mean_model=my_prior_model)
    >>> my_reg(np.array([[0,2],[1,0]]))
    101.53804950804782

    To use together with :class:`cofi.BaseProblem`:

    >>> from cofi import BaseProblem
    >>> my_problem = BaseProblem()
    >>> my_problem.set_regularization(my_reg)
    """

    def __init__(
        self,
        model_covariance_inv: Union[np.ndarray, Tuple[Tuple, float]],
        mean_model: np.ndarray,
    ):
        self._mu = np.ravel(mean_model)
        self._model_shape = mean_model.shape
        self._prepare_covariance_matrix_inv(model_covariance_inv, mean_model)

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

    @property
    def gaussian_model_covariance_inv(self) -> np.ndarray:
        return self._Cminv

    def _prepare_covariance_matrix_inv(self, model_covariance_inv, mean_model):
        if isinstance(model_covariance_inv, np.ndarray):
            mu = self._mu
            Cminv = model_covariance_inv
            if Cminv.shape != (mu.shape[0], mu.shape[0]):
                raise ValueError(
                    f"({(mu.shape[0], mu.shape[0])}) expected for the shape of "
                    f"model_covariance_inv but got matrix of shape {Cminv.shape}"
                )
            self._Cminv = model_covariance_inv
        elif isinstance(model_covariance_inv, (tuple, list)):
            self._Cminv = self._generate_covariance_matrix_inv(
                mean_model.shape,
                model_covariance_inv[0],
                model_covariance_inv[1],
            )
        else:
            raise TypeError(
                "numpy.ndarray or (tuple, float) expected for `model_covariance_inv` "
                f"but got {model_covariance_inv} of type {type(model_covariance_inv)}"
            )

    def _generate_covariance_matrix_inv(
        self, model_shape: tuple, corr_lengths: tuple, sigma: float
    ):
        # ensure model_shape and corr_lengths have the same length
        if len(model_shape) != len(corr_lengths):
            raise ValueError(
                "`model_shape` and `corr_lengths` should have the same lengths, "
                f"but got {len(model_shape)} and {len(corr_lengths)}"
            )
        # generate grid of points for each dimension
        grids = np.meshgrid(*[np.arange(dim) for dim in model_shape], indexing="ij")
        # calculate distances between points for each pair of dimensions
        d_squared = sum([
            (grid.ravel()[None, :] - grid.ravel()[:, None]) ** 2 / corr_length**2
            for grid, corr_length in zip(grids, corr_lengths)
        ])
        # construct correlation matrix
        Cp = np.exp(-np.sqrt(d_squared))
        # construct variance matrix
        Sc = np.zeros((np.prod(model_shape), np.prod(model_shape)))
        np.fill_diagonal(Sc, sigma)
        # calculate covariance matrix
        covariance_matrix = Sc @ Cp @ Sc
        # calculate inverse covariance matrix
        covariance_matrix_inv = np.linalg.inv(covariance_matrix)
        return covariance_matrix_inv

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
