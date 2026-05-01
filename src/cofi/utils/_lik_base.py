"""Base class for likelihood functions in CoFI."""

from abc import abstractmethod, ABCMeta
from numbers import Number
from functools import reduce
from typing import Optional, Callable, Tuple
import numpy as np
from scipy import sparse

# Assuming these exist in the cofi package
# from .._exceptions import DimensionMismatchError

# For now, let's define this locally
class DimensionMismatchError(Exception):
    def __init__(self, entered_name, entered_dimension, expected_source, expected_dimension):
        super().__init__(
            f"{entered_name} has dimension {entered_dimension}, but {expected_source} "
            f"expects dimension {expected_dimension}"
        )


EPS = 1e-154


class BaseLikelihood(metaclass=ABCMeta):
    r"""Base class for a likelihood function

    This base class provides a consistent interface for likelihood functions
    used in Bayesian inverse problems. Concrete implementations should define
    how to evaluate the log-likelihood and its derivatives.

    .. rubric:: Basic interface

    The basic properties / methods for a likelihood function in ``cofi.utils``
    include the following:

    .. autosummary::
        BaseLikelihood.model_shape
        BaseLikelihood.model_size
        BaseLikelihood.log_likelihood
        BaseLikelihood.gradient
        BaseLikelihood.hessian
        BaseLikelihood.__call__
        BaseLikelihood.get_ml_cov

    """

    def __init__(self):
        """Initialize the base likelihood class."""
        pass

    @property
    @abstractmethod
    def model_shape(self) -> tuple:
        """The shape of models that current likelihood function accepts.
        
        Returns
        -------
        tuple
            The shape of the model parameter array
        """
        raise NotImplementedError

    @property
    def model_size(self) -> Number:
        """The number of unknowns that current likelihood function accepts.
        
        Returns
        -------
        Number
            The total number of model parameters
        """
        return reduce(lambda a, b: a * b, np.array(self.model_shape), 1)

    def __call__(self, model: np.ndarray) -> float:
        r"""A class instance itself can also be called as a function.

        It works exactly the same as :meth:`log_likelihood`.

        In other words, the following two usages are exactly the same::

        >>> my_likelihood = GaussianLikelihood(data, forward_func)
        >>> log_p = my_likelihood(np.array([1,2,3]))                    # usage 1
        >>> log_p = my_likelihood.log_likelihood(np.array([1,2,3]))     # usage 2

        Parameters
        ----------
        model : np.ndarray
            The model parameters to evaluate

        Returns
        -------
        float
            The log-likelihood value
        """
        return self.log_likelihood(model)

    @abstractmethod
    def log_likelihood(self, model: np.ndarray) -> float:
        """Evaluate the log-likelihood function at a given model.
        
        Parameters
        ----------
        model : np.ndarray
            The model parameters at which to evaluate the log-likelihood
            
        Returns
        -------
        float
            The log-likelihood value log p(d|m)
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, model: np.ndarray) -> np.ndarray:
        """The gradient of log-likelihood with respect to model parameters.

        The gradient has shape :math:`(M,)` where :math:`M` is the number
        of model parameters.
        
        Parameters
        ----------
        model : np.ndarray
            The model parameters at which to evaluate the gradient
            
        Returns
        -------
        np.ndarray
            The gradient vector of shape (M,)
        """
        raise NotImplementedError

    @abstractmethod
    def hessian(self, model: np.ndarray) -> np.ndarray:
        """The Hessian of log-likelihood with respect to model parameters.

        The Hessian has shape :math:`(M,M)` where :math:`M` is the number
        of model parameters.
        
        Parameters
        ----------
        model : np.ndarray
            The model parameters at which to evaluate the Hessian
            
        Returns
        -------
        np.ndarray
            The Hessian matrix of shape (M, M)
        """
        raise NotImplementedError

    def get_ml_cov(self, model: np.ndarray) -> Optional[np.ndarray]:
        """Get maximum likelihood estimate of data covariance (if applicable).
        
        This is an optional method that some likelihood implementations may
        provide to return the maximum likelihood estimate of the data covariance
        matrix for a given model.
        
        Parameters
        ----------
        model : np.ndarray
            The model parameters
            
        Returns
        -------
        Optional[np.ndarray]
            The estimated data covariance matrix, or None if not applicable
        """
        return None
