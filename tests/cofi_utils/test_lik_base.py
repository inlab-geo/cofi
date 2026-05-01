import pytest
import numpy as np

from cofi.utils import BaseLikelihood
from cofi._exceptions import DimensionMismatchError


def test_base_lik():
    class subclass_lik(BaseLikelihood):
        def __init__(self):
            super().__init__()
        @property
        def model_size(self):
            return super().model_size
        @property
        def model_shape(self):
            return super().model_shape
        def log_likelihood(self, model):
            if isinstance(model, np.ndarray):
                return model
            else:
                return super().log_likelihood(model)
        def gradient(self, model):
            return super().gradient(model)
        def hessian(self, model):
            return super().hessian(model)
    test_lik = subclass_lik()
    assert test_lik(np.array([1]))[0] == 1
    with pytest.raises(NotImplementedError): test_lik.model_size
    with pytest.raises(NotImplementedError): test_lik.model_shape
    with pytest.raises(NotImplementedError): test_lik(1)
    with pytest.raises(NotImplementedError): test_lik.gradient(1)
    with pytest.raises(NotImplementedError): test_lik.hessian(1)
    # Test get_ml_cov default behavior (returns None, not NotImplementedError)
    assert test_lik.get_ml_cov(np.array([1])) is None

def test_model_size_calculation():
    """Test that model_size correctly computes from model_shape"""
    class ConcreteLikelihood(BaseLikelihood):
        def __init__(self, shape):
            super().__init__()
            self._shape = shape

        @property
        def model_shape(self):
            return self._shape

        def log_likelihood(self, model):
            return 0.0

        def gradient(self, model):
            return np.zeros(self.model_size)

        def hessian(self, model):
            return np.zeros((self.model_size, self.model_size))

    # Test 1D shape
    lik_1d = ConcreteLikelihood((10,))
    assert lik_1d.model_size == 10

    # Test 2D shape
    lik_2d = ConcreteLikelihood((3, 4))
    assert lik_2d.model_size == 12

    # Test 3D shape
    lik_3d = ConcreteLikelihood((2, 3, 4))
    assert lik_3d.model_size == 24
    print("successful")