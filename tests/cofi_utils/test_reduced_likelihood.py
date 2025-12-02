import pytest
import numpy as np

from cofi.utils import ReducedLikelihood
from cofi.utils._lik_base import DimensionMismatchError


def test_case_none():
    # Setup
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return np.array([m[0], m[1], m[0] + m[1]])
    G = np.array([[1, 0], [0, 1], [1, 1]])
    Cd_ref = np.eye(3)

    # Create likelihood
    lik = ReducedLikelihood(data=data, forward_func=forward_func, G=G, Cd_ref=Cd_ref, case='none')

    # Test model
    model = np.array([1.0, 2.0])

    # Test log_likelihood
    log_p = lik.log_likelihood(model)
    assert isinstance(log_p, (int, float))

    # Test gradient
    grad = lik.gradient(model)
    assert grad.shape == (2,)

    # Test hessian
    hess = lik.hessian(model)
    assert hess.shape == (2, 2)

    # Test get_ml_cov returns Cd_ref
    Cd_ml = lik.get_ml_cov(model)
    assert np.allclose(Cd_ml, Cd_ref)


def test_case_scaled():
    # Setup
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return np.array([m[0], m[1], m[0] + m[1]])
    G = np.array([[1, 0], [0, 1], [1, 1]])
    Ctilde = np.eye(3)

    # Create likelihood
    lik = ReducedLikelihood(data=data, forward_func=forward_func, G=G, Cd_ref=Ctilde, case='scaled')

    # Test model
    model = np.array([1.0, 2.0])

    # Test log_likelihood
    log_p = lik.log_likelihood(model)
    assert isinstance(log_p, (int, float))

    # Test gradient
    grad = lik.gradient(model)
    assert grad.shape == (2,)

    # Test hessian
    hess = lik.hessian(model)
    assert hess.shape == (2, 2)

    # Test get_ml_cov returns scaled covariance
    Cd_ml = lik.get_ml_cov(model)
    assert Cd_ml.shape == (3, 3)
    # Check it's a scaled version of Ctilde
    assert np.allclose(Cd_ml / Cd_ml[0,0], Ctilde)


def test_case_spherical():
    # Setup
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return np.array([m[0], m[1], m[0] + m[1]])
    G = np.array([[1, 0], [0, 1], [1, 1]])

    # Create likelihood
    lik = ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='spherical')

    # Test model
    model = np.array([1.0, 2.0])

    # Test log_likelihood
    log_p = lik.log_likelihood(model)
    assert isinstance(log_p, (int, float))

    # Test gradient
    grad = lik.gradient(model)
    assert grad.shape == (2,)

    # Test hessian
    hess = lik.hessian(model)
    assert hess.shape == (2, 2)

    # Test get_ml_cov returns spherical (diagonal with same variance)
    Cd_ml = lik.get_ml_cov(model)
    assert Cd_ml.shape == (3, 3)
    # Check diagonal
    assert np.allclose(np.diag(Cd_ml), Cd_ml[0,0] * np.ones(3))
    # Check off-diagonal is zero
    assert np.allclose(Cd_ml - np.diag(np.diag(Cd_ml)), 0)


def test_case_diag():
    # Setup
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return np.array([m[0], m[1], m[0] + m[1]])
    G = np.array([[1, 0], [0, 1], [1, 1]])

    # Create likelihood
    lik = ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='diag')

    # Test model
    model = np.array([1.0, 2.0])

    # Test log_likelihood
    log_p = lik.log_likelihood(model)
    assert isinstance(log_p, (int, float))

    # Test gradient
    grad = lik.gradient(model)
    assert grad.shape == (2,)

    # Test hessian
    hess = lik.hessian(model)
    assert hess.shape == (2, 2)

    # Test get_ml_cov returns diagonal matrix
    Cd_ml = lik.get_ml_cov(model)
    assert Cd_ml.shape == (3, 3)
    # Check it's diagonal
    assert np.allclose(Cd_ml - np.diag(np.diag(Cd_ml)), 0)


def test_case_full():
    # Setup
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return np.array([m[0], m[1], m[0] + m[1]])
    G = np.array([[1, 0], [0, 1], [1, 1]])

    # Create likelihood
    lik = ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='full')

    # Test model
    model = np.array([1.0, 2.0])

    # Test log_likelihood
    log_p = lik.log_likelihood(model)
    assert isinstance(log_p, (int, float))

    # Test gradient
    grad = lik.gradient(model)
    assert grad.shape == (2,)

    # Test hessian
    hess = lik.hessian(model)
    assert hess.shape == (2, 2)

    # Test get_ml_cov returns full covariance
    Cd_ml = lik.get_ml_cov(model)
    assert Cd_ml.shape == (3, 3)


def test_invalid_case():
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return m
    G = np.array([[1], [1], [1]])

    with pytest.raises(ValueError, match=r".*Unknown case.*"):
        ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='invalid')


def test_missing_cd_ref():
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return m
    G = np.array([[1], [1], [1]])

    # Test 'none' case requires Cd_ref
    with pytest.raises(ValueError, match=r".*Cd_ref is required.*"):
        ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='none')

    # Test 'scaled' case requires Cd_ref
    with pytest.raises(ValueError, match=r".*Cd_ref is required.*"):
        ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='scaled')


def test_missing_jacobian():
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return m

    # Create without G
    lik = ReducedLikelihood(data=data, forward_func=forward_func, case='spherical')

    # Should raise error when trying to evaluate (TypeError from model_shape being None)
    with pytest.raises((ValueError, TypeError)):
        lik.log_likelihood(np.array([1.0]))


def test_dimension_mismatch():
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return np.array([m[0], m[1], m[0] + m[1]])
    G = np.array([[1, 0], [0, 1], [1, 1]])

    lik = ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='spherical')

    # Wrong model size - should raise DimensionMismatchError
    # (Note: The error message shows the dimension as an integer, not tuple)
    with pytest.raises(DimensionMismatchError, match=r".*model has dimension.*"):
        lik.log_likelihood(np.array([1.0, 2.0, 3.0]))  # size 3 instead of 2


def test_caching():
    data = np.array([1.0, 2.0, 3.0])
    def forward_func(m):
        return np.array([m[0], m[1], m[0] + m[1]])
    G = np.array([[1, 0], [0, 1], [1, 1]])

    lik = ReducedLikelihood(data=data, forward_func=forward_func, G=G, case='spherical')

    model = np.array([1.0, 2.0])

    # First evaluation
    log_p1 = lik.log_likelihood(model)
    grad1 = lik.gradient(model)
    hess1 = lik.hessian(model)

    # Second evaluation (should use cache)
    log_p2 = lik.log_likelihood(model)
    grad2 = lik.gradient(model)
    hess2 = lik.hessian(model)

    # Should be identical (cached)
    assert log_p1 == log_p2
    assert np.array_equal(grad1, grad2)
    assert np.array_equal(hess1, hess2)

    # Different model should not use cache
    model2 = np.array([2.0, 3.0])
    log_p3 = lik.log_likelihood(model2)
    assert log_p3 != log_p1
