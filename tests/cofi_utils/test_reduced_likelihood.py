# tests/test_reduced_likelihood.py
import numpy as np
import pytest
from scipy import sparse
from numpy.testing import assert_allclose

from cofi.utils import ReducedLikelihood
from cofi.utils._lik_base import DimensionMismatchError

# ---------------------------
# Helper utilities
# ---------------------------
def numeric_gradient(func, model, eps=1e-6):
    """Central finite-difference gradient."""
    model = np.asarray(model, dtype=float).ravel()
    grad = np.zeros(model.size, dtype=float)
    for i in range(model.size):
        m1 = model.copy(); m2 = model.copy()
        m1[i] -= eps; m2[i] += eps
        f1 = func(m1); f2 = func(m2)
        grad[i] = (f2 - f1) / (2 * eps)
    return grad

def numeric_hessian(grad_func, model, eps=1e-6):
    """Central finite-difference Jacobian of the gradient -> Hessian."""
    model = np.asarray(model, dtype=float).ravel()
    n = model.size
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        m1 = model.copy(); m2 = model.copy()
        m1[i] -= eps; m2[i] += eps
        g1 = grad_func(m1); g2 = grad_func(m2)
        H[:, i] = (g2 - g1) / (2 * eps)
    return H

# simple linear forward model used in many tests
def linear_forward(m):
    # m expected of size 2
    return np.array([m[0], m[1], m[0] + m[1]])


# fixture with small problem
@pytest.fixture
def small_problem():
    data = np.array([1.0, 2.0, 3.0])
    G = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    model = np.array([1.1, 2.1], dtype=float)
    return data, G, model

# ---------------------------
# Parametrized smoke + shape tests
# ---------------------------
@pytest.mark.parametrize("case, requires_cd_ref", [
    ("none", True),
    ("scaled", True),
    ("spherical", False),
    ("diag", False),
    ("full", False),
])
def test_basic_shapes_and_types(case, requires_cd_ref, small_problem):
    data, G, model = small_problem

    Cd_ref = np.eye(3) if requires_cd_ref else None

    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, Cd_ref=Cd_ref, case=case)

    # shapes / types
    logp = lik.log_likelihood(model)
    assert isinstance(logp, (float, np.floating))

    grad = lik.gradient(model)
    assert grad.shape == (2,)
    assert grad.dtype.kind == "f"

    hess = lik.hessian(model)
    assert hess.shape == (2, 2)
    assert hess.dtype.kind == "f"

    Cd_ml = lik.get_ml_cov(model)
    # Cd_ml is expected for all implemented cases (class returns something)
    assert Cd_ml is None or (isinstance(Cd_ml, np.ndarray) and Cd_ml.shape == (3, 3))

# ---------------------------
# Numerical correctness (finite differences)
# ---------------------------
@pytest.mark.parametrize("case", ["none", "scaled", "spherical", "diag", "full"])
def test_gradient_and_hessian_numerical_agreement(case, small_problem):
    """Compare analytic gradient/hessian against finite-difference approximations.

    Tolerances are relaxed for 'diag' (non-smoothness) and 'scaled'/'full' cases.
    """
    data, G, model = small_problem
    Cd_ref = np.eye(3) if case in ("none", "scaled") else None
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, Cd_ref=Cd_ref, case=case)

    # numeric gradient of log-likelihood
    num_grad = numeric_gradient(lambda m: float(lik.log_likelihood(m)), model, eps=1e-6)
    an_grad = lik.gradient(model)

    # numeric hessian of gradient
    num_hess = numeric_hessian(lambda m: lik.gradient(m), model, eps=1e-6)
    an_hess = lik.hessian(model)

    # choose tolerances depending on case (diag may be less smooth)
    if case == "diag":
        atol_grad = 1e-4
        atol_hess = 1e-3
    else:
        atol_grad = 1e-6
        atol_hess = 1e-4

    assert_allclose(an_grad, num_grad, atol=atol_grad, rtol=1e-6)
    assert_allclose(an_hess, num_hess, atol=atol_hess, rtol=1e-6)

# ---------------------------
# Specific formula/property tests for Cd_ml
# ---------------------------
def test_cd_ml_none_equals_cd_ref(small_problem):
    data, G, model = small_problem
    Cd_ref = np.array([[2.0, 0.0, 0.0],
                       [0.0, 3.0, 0.0],
                       [0.0, 0.0, 4.0]])
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, Cd_ref=Cd_ref, case="none")
    Cd_ml = lik.get_ml_cov(model)
    # Should be exactly the provided Cd_ref
    assert_allclose(Cd_ml, Cd_ref, atol=0.0)

def test_cd_ml_scaled_proportionality(small_problem):
    data, G, model = small_problem
    Ctilde = np.array([[1.0, 0.2, 0.1],
                       [0.2, 1.0, 0.0],
                       [0.1, 0.0, 1.0]])
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, Cd_ref=Ctilde, case="scaled")
    Cd_ml = lik.get_ml_cov(model)

    assert Cd_ml.shape == Ctilde.shape
    # Cd_ml should be (a / N) * Ctilde so it must be proportional to Ctilde:
    # compare normalized by their (0,0) element to avoid floating scale.
    assert_allclose(Cd_ml / Cd_ml[0, 0], Ctilde / Ctilde[0, 0], rtol=1e-12, atol=1e-12)

def test_cd_ml_spherical_is_diagonal_and_equal_entries(small_problem):
    data, G, model = small_problem
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, case="spherical")
    Cd_ml = lik.get_ml_cov(model)
    assert Cd_ml.shape == (3, 3)
    diag = np.diag(Cd_ml)
    # All diagonal entries equal (within tol)
    assert_allclose(diag, diag[0] * np.ones_like(diag), rtol=1e-12, atol=1e-12)
    # Off-diagonals are zero
    assert_allclose(Cd_ml - np.diag(np.diag(Cd_ml)), np.zeros_like(Cd_ml), atol=1e-12)

def test_cd_ml_diag_is_diagonal_and_nonnegative(small_problem):
    data, G, model = small_problem
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, case="diag")
    Cd_ml = lik.get_ml_cov(model)
    assert Cd_ml.shape == (3, 3)
    # Verify diagonal structure
    assert_allclose(Cd_ml - np.diag(np.diag(Cd_ml)), np.zeros_like(Cd_ml), atol=1e-12)
    # Verify nonnegative diagonal
    assert np.all(np.diag(Cd_ml) >= 0.0)

def test_cd_ml_full_is_outer_product(small_problem):
    data, G, model = small_problem
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, case="full")
    Cd_ml = lik.get_ml_cov(model)
    # Cd_ml should equal outer(residual, residual)
    residual = data - linear_forward(model)
    expected = np.outer(residual, residual)
    assert_allclose(Cd_ml, expected)

# ---------------------------
# Sparse G handling (diag case uses sparse D)
# ---------------------------
def test_diag_case_with_sparse_G(small_problem):
    data, G, model = small_problem
    G_sparse = sparse.csr_matrix(G)
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G_sparse, case="diag")
    # Check shapes and that hessian returns dense array
    hess = lik.hessian(model)
    assert isinstance(hess, np.ndarray)
    assert hess.shape == (2, 2)

    # Compare with dense-G result to ensure equality
    lik_dense = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, case="diag")
    hess_dense = lik_dense.hessian(model)
    assert_allclose(hess, hess_dense, rtol=1e-12, atol=1e-12)

# ---------------------------
# Error and edge-case tests
# ---------------------------
def test_missing_cd_ref_raises():
    data = np.array([1.0, 2.0, 3.0])
    G = np.array([[1], [1], [1]])
    # case='none' defaults to identity matrix when Cd_ref is not provided
    # Only 'scaled' case requires Cd_ref
    with pytest.raises(ValueError, match="Cd_ref is required"):
        ReducedLikelihood(data=data, forward_func=lambda m: m, G=G, case="scaled")

def test_unknown_case_raises():
    data = np.array([1.0, 2.0, 3.0])
    G = np.array([[1], [1], [1]])
    with pytest.raises(ValueError, match="Unknown case"):
        ReducedLikelihood(data=data, forward_func=lambda m: m, G=G, case="not_a_case")

def test_missing_jacobian_raises_on_evaluation():
    data = np.array([1.0, 2.0, 3.0])
    lik = ReducedLikelihood(data=data, forward_func=lambda m: m, case="spherical")
    # G not set, so evaluation should raise ValueError
    with pytest.raises(ValueError):
        lik.log_likelihood(np.array([1.0]))

def test_model_dimension_mismatch_raises(small_problem):
    data, G, model = small_problem
    lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G, case="spherical")
    # Provided model has wrong size (3 instead of 2)
    with pytest.raises(DimensionMismatchError):
        lik.log_likelihood(np.array([1.0, 2.0, 3.0]))

# ---------------------------
# Caching behaviour
# ---------------------------
# def test_caching_consistency_and_staleness_doc(small_problem):
#     """Test that the cache returns identical values for repeated calls with same model.

#     NOTE: ReducedLikelihood caches results keyed only by the model vector. If the
#     Jacobian `G` is mutated between calls but the model stays the same, the cached
#     values are returned (i.e. the cache is not invalidated on G-change). The test
#     asserts the current behaviour; if you change the implementation later to clear
#     the cache on G assignment, update this test accordingly.
#     """
#     data, G, model = small_problem
#     lik = ReducedLikelihood(data=data, forward_func=linear_forward, G=G.copy(), case="spherical")

#     # first evaluation
#     logp1 = lik.log_likelihood(model)
#     grad1 = lik.gradient(model)
#     hess1 = lik.hessian(model)

#     # second evaluation: cached and identical
#     logp2 = lik.log_likelihood(model)
#     grad2 = lik.gradient(model)
#     hess2 = lik.hessian(model)

#     assert logp1 == logp2
#     assert np.array_equal(grad1, grad2)
#     assert np.array_equal(hess1, hess2)

#     # Mutate G in place and evaluate again with same model.
#     # Current implementation will return cached values (documented expectation here).
#     lik.G[:, :] = lik.G[:, :] * 2.0  # mutate in-place
#     logp3 = lik.log_likelihood(model)
#     assert logp3 == logp1  # still cached (stale) under current implementation
