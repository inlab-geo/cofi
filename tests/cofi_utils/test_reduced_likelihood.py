# tests/test_reduced_likelihood.py
import numpy as np
import pytest
from scipy import sparse
from numpy.testing import assert_allclose

from cofi.utils import ReducedLikelihood, SquaredExponentialKernel
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

def test_missing_jacobian_raises_on_gradient():
    data = np.array([1.0, 2.0, 3.0])
    lik = ReducedLikelihood(data=data, forward_func=lambda m: m, case="spherical")
    # G not set: log_likelihood works, but gradient/hessian should raise ValueError
    lik.log_likelihood(np.array([1.0, 2.0, 3.0]))  # should not raise
    with pytest.raises(ValueError, match="Cannot compute derivatives when G=None"):
        lik.gradient(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="Cannot compute derivatives when G=None"):
        lik.hessian(np.array([1.0, 2.0, 3.0]))

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


# ===========================================================
# Tests for 'kernel' case
# ===========================================================

@pytest.fixture
def kernel_problem():
    """Fixture for kernel tests: data, G, model, kernel, full model [m, eta]."""
    data = np.array([1.0, 2.0, 3.0])
    G = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    model_m = np.array([1.1, 2.1], dtype=float)
    positions = np.arange(3, dtype=float)
    kernel = SquaredExponentialKernel(positions, nugget=1e-8)
    eta = 0.5  # log(correlation length)
    model_full = np.concatenate([model_m, [eta]])
    return data, G, model_m, kernel, model_full


def test_kernel_shapes(kernel_problem):
    """Shape and type test for kernel case."""
    data, G, _, kernel, model_full = kernel_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel', kernel=kernel,
    )

    logp = lik.log_likelihood(model_full)
    assert isinstance(logp, (float, np.floating))

    grad = lik.gradient(model_full)
    assert grad.shape == (3,)  # n_m + n_eta = 2 + 1
    assert grad.dtype.kind == "f"

    hess = lik.hessian(model_full)
    assert hess.shape == (3, 3)
    assert hess.dtype.kind == "f"

    Cd_ml = lik.get_ml_cov(model_full)
    assert isinstance(Cd_ml, np.ndarray) and Cd_ml.shape == (3, 3)


def test_kernel_gradient_numerical(kernel_problem):
    """Analytic gradient vs central finite differences over full [m, eta]."""
    data, G, _, kernel, model_full = kernel_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel', kernel=kernel,
    )

    num_grad = numeric_gradient(
        lambda p: float(lik.log_likelihood(p)), model_full, eps=1e-6,
    )
    an_grad = lik.gradient(model_full)

    assert_allclose(an_grad, num_grad, atol=1e-5, rtol=1e-5)


def test_kernel_hessian_numerical(kernel_problem):
    """Analytic Hessian vs finite-difference Jacobian of gradient."""
    data, G, _, kernel, model_full = kernel_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel', kernel=kernel,
    )

    num_hess = numeric_hessian(
        lambda p: lik.gradient(p), model_full, eps=1e-5,
    )
    an_hess = lik.hessian(model_full)

    assert_allclose(an_hess, num_hess, atol=1e-4, rtol=1e-4)


def test_kernel_cd_ml_proportional_to_K(kernel_problem):
    """Cd_ml should equal (a/N) * K(eta)."""
    data, G, _, kernel, model_full = kernel_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel', kernel=kernel,
    )

    Cd_ml = lik.get_ml_cov(model_full)
    eta = model_full[-1]
    K = kernel.evaluate(eta)

    # Cd_ml proportional to K
    assert_allclose(Cd_ml / Cd_ml[0, 0], K / K[0, 0], rtol=1e-10, atol=1e-10)


def test_kernel_missing_kernel_raises():
    """kernel without kernel should raise ValueError."""
    data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="kernel is required"):
        ReducedLikelihood(
            data=data, forward_func=linear_forward,
            case='kernel',
        )


def test_kernel_wrong_model_size_raises(kernel_problem):
    """Wrong model size should raise DimensionMismatchError."""
    data, G, _, kernel, _ = kernel_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel', kernel=kernel,
    )

    # model of size 2 (missing eta)
    with pytest.raises(DimensionMismatchError):
        lik.log_likelihood(np.array([1.0, 2.0]))


def test_kernel_multiple_eta_values(kernel_problem):
    """Test at several eta values to ensure stability."""
    data, G, _, kernel, _ = kernel_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel', kernel=kernel,
    )

    for eta_val in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        model = np.array([1.1, 2.1, eta_val])
        logp = lik.log_likelihood(model)
        grad = lik.gradient(model)
        hess = lik.hessian(model)
        assert np.isfinite(logp)
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

        # Verify gradient numerically at each eta
        num_grad = numeric_gradient(
            lambda p: float(lik.log_likelihood(p)), model, eps=1e-6,
        )
        assert_allclose(grad, num_grad, atol=1e-5, rtol=1e-5)


# ===========================================================
# Tests for 'kernel_full' case
# ===========================================================

@pytest.fixture
def kernel_full_problem():
    """Fixture for kernel_full tests: model = [m, phi, eta]."""
    data = np.array([1.0, 2.0, 3.0])
    G = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    model_m = np.array([1.1, 2.1], dtype=float)
    positions = np.arange(3, dtype=float)
    kernel = SquaredExponentialKernel(positions, nugget=1e-8)
    phi = -0.5   # log(sigma_d)
    eta = 0.5    # log(l)
    model_full = np.concatenate([model_m, [phi], [eta]])
    return data, G, model_m, kernel, model_full


def test_kernel_full_shapes(kernel_full_problem):
    """Shape and type test for kernel_full case."""
    data, G, _, kernel, model_full = kernel_full_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel_full', kernel=kernel,
    )

    logp = lik.log_likelihood(model_full)
    assert isinstance(logp, (float, np.floating))

    grad = lik.gradient(model_full)
    assert grad.shape == (4,)  # n_m(2) + phi(1) + eta(1)
    assert grad.dtype.kind == "f"

    hess = lik.hessian(model_full)
    assert hess.shape == (4, 4)
    assert hess.dtype.kind == "f"

    Cd_ml = lik.get_ml_cov(model_full)
    assert isinstance(Cd_ml, np.ndarray) and Cd_ml.shape == (3, 3)


def test_kernel_full_gradient_numerical(kernel_full_problem):
    """Analytic gradient vs central finite differences for kernel_full."""
    data, G, _, kernel, model_full = kernel_full_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel_full', kernel=kernel,
    )

    num_grad = numeric_gradient(
        lambda p: float(lik.log_likelihood(p)), model_full, eps=1e-6,
    )
    an_grad = lik.gradient(model_full)

    assert_allclose(an_grad, num_grad, atol=1e-5, rtol=1e-5)


def test_kernel_full_hessian_numerical(kernel_full_problem):
    """Analytic Hessian vs finite-difference Jacobian of gradient."""
    data, G, _, kernel, model_full = kernel_full_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel_full', kernel=kernel,
    )

    num_hess = numeric_hessian(
        lambda p: lik.gradient(p), model_full, eps=1e-5,
    )
    an_hess = lik.hessian(model_full)

    assert_allclose(an_hess, num_hess, atol=1e-4, rtol=1e-4)


def test_kernel_full_without_G(kernel_full_problem):
    """log_likelihood should work without G; gradient/hessian should raise."""
    data, G, _, kernel, model_full = kernel_full_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward,
        case='kernel_full', kernel=kernel, n_params=G.shape[1],
    )

    logp = lik.log_likelihood(model_full)
    assert np.isfinite(logp)

    # Compare with G-enabled version
    lik_with_G = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel_full', kernel=kernel,
    )
    assert_allclose(logp, lik_with_G.log_likelihood(model_full), rtol=1e-12)

    # gradient/hessian should raise
    with pytest.raises(ValueError, match="Cannot compute derivatives"):
        lik.gradient(model_full)
    with pytest.raises(ValueError, match="Cannot compute derivatives"):
        lik.hessian(model_full)


def test_kernel_full_multiple_phi_eta(kernel_full_problem):
    """Test at several phi/eta values for stability."""
    data, G, _, kernel, _ = kernel_full_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel_full', kernel=kernel,
    )

    for phi_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for eta_val in [-0.5, 0.0, 0.5, 1.0]:
            model = np.array([1.1, 2.1, phi_val, eta_val])
            logp = lik.log_likelihood(model)
            grad = lik.gradient(model)
            hess = lik.hessian(model)
            assert np.isfinite(logp), f"logp not finite at phi={phi_val}, eta={eta_val}"
            assert np.all(np.isfinite(grad)), f"grad not finite at phi={phi_val}, eta={eta_val}"
            assert np.all(np.isfinite(hess)), f"hess not finite at phi={phi_val}, eta={eta_val}"

            # Verify gradient numerically
            num_grad = numeric_gradient(
                lambda p: float(lik.log_likelihood(p)), model, eps=1e-6,
            )
            assert_allclose(grad, num_grad, atol=1e-5, rtol=1e-5)


# ===========================================================
# Tests for ReducedLikelihoodManager with kernel
# ===========================================================

def test_manager_mixed_kernel_and_spherical():
    """Manager with one kernel and one spherical dataset."""
    from cofi.utils import ReducedLikelihoodManager

    # Dataset 1: spherical (3 data points)
    data1 = np.array([1.0, 2.0, 3.0])
    def fwd1(m): return np.array([m[0], m[1], m[0] + m[1]])
    G1 = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)

    # Dataset 2: kernel (4 data points)
    data2 = np.array([0.5, 1.5, 2.5, 3.5])
    def fwd2(m): return np.array([m[0], m[1], m[0] - m[1], m[0] * 2])
    G2 = np.array([[1, 0], [0, 1], [1, -1], [2, 0]], dtype=float)

    positions = np.arange(4, dtype=float)
    kernel = SquaredExponentialKernel(positions, nugget=1e-8)

    def jacobian_fn(m, n_data, fwd, fwd_kwargs):
        if n_data == 3:
            return G1
        else:
            return G2

    manager = ReducedLikelihoodManager(
        fwd_funcs=[(fwd1, {}), (fwd2, {})],
        d_obs_list=[data1, data2],
        jacobian_fn=jacobian_fn,
        cases=['spherical', 'kernel'],
        kernels=[None, kernel],
    )

    # model = [m0, m1, eta]
    model = np.array([1.1, 2.1, 0.5])

    # Should not raise
    obj = manager.objective(model)
    grad = manager.gradient(model)
    hess = manager.hessian(model)

    assert np.isfinite(obj)
    assert grad.shape == (3,)
    assert hess.shape == (3, 3)
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))

    # Verify gradient numerically
    # manager.gradient() returns gradient of the objective (neg log-likelihood)
    num_grad = numeric_gradient(lambda p: manager.objective(p), model, eps=1e-6)
    assert_allclose(grad, num_grad, atol=1e-5, rtol=1e-5)


# ===========================================================
# Tests for G=None (gradient-free) support for all cases
# ===========================================================

@pytest.mark.parametrize("case,needs_cd_ref", [
    ("none", True),
    ("scaled", True),
    ("spherical", False),
    ("diag", False),
    ("full", False),
])
def test_loglik_without_G(small_problem, case, needs_cd_ref):
    """log_likelihood() should work without G for all non-kernel cases."""
    data, G, model = small_problem
    n_params = G.shape[1]

    kwargs = dict(
        data=data, forward_func=linear_forward, case=case, n_params=n_params,
    )
    if needs_cd_ref:
        kwargs['Cd_ref'] = np.eye(len(data))

    # Create without G
    lik_no_G = ReducedLikelihood(**kwargs)
    logp_no_G = lik_no_G.log_likelihood(model)
    assert np.isfinite(logp_no_G)

    # Create with G for comparison
    kwargs['G'] = G
    lik_with_G = ReducedLikelihood(**kwargs)
    logp_with_G = lik_with_G.log_likelihood(model)

    assert_allclose(logp_no_G, logp_with_G, rtol=1e-12)


def test_loglik_without_G_kernel(kernel_problem):
    """log_likelihood() should work without G for kernel case."""
    data, G, _, kernel, model_full = kernel_problem
    n_m = G.shape[1]

    lik_no_G = ReducedLikelihood(
        data=data, forward_func=linear_forward,
        case='kernel', kernel=kernel, n_params=n_m,
    )
    logp_no_G = lik_no_G.log_likelihood(model_full)
    assert np.isfinite(logp_no_G)

    lik_with_G = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G,
        case='kernel', kernel=kernel,
    )
    logp_with_G = lik_with_G.log_likelihood(model_full)

    assert_allclose(logp_no_G, logp_with_G, rtol=1e-12)


def test_get_ml_cov_without_G(small_problem):
    """get_ml_cov() should work without G."""
    data, G, model = small_problem

    lik_no_G = ReducedLikelihood(
        data=data, forward_func=linear_forward, case='spherical', n_params=G.shape[1],
    )
    Cd_no_G = lik_no_G.get_ml_cov(model)
    assert isinstance(Cd_no_G, np.ndarray)

    lik_with_G = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G, case='spherical',
    )
    Cd_with_G = lik_with_G.get_ml_cov(model)

    assert_allclose(Cd_no_G, Cd_with_G, rtol=1e-12)


@pytest.mark.parametrize("case", ["none", "scaled", "spherical", "diag", "full"])
def test_gradient_hessian_raise_without_G(case):
    """gradient() and hessian() should raise ValueError when G=None."""
    data = np.array([1.0, 2.0, 3.0])
    model = np.array([1.0, 2.0, 3.0])

    kwargs = dict(
        data=data, forward_func=lambda m: m, case=case, n_params=3,
    )
    if case in ("none", "scaled"):
        kwargs['Cd_ref'] = np.eye(3)

    lik = ReducedLikelihood(**kwargs)

    with pytest.raises(ValueError, match="Cannot compute derivatives when G=None"):
        lik.gradient(model)
    with pytest.raises(ValueError, match="Cannot compute derivatives when G=None"):
        lik.hessian(model)


def test_gradient_hessian_raise_without_G_kernel(kernel_problem):
    """gradient()/hessian() should raise for kernel when G=None."""
    data, G, _, kernel, model_full = kernel_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward,
        case='kernel', kernel=kernel, n_params=G.shape[1],
    )

    with pytest.raises(ValueError, match="Cannot compute derivatives when G=None"):
        lik.gradient(model_full)
    with pytest.raises(ValueError, match="Cannot compute derivatives when G=None"):
        lik.hessian(model_full)


def test_dynamic_G_assignment(small_problem):
    """After assigning G, gradient/hessian should work."""
    data, G, model = small_problem

    # Start without G
    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, case='spherical', n_params=G.shape[1],
    )

    # log_likelihood works
    logp1 = lik.log_likelihood(model)
    assert np.isfinite(logp1)

    # gradient raises
    with pytest.raises(ValueError):
        lik.gradient(model)

    # Assign G
    lik.G = G

    # Now gradient and hessian work
    grad = lik.gradient(model)
    hess = lik.hessian(model)
    assert grad.shape == (G.shape[1],)
    assert hess.shape == (G.shape[1], G.shape[1])
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))

    # log_likelihood still returns correct value
    logp2 = lik.log_likelihood(model)
    assert_allclose(logp1, logp2, rtol=1e-12)


def test_G_setter_invalidates_cache(small_problem):
    """Assigning G should invalidate the cache."""
    data, G, model = small_problem

    lik = ReducedLikelihood(
        data=data, forward_func=linear_forward, G=G, case='spherical',
    )

    # Evaluate to populate cache
    logp1 = lik.log_likelihood(model)
    grad1 = lik.gradient(model)

    # Assign new G (2x the original)
    lik.G = 2.0 * G

    # Gradient should change because G changed
    grad2 = lik.gradient(model)
    # logp should be same (doesn't depend on G)
    logp2 = lik.log_likelihood(model)

    assert_allclose(logp1, logp2, rtol=1e-12)
    # Gradient should differ since G changed
    assert not np.allclose(grad1, grad2)


def test_n_params_consistency_validation():
    """n_params inconsistent with G.shape[1] should raise."""
    data = np.array([1.0, 2.0, 3.0])
    G = np.array([[1, 0], [0, 1], [1, 1]])

    with pytest.raises(ValueError, match="Inconsistent n_params"):
        ReducedLikelihood(
            data=data, forward_func=linear_forward, G=G,
            case='spherical', n_params=5,  # wrong: G has 2 columns
        )


def test_no_G_no_n_params_skips_validation():
    """Without G or n_params, model validation is skipped."""
    data = np.array([1.0, 2.0, 3.0])
    lik = ReducedLikelihood(
        data=data, forward_func=lambda m: m, case='spherical',
    )
    assert lik.model_shape is None
    # Should not raise (validation skipped) — forward function handles size
    logp = lik.log_likelihood(np.array([1.0, 2.0, 3.0]))
    assert np.isfinite(logp)
