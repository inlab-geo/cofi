# tests/cofi_utils/test_rl_manager.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cofi.utils import ReducedLikelihoodManager

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

# Simple linear forward models
def fwd1(m):
    """Forward model 1: 2 data points from 2 params."""
    return np.array([m[0], m[1]])

def fwd2(m):
    """Forward model 2: 1 data point from sum of params."""
    return np.array([m[0] + m[1]])

def fwd_single(m):
    """Single forward model: 3 data points."""
    return np.array([m[0], m[1], m[0] + m[1]])

# Jacobian functions
def jacobian_fn(model, n_data, fwd_func, fwd_kwargs):
    """Compute Jacobian based on forward function."""
    if fwd_func is fwd1:
        return np.array([[1.0, 0.0], [0.0, 1.0]])
    elif fwd_func is fwd2:
        return np.array([[1.0, 1.0]])
    elif fwd_func is fwd_single:
        return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    else:
        # Generic finite-difference fallback
        eps = 1e-7
        m = np.asarray(model).ravel()
        f0 = fwd_func(m, **fwd_kwargs)
        n_params = m.size
        G = np.zeros((n_data, n_params))
        for i in range(n_params):
            m_pert = m.copy()
            m_pert[i] += eps
            G[:, i] = (fwd_func(m_pert, **fwd_kwargs) - f0) / eps
        return G

# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def single_dataset():
    """Single dataset problem setup."""
    fwd_funcs = [(fwd_single, {})]
    d_obs_list = [np.array([1.0, 2.0, 3.0])]
    model = np.array([1.1, 2.1])
    return fwd_funcs, d_obs_list, model

@pytest.fixture
def multi_dataset():
    """Multiple dataset problem setup."""
    fwd_funcs = [(fwd1, {}), (fwd2, {})]
    d_obs_list = [np.array([1.0, 2.0]), np.array([3.0])]
    model = np.array([1.1, 2.1])
    return fwd_funcs, d_obs_list, model

# ---------------------------
# Basic functionality tests
# ---------------------------
def test_basic_shapes_single_dataset(single_dataset):
    """Test output shapes for single dataset."""
    fwd_funcs, d_obs_list, model = single_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical'
    )

    obj = manager.objective(model)
    assert isinstance(obj, (float, np.floating))

    grad = manager.gradient(model)
    assert grad.shape == (2,)
    assert grad.dtype.kind == 'f'

    hess = manager.hessian(model)
    assert hess.shape == (2, 2)
    assert hess.dtype.kind == 'f'

def test_basic_shapes_multi_dataset(multi_dataset):
    """Test output shapes for multiple datasets."""
    fwd_funcs, d_obs_list, model = multi_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical'
    )

    obj = manager.objective(model)
    assert isinstance(obj, (float, np.floating))

    grad = manager.gradient(model)
    assert grad.shape == (2,)

    hess = manager.hessian(model)
    assert hess.shape == (2, 2)

# ---------------------------
# Cases handling tests
# ---------------------------
@pytest.mark.parametrize("case", ["none", "scaled", "spherical", "diag", "full"])
def test_single_case_string(single_dataset, case):
    """Test that single case string is applied to all datasets."""
    fwd_funcs, d_obs_list, model = single_dataset

    Cd_ref = np.eye(3) if case in ("none", "scaled") else None

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases=case,
        Cd_refs=Cd_ref
    )

    # Should not raise
    obj = manager.objective(model)
    assert isinstance(obj, (float, np.floating))

def test_list_of_cases(multi_dataset):
    """Test that list of cases works correctly."""
    fwd_funcs, d_obs_list, model = multi_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases=['spherical', 'diag']
    )

    obj = manager.objective(model)
    assert isinstance(obj, (float, np.floating))

    # Verify each ReducedLikelihood has correct case
    assert manager.reduced_likelihoods[0].case == 'spherical'
    assert manager.reduced_likelihoods[1].case == 'diag'

# ---------------------------
# Cd_refs handling tests
# ---------------------------
def test_single_cd_ref_applied_to_all(multi_dataset):
    """Test that single Cd_ref array is applied to all datasets."""
    fwd_funcs, d_obs_list, model = multi_dataset

    # Use 'none' case which accepts Cd_ref
    Cd_ref1 = np.eye(2) * 2.0  # For first dataset (2 data points)
    Cd_ref2 = np.eye(1) * 3.0  # For second dataset (1 data point)

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases=['none', 'none'],
        Cd_refs=[Cd_ref1, Cd_ref2]
    )

    obj = manager.objective(model)
    assert isinstance(obj, (float, np.floating))

def test_cd_refs_list(multi_dataset):
    """Test that list of Cd_refs works correctly."""
    fwd_funcs, d_obs_list, model = multi_dataset

    Cd_ref1 = np.eye(2) * 0.5
    Cd_ref2 = np.eye(1) * 0.25

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases=['scaled', 'scaled'],
        Cd_refs=[Cd_ref1, Cd_ref2]
    )

    obj = manager.objective(model)
    assert isinstance(obj, (float, np.floating))

# ---------------------------
# Numerical correctness tests
# ---------------------------
@pytest.mark.parametrize("case", ["spherical", "diag", "full"])
def test_gradient_numerical_agreement(single_dataset, case):
    """Compare analytic gradient against finite-difference."""
    fwd_funcs, d_obs_list, model = single_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases=case
    )

    num_grad = numeric_gradient(manager.objective, model, eps=1e-6)
    an_grad = manager.gradient(model)

    atol = 1e-4 if case == "diag" else 1e-5
    assert_allclose(an_grad, num_grad, atol=atol, rtol=1e-5)

@pytest.mark.parametrize("case", ["spherical", "diag", "full"])
def test_hessian_numerical_agreement(single_dataset, case):
    """Compare analytic Hessian against finite-difference."""
    fwd_funcs, d_obs_list, model = single_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases=case
    )

    num_hess = numeric_hessian(manager.gradient, model, eps=1e-5)
    an_hess = manager.hessian(model)

    atol = 1e-3 if case == "diag" else 1e-4
    assert_allclose(an_hess, num_hess, atol=atol, rtol=1e-4)

# ---------------------------
# get_ml_covs tests
# ---------------------------
def test_get_ml_covs_returns_list(multi_dataset):
    """Test that get_ml_covs returns correct number of covariances."""
    fwd_funcs, d_obs_list, model = multi_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical'
    )

    covs = manager.get_ml_covs(model)
    assert isinstance(covs, list)
    assert len(covs) == 2
    assert covs[0].shape == (2, 2)  # First dataset has 2 data points
    assert covs[1].shape == (1, 1)  # Second dataset has 1 data point

# ---------------------------
# Caching tests
# ---------------------------
def test_cache_hits_with_same_model(single_dataset):
    """Test that cache is hit when evaluating same model."""
    fwd_funcs, d_obs_list, model = single_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical',
        track_stats=True
    )

    # First evaluation
    _ = manager.objective(model)
    stats1 = manager.get_stats()
    assert stats1['n_jacobian_evals'] == 1
    assert stats1['n_cache_hits'] == 0

    # Second evaluation with same model
    _ = manager.objective(model)
    stats2 = manager.get_stats()
    assert stats2['n_jacobian_evals'] == 1  # No new Jacobian computation
    assert stats2['n_cache_hits'] == 1

def test_cache_miss_with_different_model(single_dataset):
    """Test that cache is invalidated with different model."""
    fwd_funcs, d_obs_list, model = single_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical',
        track_stats=True
    )

    # First evaluation
    _ = manager.objective(model)

    # Second evaluation with different model
    new_model = model + 0.1
    _ = manager.objective(new_model)

    stats = manager.get_stats()
    assert stats['n_jacobian_evals'] == 2
    assert stats['n_cache_hits'] == 0

def test_invalidate_cache(single_dataset):
    """Test manual cache invalidation."""
    fwd_funcs, d_obs_list, model = single_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical',
        track_stats=True
    )

    # First evaluation
    _ = manager.objective(model)

    # Invalidate cache
    manager.invalidate_cache()

    # Same model but cache was invalidated
    _ = manager.objective(model)

    stats = manager.get_stats()
    assert stats['n_jacobian_evals'] == 2
    assert stats['n_cache_hits'] == 0

# ---------------------------
# Statistics tracking tests
# ---------------------------
def test_stats_tracking_disabled_raises():
    """Test that get_stats raises when tracking disabled."""
    fwd_funcs = [(fwd_single, {})]
    d_obs_list = [np.array([1.0, 2.0, 3.0])]

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical',
        track_stats=False
    )

    with pytest.raises(ValueError, match="tracking was not enabled"):
        manager.get_stats()

def test_stats_returns_copy(single_dataset):
    """Test that get_stats returns a copy."""
    fwd_funcs, d_obs_list, model = single_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical',
        track_stats=True
    )

    _ = manager.objective(model)
    stats1 = manager.get_stats()
    stats1['n_cache_hits'] = 999  # Mutate the returned copy

    stats2 = manager.get_stats()
    assert stats2['n_cache_hits'] == 0  # Original unchanged

# ---------------------------
# Input validation tests
# ---------------------------
def test_mismatched_fwd_funcs_and_data_raises():
    """Test that mismatched lengths raise ValueError."""
    fwd_funcs = [(fwd1, {}), (fwd2, {})]
    d_obs_list = [np.array([1.0, 2.0])]  # Only one dataset

    with pytest.raises(ValueError, match="same length"):
        ReducedLikelihoodManager(
            fwd_funcs=fwd_funcs,
            d_obs_list=d_obs_list,
            jacobian_fn=jacobian_fn
        )

def test_mismatched_cases_list_raises():
    """Test that mismatched cases list raises ValueError."""
    fwd_funcs = [(fwd1, {}), (fwd2, {})]
    d_obs_list = [np.array([1.0, 2.0]), np.array([3.0])]

    with pytest.raises(ValueError, match="cases list length"):
        ReducedLikelihoodManager(
            fwd_funcs=fwd_funcs,
            d_obs_list=d_obs_list,
            jacobian_fn=jacobian_fn,
            cases=['spherical', 'spherical', 'diag']  # 3 cases for 2 datasets
        )

def test_mismatched_cd_refs_list_raises():
    """Test that mismatched Cd_refs list raises ValueError."""
    fwd_funcs = [(fwd1, {}), (fwd2, {})]
    d_obs_list = [np.array([1.0, 2.0]), np.array([3.0])]

    with pytest.raises(ValueError, match="Cd_refs list length"):
        ReducedLikelihoodManager(
            fwd_funcs=fwd_funcs,
            d_obs_list=d_obs_list,
            jacobian_fn=jacobian_fn,
            cases='scaled',
            Cd_refs=[np.eye(2)]  # Only one Cd_ref for 2 datasets
        )

def test_scaled_case_without_cd_ref_raises():
    """Test that scaled case without Cd_ref raises ValueError."""
    fwd_funcs = [(fwd_single, {})]
    d_obs_list = [np.array([1.0, 2.0, 3.0])]

    with pytest.raises(ValueError, match="requires a Cd_ref"):
        ReducedLikelihoodManager(
            fwd_funcs=fwd_funcs,
            d_obs_list=d_obs_list,
            jacobian_fn=jacobian_fn,
            cases='scaled',
            Cd_refs=None
        )

# ---------------------------
# copy_G parameter tests
# ---------------------------
def test_copy_G_true_copies_jacobian(single_dataset):
    """Test that copy_G=True copies the Jacobian."""
    fwd_funcs, d_obs_list, model = single_dataset

    # Track if Jacobian was called
    jacobian_results = []

    def tracking_jacobian(m, n_data, fwd, fwd_kwargs):
        G = jacobian_fn(m, n_data, fwd, fwd_kwargs)
        jacobian_results.append(G)
        return G

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=tracking_jacobian,
        cases='spherical',
        copy_G=True
    )

    _ = manager.objective(model)

    # Verify that stored G is not the same object as returned
    stored_G = manager.reduced_likelihoods[0].G
    returned_G = jacobian_results[0]
    assert stored_G is not returned_G

def test_copy_G_false_no_copy(single_dataset):
    """Test that copy_G=False does not copy the Jacobian."""
    fwd_funcs, d_obs_list, model = single_dataset

    jacobian_results = []

    def tracking_jacobian(m, n_data, fwd, fwd_kwargs):
        G = jacobian_fn(m, n_data, fwd, fwd_kwargs)
        jacobian_results.append(G)
        return G

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=tracking_jacobian,
        cases='spherical',
        copy_G=False
    )

    _ = manager.objective(model)

    # Verify that stored G is the same object as returned
    stored_G = manager.reduced_likelihoods[0].G
    returned_G = jacobian_results[0]
    assert stored_G is returned_G

# ---------------------------
# Combined objective consistency
# ---------------------------
def test_combined_objective_equals_sum_of_individuals(multi_dataset):
    """Test that combined objective equals sum of individual objectives."""
    fwd_funcs, d_obs_list, model = multi_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical'
    )

    combined_obj = manager.objective(model)

    # Compute individual objectives
    individual_sum = 0.0
    for rl in manager.reduced_likelihoods:
        individual_sum += -rl.log_likelihood(model)

    assert_allclose(combined_obj, individual_sum, rtol=1e-12)

def test_combined_gradient_equals_sum_of_individuals(multi_dataset):
    """Test that combined gradient equals sum of individual gradients."""
    fwd_funcs, d_obs_list, model = multi_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical'
    )

    combined_grad = manager.gradient(model)

    # Compute individual gradients
    individual_sum = np.zeros_like(combined_grad)
    for rl in manager.reduced_likelihoods:
        individual_sum += -rl.gradient(model)

    assert_allclose(combined_grad, individual_sum, rtol=1e-12)

def test_combined_hessian_equals_sum_of_individuals(multi_dataset):
    """Test that combined Hessian equals sum of individual Hessians."""
    fwd_funcs, d_obs_list, model = multi_dataset

    manager = ReducedLikelihoodManager(
        fwd_funcs=fwd_funcs,
        d_obs_list=d_obs_list,
        jacobian_fn=jacobian_fn,
        cases='spherical'
    )

    combined_hess = manager.hessian(model)

    # Compute individual Hessians
    individual_sum = np.zeros_like(combined_hess)
    for rl in manager.reduced_likelihoods:
        individual_sum += -rl.hessian(model)

    assert_allclose(combined_hess, individual_sum, rtol=1e-12)
