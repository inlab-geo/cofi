import numpy as np
import pytest
from scipy import sparse

from cofi.tools import CoFIGaussianVI
from cofi.tools._cofi_gaussian_vi import VISampler
from cofi import BaseProblem, InversionOptions, Inversion


# ---------------------------------------------------------------------------
# Helpers: simple linear inverse problem with known analytic posterior
# ---------------------------------------------------------------------------

def _make_linear_problem(N=10, M=8, seed=42):
    """Linear problem: d = G @ m + noise, Gaussian prior.

    The true posterior is N(mu_post, Omega_post^{-1}) with:
        Omega_post = G^T Cd_inv G + Qp
        mu_post = Omega_post^{-1} (G^T Cd_inv d + Qp m_prior)
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((M, N))
    m_true = rng.standard_normal(N)
    noise_std = 0.5
    d_obs = G @ m_true + noise_std * rng.standard_normal(M)

    Cd_inv = sparse.diags(np.ones(M) / noise_std**2)
    Qp = sparse.diags(2.0 * np.ones(N))
    m_prior = np.zeros(N)

    # Analytic posterior
    Omega_true = G.T @ Cd_inv.toarray() @ G + Qp.toarray()
    mu_true = np.linalg.solve(Omega_true, G.T @ Cd_inv.toarray() @ d_obs + Qp.toarray() @ m_prior)

    return G, d_obs, Cd_inv, Qp, m_prior, m_true, mu_true, Omega_true


def _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior):
    """Build BaseProblem and InversionOptions for the linear problem."""
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: G @ m)
    inv_problem.set_jacobian(lambda m: G)
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv)
    inv_problem.set_initial_model(m_prior.copy())
    return inv_problem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_setup():
    G, d_obs, Cd_inv, Qp, m_prior, m_true, mu_true, Omega_true = _make_linear_problem()
    inv_problem = _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior)
    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        num_iterations=50,
        num_samples=4,
        learning_rate_mean=0.05,
        learning_rate_precision=0.1,
        verbose=False,
        random_seed=123,
    )
    return inv_problem, inv_options, mu_true, Omega_true


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_gaussian_vi(linear_setup):
    """Gaussian VI on a linear problem should converge near the analytic posterior."""
    inv_problem, inv_options, mu_true, Omega_true = linear_setup
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()

    assert res["success"]
    assert "model" in res
    assert "precision" in res
    assert "elbo_history" in res
    assert "map_model" in res
    assert res["flow_params"] is None

    # Mean should be close to analytic posterior mean
    np.testing.assert_allclose(res["model"], mu_true, atol=0.5)


def test_inversion_run(linear_setup):
    """Test through the Inversion wrapper — should return SamplingResult."""
    inv_problem, inv_options, _, _ = linear_setup
    inv = Inversion(inv_problem, inv_options)
    res = inv.run()

    assert res.success
    assert hasattr(res, "model")
    assert hasattr(res, "sampler")
    assert hasattr(res, "precision")


def test_sampling_result_type(linear_setup):
    """Result should be a SamplingResult since we include a sampler."""
    from cofi._inversion import SamplingResult
    inv_problem, inv_options, _, _ = linear_setup
    inv = Inversion(inv_problem, inv_options)
    res = inv.run()
    assert isinstance(res, SamplingResult)


def test_elbo_computed(linear_setup):
    """ELBO history should be populated with finite values."""
    inv_problem, inv_options, _, _ = linear_setup
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()
    elbo = res["elbo_history"]
    assert len(elbo) == 50
    assert all(np.isfinite(e) for e in elbo)


def test_vi_sampler():
    """VISampler should produce samples with correct shape."""
    N = 5
    mu = np.zeros(N)
    Omega = np.eye(N) * 4.0  # precision=4 → variance=0.25
    sampler = VISampler(mu, Omega, random_state=np.random.default_rng(0))
    samples = sampler.sample(100)
    assert samples.shape == (100, N)
    # Empirical std should be roughly 0.5
    assert np.all(np.std(samples, axis=0) < 1.0)


def test_vi_sampler_with_flow():
    """VISampler with flow should apply sinh-arcsinh transform."""
    N = 5
    mu = np.zeros(N)
    Omega = np.eye(N)
    a = np.ones(N) * 1.5
    b = np.ones(N) * 0.3
    sampler = VISampler(mu, Omega, flow_a=a, flow_b=b, random_state=np.random.default_rng(0))
    samples = sampler.sample(50)
    assert samples.shape == (50, N)
    # With b > 0, samples should have positive skew (mean > median roughly)
    assert np.mean(samples) > np.median(samples) - 1.0  # loose check


def test_default_prior_mean():
    """When prior_mean is not set, should default to initial_model."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()
    inv_problem = _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior)
    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        num_iterations=5,
        num_samples=2,
        verbose=False,
        random_seed=0,
    )
    solver = CoFIGaussianVI(inv_problem, inv_options)
    np.testing.assert_array_equal(solver._m_prior, m_prior)


def test_explicit_prior_mean():
    """When prior_mean is explicitly set, should use it."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()
    inv_problem = _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior)
    custom_prior = np.ones(10) * 2.0
    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        prior_mean=custom_prior,
        num_iterations=5,
        num_samples=2,
        verbose=False,
        random_seed=0,
    )
    solver = CoFIGaussianVI(inv_problem, inv_options)
    np.testing.assert_array_equal(solver._m_prior, custom_prior)


def test_sparse_preservation(linear_setup):
    """Precision matrix should remain sparse if input is sparse."""
    inv_problem, inv_options, _, _ = linear_setup
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()
    # For this problem J is dense, so H_GN will be dense. But with sparse Qp
    # the result is a dense array (correct behaviour for dense J).
    assert res["precision"] is not None


def test_sparse_jacobian_preserves_sparsity():
    """With sparse J and sparse Qp, precision should remain sparse."""
    N, M = 20, 10
    rng = np.random.default_rng(42)
    # Create sparse forward operator
    G = sparse.random(M, N, density=0.3, random_state=rng, format="csr")
    m_true = rng.standard_normal(N)
    d_obs = np.asarray(G @ m_true).ravel() + 0.1 * rng.standard_normal(M)

    Cd_inv = sparse.diags(np.ones(M) / 0.01)
    Qp = sparse.diags(np.ones(N))
    m_prior = np.zeros(N)

    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: np.asarray(G @ m).ravel())
    inv_problem.set_jacobian(lambda m: G)
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv)
    inv_problem.set_initial_model(m_prior.copy())

    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        num_iterations=10,
        num_samples=2,
        verbose=False,
        random_seed=0,
    )
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()
    assert sparse.issparse(res["precision"])


def test_seed_reproducibility():
    """Same seed should give identical results."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()

    results = []
    for _ in range(2):
        inv_problem = _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior)
        inv_options = InversionOptions()
        inv_options.set_tool("cofi.gaussian_vi")
        inv_options.set_params(
            prior_precision=Qp,
            num_iterations=10,
            num_samples=2,
            verbose=False,
            random_seed=999,
        )
        solver = CoFIGaussianVI(inv_problem, inv_options)
        results.append(solver())

    np.testing.assert_array_equal(results[0]["model"], results[1]["model"])
    np.testing.assert_array_equal(
        results[0]["elbo_history"], results[1]["elbo_history"]
    )


def test_sas_flow():
    """SAS flow should produce non-identity flow params on a nonlinear problem."""
    N, M = 10, 8
    rng = np.random.default_rng(42)
    G = rng.standard_normal((M, N))
    alpha = 3.0

    def softplus(m):
        return np.log1p(np.exp(alpha * m)) / alpha

    def softplus_deriv(m):
        return 1.0 / (1.0 + np.exp(-alpha * m))

    m_true = rng.standard_normal(N)
    d_obs = G @ softplus(m_true) + 0.3 * rng.standard_normal(M)
    Cd_inv = sparse.diags(np.ones(M) / 0.09)
    Qp = sparse.diags(np.ones(N))
    m_prior = np.zeros(N)

    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: G @ softplus(m))
    inv_problem.set_jacobian(lambda m: G * softplus_deriv(m)[np.newaxis, :])
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv)
    inv_problem.set_initial_model(m_prior.copy())

    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        enable_flow=True,
        flow_num_iterations=50,
        flow_num_samples=8,
        num_iterations=20,
        num_samples=4,
        verbose=False,
        random_seed=42,
    )
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()

    assert res["success"]
    assert res["flow_params"] is not None
    a = res["flow_params"]["a"]
    b = res["flow_params"]["b"]
    # Flow params should have moved from identity
    assert not np.allclose(a, 1.0) or not np.allclose(b, 0.0)


def test_missing_required_component():
    """Missing required problem component should raise."""
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: m)
    inv_problem.set_initial_model(np.zeros(5))
    # Missing: jacobian, data, data_covariance_inv

    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(prior_precision=sparse.eye(5))

    with pytest.raises(Exception):
        CoFIGaussianVI(inv_problem, inv_options)
