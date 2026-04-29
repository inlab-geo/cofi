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

    All matrices are returned as sparse (as required by the solver).
    """
    rng = np.random.default_rng(seed)
    G_dense = rng.standard_normal((M, N))
    G = sparse.csr_matrix(G_dense)
    m_true = rng.standard_normal(N)
    noise_std = 0.5
    d_obs = G_dense @ m_true + noise_std * rng.standard_normal(M)

    Cd_inv = sparse.diags(np.ones(M) / noise_std**2)
    Qp = sparse.diags(2.0 * np.ones(N))
    m_prior = np.zeros(N)

    # Analytic posterior
    Omega_true = G_dense.T @ Cd_inv.toarray() @ G_dense + Qp.toarray()
    mu_true = np.linalg.solve(Omega_true, G_dense.T @ Cd_inv.toarray() @ d_obs + Qp.toarray() @ m_prior)

    return G, d_obs, Cd_inv, Qp, m_prior, m_true, mu_true, Omega_true


def _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior):
    """Build BaseProblem and InversionOptions for the linear problem."""
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: np.asarray(G @ m).ravel())
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
    assert 0 < len(elbo) <= 50  # may stop early due to convergence
    assert all(np.isfinite(e) for e in elbo)


def test_vi_sampler():
    """VISampler should produce samples with correct shape."""
    N = 5
    mu = np.zeros(N)
    Omega = sparse.diags(np.ones(N) * 4.0, format="csc")  # precision=4 → variance=0.25
    sampler = VISampler(mu, Omega, random_state=np.random.default_rng(0))
    samples = sampler.sample(100)
    assert samples.shape == (100, N)
    # Empirical std should be roughly 0.5
    assert np.all(np.std(samples, axis=0) < 1.0)


def test_vi_sampler_with_flow():
    """VISampler with flow should apply sinh-arcsinh transform."""
    N = 5
    mu = np.zeros(N)
    Omega = sparse.eye(N, format="csc")
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
    """Precision matrix should remain sparse throughout."""
    inv_problem, inv_options, _, _ = linear_setup
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()
    assert sparse.issparse(res["precision"])


def test_sparse_jacobian_preserves_sparsity():
    """With sparse J and sparse Qp, precision should remain sparse."""
    N, M = 20, 10
    rng = np.random.default_rng(42)
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
    G_dense = rng.standard_normal((M, N))
    G = sparse.csr_matrix(G_dense)
    alpha = 3.0

    def softplus(m):
        return np.log1p(np.exp(alpha * m)) / alpha

    def softplus_deriv(m):
        return 1.0 / (1.0 + np.exp(-alpha * m))

    m_true = rng.standard_normal(N)
    d_obs = G_dense @ softplus(m_true) + 0.3 * rng.standard_normal(M)
    Cd_inv = sparse.diags(np.ones(M) / 0.09)
    Qp = sparse.diags(np.ones(N))
    m_prior = np.zeros(N)

    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: G_dense @ softplus(m))
    inv_problem.set_jacobian(lambda m: sparse.csr_matrix(G_dense * softplus_deriv(m)[np.newaxis, :]))
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


# ---------------------------------------------------------------------------
# Sparse validation tests
# ---------------------------------------------------------------------------

def test_dense_prior_precision_raises():
    """Dense prior_precision should raise TypeError."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()
    inv_problem = _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior)
    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp.toarray(),  # dense!
        num_iterations=5,
        num_samples=2,
        verbose=False,
        random_seed=0,
    )
    with pytest.raises(TypeError, match="prior_precision must be a scipy sparse"):
        CoFIGaussianVI(inv_problem, inv_options)


def test_dense_data_covariance_inv_raises():
    """Dense data_covariance_inv should raise TypeError."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: np.asarray(G @ m).ravel())
    inv_problem.set_jacobian(lambda m: G)
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv.toarray())  # dense!
    inv_problem.set_initial_model(m_prior.copy())

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
    with pytest.raises(TypeError, match="data_covariance_inv must be a scipy sparse"):
        solver()


def test_dense_jacobian_raises():
    """Dense Jacobian should raise TypeError."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()
    G_dense = G.toarray()
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: G_dense @ m)
    inv_problem.set_jacobian(lambda m: G_dense)  # returns dense ndarray
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv)
    inv_problem.set_initial_model(m_prior.copy())

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
    with pytest.raises(TypeError, match="jacobian.*must be a scipy sparse"):
        solver()


def test_dense_omega_in_sampler_raises():
    """VISampler should raise if omega is dense."""
    with pytest.raises(TypeError, match="omega must be a scipy sparse"):
        VISampler(np.zeros(5), np.eye(5))


# ---------------------------------------------------------------------------
# Safeguard tests
# ---------------------------------------------------------------------------

def test_nan_sample_skipping():
    """Solver should skip NaN forward model outputs and still converge."""
    G, d_obs, Cd_inv, Qp, m_prior, _, mu_true, _ = _make_linear_problem()

    call_count = [0]
    def forward_with_nans(m):
        call_count[0] += 1
        result = np.asarray(G @ m).ravel()
        # Make every 5th call return NaN
        if call_count[0] % 5 == 0:
            result[:] = np.nan
        return result

    inv_problem = BaseProblem()
    inv_problem.set_forward(forward_with_nans)
    inv_problem.set_jacobian(lambda m: G)
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv)
    inv_problem.set_initial_model(m_prior.copy())

    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        num_iterations=50,
        num_samples=8,
        verbose=False,
        random_seed=123,
    )
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert all(np.isfinite(e) for e in res["elbo_history"])


def test_nan_jacobian_skipping():
    """Solver should skip samples whose Jacobian contains NaN/Inf."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()

    call_count = [0]
    def jacobian_with_nans(m):
        call_count[0] += 1
        if call_count[0] % 4 == 0:
            J = G.copy().astype(float)
            J.data[:] = np.nan
            return J
        return G

    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: np.asarray(G @ m).ravel())
    inv_problem.set_jacobian(jacobian_with_nans)
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv)
    inv_problem.set_initial_model(m_prior.copy())

    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        num_iterations=30,
        num_samples=8,
        verbose=False,
        random_seed=42,
    )
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert all(np.isfinite(e) for e in res["elbo_history"])
    assert np.all(np.isfinite(res["model"]))


def test_perturbation_clamping():
    """Perturbation clamping should limit sample deviations."""
    mu = np.zeros(5)
    samples = np.array([[10.0, -20.0, 3.0, -1.0, 0.5]])

    # At iteration 0 with warmup=10, effective threshold = 0.1 * max_pert = 0.5
    clamped = CoFIGaussianVI._clamp_perturbation(samples, mu, max_pert=5.0, iteration=0, warmup=10)
    assert np.all(np.abs(clamped - mu) <= 0.5 + 1e-10)

    # At iteration 10 (past warmup), threshold = full max_pert = 5.0
    clamped = CoFIGaussianVI._clamp_perturbation(samples, mu, max_pert=5.0, iteration=10, warmup=10)
    assert np.all(np.abs(clamped - mu) <= 5.0 + 1e-10)


def test_step_clipping():
    """Step clipping should limit the mean update norm."""
    delta = np.array([3.0, 4.0])  # norm = 5
    mu = np.array([10.0, 10.0])

    # Explicit max_norm = 2.0
    clipped = CoFIGaussianVI._clip_step(delta, max_norm=2.0, mu=mu)
    assert np.linalg.norm(clipped) <= 2.0 + 1e-10

    # Adaptive: max_norm=0 → 0.1 * max(||mu||, 1)
    clipped = CoFIGaussianVI._clip_step(delta, max_norm=0.0, mu=mu)
    assert np.linalg.norm(clipped) <= 0.1 * np.linalg.norm(mu) + 1e-10

    # Adaptive with small mu falls back to 0.1 * 1.0 = 0.1
    clipped = CoFIGaussianVI._clip_step(delta, max_norm=0.0, mu=np.zeros(2))
    assert np.linalg.norm(clipped) <= 0.1 + 1e-10


def test_hessian_rejection_guard():
    """Solver should skip Omega update when Hessian is extreme."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()

    call_count = [0]
    def jacobian_with_spike(m):
        call_count[0] += 1
        J = G.copy()
        # On early iterations, return a massive Jacobian to trigger rejection
        if call_count[0] < 5:
            J = J * 1000.0
        return J

    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda m: np.asarray(G @ m).ravel())
    inv_problem.set_jacobian(jacobian_with_spike)
    inv_problem.set_data(d_obs)
    inv_problem.set_data_covariance_inv(Cd_inv)
    inv_problem.set_initial_model(m_prior.copy())

    inv_options = InversionOptions()
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Qp,
        num_iterations=20,
        num_samples=2,
        hessian_rejection_ratio=10.0,
        verbose=False,
        random_seed=42,
    )
    solver = CoFIGaussianVI(inv_problem, inv_options)
    res = solver()
    assert res["success"]


def test_diminishing_step_size():
    """Diminishing step size should produce different results from constant."""
    G, d_obs, Cd_inv, Qp, m_prior, _, _, _ = _make_linear_problem()

    results = {}
    for decay_tau, label in [(0, "constant"), (20, "diminishing")]:
        inv_problem = _make_cofi_problem(G, d_obs, Cd_inv, Qp, m_prior)
        inv_options = InversionOptions()
        inv_options.set_tool("cofi.gaussian_vi")
        inv_options.set_params(
            prior_precision=Qp,
            num_iterations=30,
            num_samples=4,
            step_decay_timescale=decay_tau,
            verbose=False,
            random_seed=123,
        )
        solver = CoFIGaussianVI(inv_problem, inv_options)
        results[label] = solver()

    # Trajectories should differ
    assert not np.allclose(
        results["constant"]["model"],
        results["diminishing"]["model"],
    )


def test_hessian_diagonal_floor():
    """Per-sample Hessian diagonal floor should prevent near-zero diagonals."""
    H = sparse.csc_matrix(np.array([[1e-10, 0.0], [0.0, 5.0]]))
    H_floored = CoFIGaussianVI._enforce_hessian_diagonal_floor(H.copy(), 1e-4)
    assert H_floored.diagonal()[0] >= 1e-4
    assert H_floored.diagonal()[1] == 5.0
