import numpy as np
import pytest
from scipy import sparse

# CoFIGaussianVI needs CHOLMOD; it is an optional extra (pip install cofi[gaussian-vi])
# because scikit-sparse is a compiled extension requiring SuiteSparse headers.
pytest.importorskip("sksparse.cholmod", reason="requires scikit-sparse (CHOLMOD)")

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
    clamped, clipped = CoFIGaussianVI._clamp_perturbation(samples, mu, max_pert=5.0, iteration=0, warmup=10)
    assert np.all(np.abs(clamped - mu) <= 0.5 + 1e-10)
    assert clipped is True

    # At iteration 10 (past warmup), threshold = full max_pert = 5.0
    clamped, clipped = CoFIGaussianVI._clamp_perturbation(samples, mu, max_pert=5.0, iteration=10, warmup=10)
    assert np.all(np.abs(clamped - mu) <= 5.0 + 1e-10)
    assert clipped is True

    # Samples already inside the threshold must report clipped=False, which is
    # what gates an iteration's ELBO into the convergence history.
    small = np.array([[0.1, -0.2, 0.05, 0.0, 0.3]])
    unclamped, clipped = CoFIGaussianVI._clamp_perturbation(small, mu, max_pert=5.0, iteration=10, warmup=10)
    assert clipped is False
    np.testing.assert_allclose(unclamped, small)


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
    H_floored = CoFIGaussianVI._enforce_diagonal_floor(H.copy(), 1e-4)
    assert H_floored.diagonal()[0] >= 1e-4
    assert H_floored.diagonal()[1] == 5.0


# ---------------------------------------------------------------------------
# Regression tests for fixes on the gaussian-vi branch
# ---------------------------------------------------------------------------


def _linear_problem(N=4, M=6, seed=0):
    """Small well-posed linear problem, returned as (problem, options)."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((M, N))
    d = G @ np.ones(N)
    p = BaseProblem()
    p.set_forward(lambda m: G @ m)
    p.set_jacobian(lambda m: sparse.csr_matrix(G))
    p.set_data(d)
    p.set_data_covariance_inv(sparse.eye(M).tocsc())
    p.set_initial_model(np.zeros(N))
    o = InversionOptions()
    o.set_tool("cofi.gaussian_vi")
    o.set_params(prior_precision=sparse.eye(N).tocsc(), verbose=False)
    return p, o


def test_flow_grad_z_matches_finite_difference():
    """Latent gradient must match d/dz of -log p(d,T(z)) - log|det J_T(z)|.

    The log-det half is identically zero at a=1, b=0, so this asserts at
    non-identity flow parameters where omitting it is a real error.
    """
    rng = np.random.default_rng(3)
    N = 5
    A = rng.standard_normal((N, N))
    Qp = sparse.csc_matrix(A @ A.T + N * np.eye(N))
    m_ref = rng.standard_normal(N)

    def neg_log_joint(m):
        dm = m - m_ref
        return 0.5 * dm @ (Qp @ dm)

    def grad_neg_log_joint(m):
        return Qp @ (m - m_ref)

    def objective(z, a, b):
        m = CoFIGaussianVI._flow_forward(z, a, b)
        return neg_log_joint(m) - CoFIGaussianVI._flow_log_det_jac(z, a, b)

    for a, b in [
        (np.ones(N), np.zeros(N)),                       # identity: term is 0
        (np.full(N, 1.4), np.full(N, 0.3)),              # moved
        (rng.uniform(0.5, 1.8, N), rng.normal(0, 0.4, N)),
    ]:
        z = rng.standard_normal(N) * 0.7
        g_m = grad_neg_log_joint(CoFIGaussianVI._flow_forward(z, a, b))
        analytic, _ = CoFIGaussianVI._flow_grad_z(g_m, z, a, b)

        fd = np.zeros(N)
        h = 1e-6
        for i in range(N):
            zp, zm = z.copy(), z.copy()
            zp[i] += h
            zm[i] -= h
            fd[i] = (objective(zp, a, b) - objective(zm, a, b)) / (2 * h)

        assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-6), (
            f"a={a}, b={b}: analytic={analytic}, fd={fd}"
        )


def test_flow_grad_z_omitting_logdet_is_detectably_wrong():
    """Guard the guard: the naive (log-det-free) gradient must fail the check."""
    rng = np.random.default_rng(11)
    N = 4
    a = np.full(N, 1.5)
    b = np.full(N, 0.4)
    z = rng.standard_normal(N)
    g_m = rng.standard_normal(N)
    correct, dTdz = CoFIGaussianVI._flow_grad_z(g_m, z, a, b)
    naive = g_m * dTdz
    assert not np.allclose(correct, naive, atol=1e-8)


def test_non_symmetric_prior_precision_rejected():
    """A non-symmetric precision must raise, not be silently symmetrised."""
    p, o = _linear_problem()
    D = sparse.diags([1.0, 1.0, 1.0, 1.0]).tolil()
    D[0, 1] = 5.0  # break symmetry
    o.set_params(prior_precision=D.tocsc(), verbose=False)
    with pytest.raises(ValueError, match="symmetric"):
        CoFIGaussianVI(p, o)


def test_non_symmetric_data_covariance_inv_rejected():
    p, o = _linear_problem()
    Cd = sparse.eye(6).tolil()
    Cd[0, 1] = 3.0
    p.set_data_covariance_inv(Cd.tocsc())
    with pytest.raises(ValueError, match="symmetric"):
        CoFIGaussianVI(p, o)()


def test_zero_line_search_steps_does_not_crash():
    """map_line_search_steps=0 used to raise UnboundLocalError on f_trial."""
    p, o = _linear_problem()
    o.set_params(
        prior_precision=sparse.eye(4).tocsc(),
        map_line_search_steps=0,
        map_num_iterations=2,
        num_iterations=2,
        verbose=True,
    )
    result = CoFIGaussianVI(p, o)()
    assert np.all(np.isfinite(result["model"]))


def test_zero_diagonal_floor_still_factorises():
    """diagonal_floor=0 must not collapse the ridge escalation to all zeros."""
    p, o = _linear_problem()
    o.set_params(
        prior_precision=sparse.eye(4).tocsc(),
        diagonal_floor=0.0,
        num_iterations=3,
        verbose=False,
    )
    result = CoFIGaussianVI(p, o)()
    assert np.all(np.isfinite(result["model"]))


def test_non_positive_flow_a_min_rejected():
    p, o = _linear_problem()
    o.set_params(prior_precision=sparse.eye(4).tocsc(), flow_a_min=0.0, verbose=False)
    with pytest.raises(ValueError, match="flow_a_min"):
        CoFIGaussianVI(p, o)


def test_clip_step_rejects_non_finite():
    """inf in the step must not be laundered into nan by the rescaling."""
    out = CoFIGaussianVI._clip_step(np.array([np.inf, 1.0, 2.0]), 5.0, np.ones(3))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_convergence_not_permanently_disabled_by_early_spike():
    """A single early high ELBO must not switch off early stopping forever.

    The spike sits inside the 2*patience window, so a running-prefix-max
    baseline would give improvement < 0 and refuse to converge; a windowed
    baseline sees a flat plateau and converges.
    """
    patience, rtol = 3, 1e-4
    history = [1.0, 1.0, 100.0] + [1.0] * 6  # spike within 2*patience of the tail
    assert CoFIGaussianVI._check_convergence(history, patience, rtol)


def test_convergence_still_false_while_improving():
    history = [float(i) for i in range(40)]
    assert not CoFIGaussianVI._check_convergence(history, 3, 1e-4)


def test_batched_sampling_matches_per_draw():
    """sample_deltas must agree with looped sample_delta on the same inputs."""
    from cofi.tools._cofi_gaussian_vi import _sparse_cholesky

    rng = np.random.default_rng(5)
    n_dim = 12
    B = rng.standard_normal((n_dim, n_dim))
    Omega = sparse.csc_matrix(B @ B.T + n_dim * np.eye(n_dim))
    factor = _sparse_cholesky(Omega)
    Z = rng.standard_normal((7, n_dim))
    batched = factor.sample_deltas(Z)
    looped = np.array([factor.sample_delta(z) for z in Z])
    assert np.allclose(batched, looped)


def test_sampler_covariance_matches_precision_inverse():
    """End-to-end check that draws really have covariance Omega^{-1}."""
    rng = np.random.default_rng(7)
    n_dim = 5
    B = rng.standard_normal((n_dim, n_dim))
    Omega = sparse.csc_matrix(B @ B.T + n_dim * np.eye(n_dim))
    sampler = VISampler(
        np.zeros(n_dim), Omega, random_state=np.random.default_rng(0)
    )
    draws = sampler.sample(40000)
    emp = np.cov(draws.T)
    target = np.linalg.inv(Omega.toarray())
    assert np.max(np.abs(emp - target)) / np.max(np.abs(target)) < 0.15


def test_solve_raises_on_rank_deficient_hessian():
    """A singular Gauss-Newton Hessian must raise, not be silently ridged."""
    G = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 0.0], [3.0, 6.0, 1.0]])  # col1 = 2*col0
    H = sparse.csc_matrix(G.T @ G)
    assert np.linalg.matrix_rank(H.toarray()) < H.shape[0]
    with pytest.raises(np.linalg.LinAlgError):
        CoFIGaussianVI._solve(H, np.array([1.0, 2.0, 3.0]))


def test_convergence_reachable_when_clamp_always_binds():
    """A posterior wider than max_perturbation must still be able to converge.

    Excluding every clamped iteration from the convergence view would leave
    it permanently empty here, silently disabling convergence_patience.
    """
    p, o = _linear_problem()
    o.set_params(
        prior_precision=sparse.eye(4).tocsc(),
        max_perturbation=1e-6,   # clamp binds on every iteration, forever
        perturbation_warmup=1,
        num_iterations=80,
        convergence_patience=5,
        convergence_rtol=1e-3,
        verbose=False,
        random_seed=0,
    )
    result = CoFIGaussianVI(p, o)()
    assert len(result["elbo_history"]) < 80, "convergence never fired under clamping"


def test_to_arviz_path_with_vi_sampler():
    """Inversion -> to_arviz must work for VISampler on arviz >= 1.0.

    arviz 1.0 moved the group mapping to from_dict's first positional
    argument; the pre-1.0 ``posterior=`` keyword form raises TypeError.
    """
    pytest.importorskip("arviz")
    p, o = _linear_problem()
    o.set_params(
        prior_precision=sparse.eye(4).tocsc(), num_iterations=5, verbose=False
    )
    res = Inversion(p, o).run()
    idata = res.to_arviz(num_samples=200)
    model = idata.posterior["model"]
    assert model.shape == (1, 200, 4)
    assert model.dims == ("chain", "draw", "model_dim_0")
