"""Natural-gradient Gaussian variational inference for sparse Bayesian
inverse problems, with an optional per-coordinate sinh-arcsinh normalising
flow.

This module provides:

- :class:`CoFIGaussianVI` -- the inference tool, registered with CoFI under
  ``cofi.gaussian_vi`` (variational-inference family).
- :class:`VISampler` -- a lightweight on-demand sampler over the fitted
  posterior, suitable for downstream use (e.g. arviz).

External dependency: scikit-sparse >= 0.5 (CHOLMOD) for the sparse Cholesky
factorisation, triangular solves, and log-determinant.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from . import BaseInferenceTool

try:
    from sksparse import cholmod as _cholmod

    _HAS_CHOLMOD = True
except ImportError:
    _HAS_CHOLMOD = False


def _require_sparse(M, name):
    """Raise TypeError if M is not a scipy sparse matrix."""
    if not sparse.issparse(M):
        raise TypeError(
            f"{name} must be a scipy sparse matrix, got {type(M).__name__}"
        )


def _require_cholmod():
    """Raise ImportError if CHOLMOD is not available.

    Requires scikit-sparse >= 0.5, which exposes ``cho_factor`` returning a
    ``CholeskyFactor`` with C-level ``solve``/``logdet`` methods.
    """
    if not _HAS_CHOLMOD:
        raise ImportError(
            "CoFIGaussianVI requires scikit-sparse>=0.5 (CHOLMOD). "
            "Install with: pip install scikit-sparse"
        )
    if not hasattr(_cholmod, "cho_factor"):
        raise ImportError(
            "CoFIGaussianVI requires scikit-sparse>=0.5; the installed "
            "version does not expose `cho_factor`."
        )


class _CholmodFactor:
    """Thin wrapper around ``sksparse.cholmod.CholeskyFactor``.

    Holds an LL^T factorisation with permutation P such that
    ``P A P^T = L L^T``.  All triangular and full-system solves use
    CHOLMOD's C implementation rather than ``scipy.sparse.linalg``.
    """

    __slots__ = ("_factor", "_perm_inv")

    def __init__(self, factor):
        self._factor = factor
        perm = factor.get_perm()
        self._perm_inv = np.empty_like(perm)
        self._perm_inv[perm] = np.arange(len(perm))

    def logdet(self):
        """log|A| via CHOLMOD."""
        return self._factor.logdet()

    def solve_A(self, b):
        """Solve A x = b (handles permutation internally)."""
        return self._factor.solve(b, system="A")

    def sample_delta(self, z):
        """Given z ~ N(0,I), return v ~ N(0, A^{-1}).

        For the LL^T factorisation P A P^T = L L^T:
            A^{-1} = P^T L^{-T} L^{-1} P
            v = P^T (L^{-T} z) gives Cov(v) = A^{-1} when z ~ N(0,I)
        """
        y = self._factor.solve(z, system="Lt")
        return y[self._perm_inv]


def _sparse_cholesky(Omega):
    """Compute sparse LL^T Cholesky via CHOLMOD and return a _CholmodFactor."""
    factor = _cholmod.cho_factor(Omega.tocsc())
    if not factor.is_ll:
        factor.change_factor(to_ll=True)
    return _CholmodFactor(factor)


class VISampler:
    r"""On-demand sampler for a fitted Gaussian (or sinh-arcsinh) VI posterior.

    Holds the variational mean :math:`\mu`, precision :math:`\Omega`, and
    optional per-coordinate sinh-arcsinh flow parameters.  Caches a CHOLMOD
    Cholesky factorisation of :math:`\Omega` so subsequent ``sample(n)``
    calls amortise the factorisation cost.

    The samples are drawn as

    .. math::

        z &\sim \mathcal{N}(\mu, \Omega^{-1}) \\
        m_i &= \begin{cases}
            z_i & \text{(Gaussian only)} \\
            \sinh\!\big(a_i\,\mathrm{arcsinh}(z_i) + b_i\big) & \text{(flow)}
        \end{cases}

    using :math:`v = P^\top L^{-\top} \xi` with :math:`\xi \sim \mathcal{N}(0, I)`
    and :math:`P\,\Omega\,P^\top = L L^\top`.

    Parameters
    ----------
    mu : numpy.ndarray, shape (N,)
        Posterior mean.
    omega : scipy.sparse matrix, shape (N, N)
        Posterior precision :math:`\Omega = \Sigma^{-1}`.  Must be sparse,
        symmetric, and positive-definite.
    flow_a : numpy.ndarray, shape (N,), optional
        Sinh-arcsinh tail-weight parameters.  ``None`` (default) disables
        the flow and produces purely Gaussian samples.
    flow_b : numpy.ndarray, shape (N,), optional
        Sinh-arcsinh skewness parameters.  ``None`` (default) disables the
        flow.
    random_state : numpy.random.Generator, optional
        Generator for the i.i.d. standard normals.  Defaults to
        :func:`numpy.random.default_rng()`.

    Raises
    ------
    TypeError
        If ``omega`` is not a scipy sparse matrix.
    ImportError
        If scikit-sparse >= 0.5 (CHOLMOD) is not available.

    Examples
    --------
    >>> from scipy import sparse
    >>> import numpy as np
    >>> sampler = VISampler(
    ...     mu=np.zeros(5),
    ...     omega=sparse.eye(5, format="csc") * 4.0,  # variance = 0.25
    ...     random_state=np.random.default_rng(0),
    ... )
    >>> samples = sampler.sample(1000)
    >>> samples.shape
    (1000, 5)
    """

    def __init__(self, mu, omega, flow_a=None, flow_b=None, random_state=None):
        _require_sparse(omega, "omega")
        _require_cholmod()
        self.mu = mu
        self.omega = omega
        self.flow_a = flow_a
        self.flow_b = flow_b
        self._rng = (
            random_state if random_state is not None else np.random.default_rng()
        )
        self._factor = _sparse_cholesky(omega)

    def sample(self, n=1):
        r"""Draw ``n`` independent samples from the approximate posterior.

        For each draw, a Gaussian latent
        :math:`z \sim \mathcal{N}(\mu, \Omega^{-1})` is generated via the
        cached Cholesky factor; if flow parameters are set, the elementwise
        sinh-arcsinh transform :math:`m_i = \sinh(a_i\,\mathrm{arcsinh}(z_i) + b_i)`
        is applied.

        Parameters
        ----------
        n : int, default 1
            Number of independent samples to draw.

        Returns
        -------
        samples : numpy.ndarray, shape (n, N)
            One row per draw.
        """
        N = len(self.mu)
        Z = self._rng.standard_normal((n, N))

        samples = np.empty((n, N))
        for i in range(n):
            samples[i] = self.mu + self._factor.sample_delta(Z[i])

        if self.flow_a is not None and self.flow_b is not None:
            samples = np.sinh(
                self.flow_a[np.newaxis, :] * np.arcsinh(samples)
                + self.flow_b[np.newaxis, :]
            )
        return samples


class CoFIGaussianVI(BaseInferenceTool):
    r"""Natural-gradient Gaussian variational inference with optional
    sinh-arcsinh normalising flow.

    Approximates the posterior :math:`p(m \mid d) \propto p(d \mid m)\,p(m)`
    of a non-linear, large-scale Bayesian inverse problem with a parametric
    distribution :math:`q(m)`, by maximising the Evidence Lower Bound (ELBO).
    The implementation is **precision-first** (parameterised by the precision
    matrix :math:`\Omega = \Sigma^{-1}` rather than :math:`\Sigma`) so that
    sparsity from the prior precision and Jacobian is preserved end-to-end,
    and uses **natural-gradient updates** for reparameterisation-invariant
    optimisation.

    The variational family is, depending on ``enable_flow``:

    - ``enable_flow=False``: :math:`q(m) = \mathcal{N}(m \mid \mu, \Omega^{-1})`.
    - ``enable_flow=True``: :math:`q(m) = q_z(z) \prod_i \delta(m_i - T_i(z_i))`
      with :math:`z \sim \mathcal{N}(\mu, \Omega^{-1})` and
      :math:`T_i(z_i) = \sinh(a_i\,\mathrm{arcsinh}(z_i) + b_i)` (a per-coordinate
      sinh-arcsinh transform; see Jones & Pewsey 2009).  This adds tunable
      skewness and kurtosis on top of the Gaussian.

    The likelihood and prior are assumed Gaussian:

    .. math::

        p(d \mid m) \propto \exp\!\big(-\tfrac{1}{2}(f(m) - d)^\top C_d^{-1}(f(m) - d)\big),
        \quad
        p(m) = \mathcal{N}(m \mid m_\text{prior}, Q_p^{-1}),

    where :math:`f` is the (nonlinear) forward map, :math:`C_d^{-1}` is the
    data precision (``data_covariance_inv``), and :math:`Q_p` is the prior
    precision (``prior_precision``).  All three matrices, plus the Jacobian
    returned by ``inv_problem.jacobian()``, **must be scipy sparse**.

    Algorithm
    ---------
    The solver runs in three phases:

    1. **MAP initialisation** (``map_*`` options).  Gauss-Newton iteration
       with Armijo backtracking line search to find
       :math:`m_\text{MAP} = \arg\max_m \log p(d \mid m) + \log p(m)`.
    2. **Gaussian VI** (``num_*``, ``learning_rate_*`` options).  Initialise
       :math:`\mu = m_\text{MAP}`, :math:`\Omega = J^\top C_d^{-1} J + Q_p`
       at MAP, then iterate stochastic natural-gradient updates:

       .. math::

           \Omega_{t+1} &= (1 - \rho_\Omega)\Omega_t
                          + \rho_\Omega\, \mathbb{E}_{q_t}\!\big[J^\top C_d^{-1} J + Q_p\big] \\
           \mu_{t+1}    &= \mu_t
                          - \rho_\mu\, \Omega_t^{-1}\, \mathbb{E}_{q_t}\!\big[\nabla \mathcal{L}(m)\big]

       where :math:`\mathcal{L}(m) = -\log p(d, m)` and the expectations are
       Monte-Carlo estimated from ``num_samples`` draws of :math:`q_t`.
    3. **Sinh-arcsinh flow VI** (``flow_*`` options, only if
       ``enable_flow=True``).  Warm-started from Phase 2, simultaneously
       updates :math:`(\mu, \Omega)` (natural-gradient) and per-coordinate
       flow parameters :math:`(a, b)` (Adam ascent on the ELBO).

    All sparse Cholesky factorisations and triangular/full solves use
    CHOLMOD via ``sksparse.cholmod``.  When :math:`\Omega` is detected as not
    PD, ridge regularisation :math:`\Omega + \lambda I` is added with
    geometrically increasing :math:`\lambda` until factorisation succeeds.

    .. note::

        The textbook Khan & Lin (2017) update solves the natural-gradient mean
        step with the *updated* precision :math:`\Omega_{t+1}^{-1}`.  To avoid
        a second factorisation per iteration, this implementation uses the
        pre-update :math:`\Omega_t^{-1}` (a common practical simplification);
        see Khan & Rue (2023, Sec. 4) for discussion.

    When to use this solver
    -----------------------
    Suitable when:

    - The model dimension :math:`N` is too large for full-covariance VI or
      MCMC, but the prior precision :math:`Q_p` and Jacobian
      :math:`J(m) = \partial f / \partial m` are sparse.
    - You have a Gaussian likelihood and prior; the forward map :math:`f` may
      be nonlinear but should be differentiable.
    - You need posterior uncertainty (not just a MAP point estimate).

    Less suitable when:

    - The posterior is strongly multimodal (VI tends to fit one mode).
    - The Jacobian is dense; in that case Frobenius cost dominates and a
      dense VI implementation may be faster.

    Required problem components
    ---------------------------
    - ``forward(m) -> (M,)`` : Forward map.  Must accept a 1-D array of
      length :math:`N` and return a 1-D array of length :math:`M`.
    - ``jacobian(m) -> (M, N) sparse`` : Jacobian of ``forward`` at ``m``.
      Must return a scipy sparse matrix.
    - ``data`` : 1-D array of observations, length :math:`M`.
    - ``data_covariance_inv`` : :math:`(M, M)` scipy sparse matrix
      :math:`C_d^{-1}`.
    - ``initial_model`` : 1-D array of length :math:`N`, used for the MAP
      starting point and (by default) as the prior mean.

    Required options
    ----------------
    - ``prior_precision`` : :math:`(N, N)` scipy sparse matrix :math:`Q_p`.

    Optional options
    ----------------
    Prior and initialisation:

    - ``prior_mean`` (None) : Prior mean :math:`m_\text{prior}`.  Defaults to
      ``initial_model`` when ``None``.

    Phase 2 (Gaussian VI) loop:

    - ``num_iterations`` (100) : Maximum natural-gradient iterations.
    - ``num_samples`` (8) : Monte-Carlo samples drawn from
      :math:`q_t` per iteration.
    - ``learning_rate_mean`` (0.02) : :math:`\rho_\mu`, step size for the
      mean update.
    - ``learning_rate_precision`` (0.05) : :math:`\rho_\Omega`, mixing rate
      for the precision update.

    Numerical safeguards (see private static methods for details):

    - ``diagonal_floor`` (1e-4) : Minimum diagonal entry enforced on
      :math:`\Omega` after each update.
    - ``hessian_diagonal_floor`` (1e-4) : Minimum diagonal entry enforced
      on per-sample Gauss-Newton Hessians before averaging.
    - ``max_perturbation`` (10.0) : Coordinate-wise cap (in model units) on
      sample deviations from :math:`\mu` to prevent extreme outliers.  Set
      to ``0`` to disable.
    - ``perturbation_warmup`` (10) : Number of iterations over which the
      perturbation cap ramps from 10% to 100%.
    - ``max_step_norm`` (0.0) : Trust-region cap on :math:`\|\delta\mu\|_2`.
      If ``<= 0``, an adaptive cap of :math:`0.1 \cdot \max(\|\mu\|, 1)` is
      used.
    - ``step_decay_timescale`` (0) : If ``> 0``, scales :math:`\rho_\mu` by
      :math:`1 / (1 + t/\tau)` for diminishing step sizes.
    - ``hessian_rejection_ratio`` (10.0) : Skip the :math:`\Omega` update
      when :math:`\|H_\text{avg}\|_F / \|\Omega\|_F` exceeds this ratio
      (rejects rare extreme-Hessian samples).

    Convergence:

    - ``convergence_patience`` (10) : Stop early if the ELBO has not
      improved by ``convergence_rtol`` over the last ``patience``
      iterations.  Set ``0`` to disable.
    - ``convergence_rtol`` (1e-4) : Relative tolerance for the patience
      check.

    Phase 1 (MAP) options:

    - ``map_num_iterations`` (30), ``map_convergence_tol`` (1e-8),
      ``map_line_search_steps`` (10), ``map_line_search_shrink`` (0.5).

    Phase 3 (sinh-arcsinh flow) options (used iff ``enable_flow=True``):

    - ``enable_flow`` (False) : Activate Phase 3.
    - ``flow_num_iterations`` (300), ``flow_num_samples`` (32).
    - ``flow_learning_rate`` (0.003) : Adam learning rate for ``(a, b)``.
    - ``flow_adam_beta1`` (0.9), ``flow_adam_beta2`` (0.999).
    - ``flow_a_min`` (0.1) : Lower clip on ``a`` to keep the flow invertible
      and stable.

    Other:

    - ``verbose`` (True) : Print per-iteration progress.
    - ``random_seed`` (None) : Seed for the internal
      :class:`numpy.random.Generator`.

    Returns (from :meth:`__call__`)
    -------------------------------
    dict with keys:

    - ``model`` : (N,) posterior mean :math:`\mu`.
    - ``precision`` : (N, N) sparse posterior precision :math:`\Omega`.
    - ``sampler`` : :class:`VISampler` for drawing posterior samples on
      demand.  When converted via
      :meth:`cofi.SamplingResult.to_arviz`, draws are wrapped into an
      :class:`arviz.InferenceData` object.
    - ``elbo_history`` : list of ELBO values per iteration (Phase 2, or
      Phase 3 if ``enable_flow=True``).
    - ``num_iterations`` : number of completed iterations.
    - ``map_model`` : (N,) MAP estimate from Phase 1.
    - ``flow_params`` : ``{"a": ..., "b": ...}`` if ``enable_flow=True``,
      else ``None``.
    - ``success`` : True if the run completed.

    Examples
    --------
    Linear inverse problem with Gaussian prior::

        >>> import numpy as np
        >>> from scipy import sparse
        >>> from cofi import BaseProblem, InversionOptions, Inversion
        >>>
        >>> rng = np.random.default_rng(0)
        >>> N, M = 20, 30
        >>> G = sparse.csr_matrix(rng.standard_normal((M, N)))
        >>> m_true = rng.standard_normal(N)
        >>> d_obs = G @ m_true + 0.3 * rng.standard_normal(M)
        >>>
        >>> problem = BaseProblem()
        >>> problem.set_forward(lambda m: np.asarray(G @ m).ravel())
        >>> problem.set_jacobian(lambda m: G)
        >>> problem.set_data(d_obs)
        >>> problem.set_data_covariance_inv(sparse.eye(M) / 0.09)
        >>> problem.set_initial_model(np.zeros(N))
        >>>
        >>> options = InversionOptions()
        >>> options.set_tool("cofi.gaussian_vi")
        >>> options.set_params(
        ...     prior_precision=sparse.eye(N),
        ...     num_iterations=200,
        ...     learning_rate_mean=0.1,
        ...     learning_rate_precision=0.2,
        ...     verbose=False,
        ...     random_seed=42,
        ... )
        >>>
        >>> result = Inversion(problem, options).run()
        >>> mu_hat = result.model              # posterior mean
        >>> samples = result.sampler.sample(1000)  # 1000 posterior draws

    Adding the sinh-arcsinh flow for skewed/heavy-tailed posteriors::

        >>> options.set_params(
        ...     enable_flow=True,
        ...     flow_num_iterations=300,
        ...     flow_num_samples=32,
        ... )

    Dependencies
    ------------
    Requires scikit-sparse >= 0.5 (CHOLMOD); install with
    ``pip install scikit-sparse``.

    See Also
    --------
    VISampler : Lightweight wrapper for drawing samples from a fitted posterior.

    References
    ----------
    .. [1] M. E. Khan, W. Lin (2017). "Conjugate-Computation Variational
       Inference: Converting Variational Inference in Non-Conjugate Models
       to Inferences in Conjugate Models." AISTATS.
       https://arxiv.org/abs/1703.04265
    .. [2] M. E. Khan, H. Rue (2023). "The Bayesian Learning Rule."
       JMLR. https://arxiv.org/abs/2107.04562
    .. [3] M. C. Jones, A. Pewsey (2009). "Sinh-arcsinh distributions."
       Biometrika 96(4):761-780.
    """

    documentation_links = [
        "https://arxiv.org/abs/1703.04265",  # Khan & Lin 2017
        "https://arxiv.org/abs/2107.04562",  # Khan & Rue 2023
    ]
    short_description = (
        "Natural-gradient Gaussian VI with optional sinh-arcsinh normalising flow"
    )

    @classmethod
    def required_in_problem(cls) -> set:
        return {"forward", "jacobian", "data", "data_covariance_inv", "initial_model"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {"data_covariance": None}

    @classmethod
    def required_in_options(cls) -> set:
        return {"prior_precision"}

    @classmethod
    def optional_in_options(cls) -> dict:
        return {
            "prior_mean": None,
            "num_iterations": 100,
            "num_samples": 8,
            "learning_rate_mean": 0.02,
            "learning_rate_precision": 0.05,
            "diagonal_floor": 1e-4,
            "hessian_diagonal_floor": 1e-4,
            "max_perturbation": 10.0,
            "perturbation_warmup": 10,
            "max_step_norm": 0.0,
            "step_decay_timescale": 0,
            "hessian_rejection_ratio": 10.0,
            "enable_flow": False,
            "flow_num_iterations": 300,
            "flow_num_samples": 32,
            "flow_learning_rate": 0.003,
            "flow_adam_beta1": 0.9,
            "flow_adam_beta2": 0.999,
            "flow_a_min": 0.1,
            "map_num_iterations": 30,
            "map_convergence_tol": 1e-8,
            "map_line_search_steps": 10,
            "map_line_search_shrink": 0.5,
            "convergence_patience": 10,
            "convergence_rtol": 1e-4,
            "verbose": True,
            "random_seed": None,
        }

    def __init__(self, inv_problem, inv_options):
        """Validate inputs and stash configuration; no expensive work.

        Heavy work (MAP, VI loops) is deferred to :meth:`__call__` so the
        same instance can be re-run with different RNG states or inspected
        before launching.

        Raises
        ------
        ImportError
            If scikit-sparse >= 0.5 (CHOLMOD) is unavailable.
        TypeError
            If ``prior_precision`` is not a scipy sparse matrix.
        """
        super().__init__(inv_problem, inv_options)
        _require_cholmod()

        self._Qp = self._params["prior_precision"]
        _require_sparse(self._Qp, "prior_precision")
        self._Qp = self._Qp.tocsc()

        self._m_prior = (
            self._params["prior_mean"]
            if self._params["prior_mean"] is not None
            else np.asarray(self.inv_problem.initial_model).copy()
        )
        seed = self._params["random_seed"]
        self._rng = np.random.default_rng(seed)

    def __call__(self) -> dict:
        """Run the three-phase VI procedure and return the fitted posterior.

        Executes Phase 1 (MAP), Phase 2 (Gaussian VI), and — if
        ``enable_flow`` was set — Phase 3 (sinh-arcsinh flow VI).  See the
        class docstring for the algorithm and the structure of the returned
        dictionary.

        Returns
        -------
        dict
            See "Returns" section in the :class:`CoFIGaussianVI` class
            docstring for the full key list.

        Raises
        ------
        TypeError
            If ``data_covariance_inv`` or any ``jacobian(m)`` returned at
            runtime is not a scipy sparse matrix.
        numpy.linalg.LinAlgError
            If the precision matrix cannot be made positive-definite even
            after ridge regularisation.
        """
        m0 = np.asarray(self.inv_problem.initial_model).copy()
        Cd_inv = self.inv_problem.data_covariance_inv
        _require_sparse(Cd_inv, "data_covariance_inv")
        Cd_inv = Cd_inv.tocsc()
        d_obs = np.asarray(self.inv_problem.data)

        # Phase 1: MAP initialisation
        m_map, H_map = self._run_map(m0, d_obs, Cd_inv)

        # Phase 2: Gaussian VI
        mu, Omega, elbo_history = self._run_gaussian_vi(m_map, H_map, d_obs, Cd_inv)

        flow_params = None

        # Phase 3: Optional SAS flow (warm-started from Phase 2)
        if self._params["enable_flow"]:
            mu, Omega, a_flow, b_flow, elbo_sa = self._run_sas_vi(
                mu, Omega, d_obs, Cd_inv
            )
            flow_params = {"a": a_flow, "b": b_flow}
            elbo_history = elbo_sa

        sampler = VISampler(
            mu,
            Omega,
            flow_a=flow_params["a"] if flow_params else None,
            flow_b=flow_params["b"] if flow_params else None,
            random_state=np.random.default_rng(self._params["random_seed"]),
        )

        return {
            "model": mu,
            "success": True,
            "sampler": sampler,
            "precision": Omega,
            "elbo_history": elbo_history,
            "num_iterations": len(elbo_history),
            "map_model": m_map,
            "flow_params": flow_params,
        }

    # ------------------------------------------------------------------
    # Sparse Cholesky helpers (CHOLMOD)
    # ------------------------------------------------------------------

    @staticmethod
    def _factor_and_logdet(Omega, diag_floor):
        """Factor sparse Omega via CHOLMOD with PD enforcement.

        Returns (_CholmodFactor, logdet, Omega).  Omega may be
        ridge-corrected if the original was not PD.
        """
        N = Omega.shape[0]
        for attempt in range(10):
            ridge = diag_floor * (10**attempt) if attempt > 0 else 0.0
            try:
                test = (
                    Omega + ridge * sparse.eye(N, format="csc")
                    if ridge > 0
                    else Omega
                )
                factor = _sparse_cholesky(test.tocsc())
                return factor, factor.logdet(), test if ridge > 0 else Omega
            except Exception:
                continue
        raise np.linalg.LinAlgError(
            "Precision matrix not positive definite after ridge correction"
        )

    def _sample_from_factor(self, mu, factor, n):
        """Draw n samples from N(mu, Omega^{-1}) using CHOLMOD factor."""
        N = len(mu)
        Z = self._rng.standard_normal((n, N))
        samples = np.empty((n, N))
        for i in range(n):
            samples[i] = mu + factor.sample_delta(Z[i])
        return samples

    @staticmethod
    def _solve_from_factor(factor, b):
        """Solve Omega x = b using CHOLMOD factor."""
        return factor.solve_A(b)

    # ------------------------------------------------------------------
    # Linear algebra helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _solve(Omega, b):
        """Solve Omega x = b via sparse direct solve."""
        return splinalg.spsolve(Omega.tocsc(), b)

    @staticmethod
    def _enforce_diagonal_floor(Omega, floor):
        """Clamp diagonal entries of sparse Omega to at least floor."""
        Omega = Omega.tocsc()
        Omega.setdiag(np.maximum(Omega.diagonal(), floor))
        return Omega

    @staticmethod
    def _enforce_hessian_diagonal_floor(H, floor):
        """Clamp diagonal entries of a per-sample sparse Hessian to at least floor."""
        H = H.tocsc()
        H.setdiag(np.maximum(H.diagonal(), floor))
        return H

    @staticmethod
    def _clamp_perturbation(samples, mu, max_pert, iteration, warmup):
        """Clamp sample deviations from mu to max_pert with warmup ramp.

        During warmup the effective threshold ramps linearly from 10% to 100%.

        ``max_pert`` is interpreted in the model's native units, applied
        coordinate-wise. For models whose components span very different
        scales, scale them externally (or set ``max_pert=0`` to disable).
        """
        if max_pert <= 0:
            return samples
        frac = min(1.0, 0.1 + 0.9 * iteration / max(warmup, 1))
        threshold = frac * max_pert
        delta = samples - mu[np.newaxis, :]
        delta = np.clip(delta, -threshold, threshold)
        return mu[np.newaxis, :] + delta

    @staticmethod
    def _clip_step(delta_mu, max_norm, mu):
        """Clip the mean update step to max_norm.

        If ``max_norm <= 0``, use an adaptive trust region of
        ``0.1 * max(||mu||, 1)``, so a single update can move the mean by at
        most ~10% of its current magnitude. Set ``max_norm`` explicitly to
        override.
        """
        if max_norm <= 0:
            max_norm = 0.1 * max(np.linalg.norm(mu), 1.0)
        norm = np.linalg.norm(delta_mu)
        if norm > max_norm:
            delta_mu = delta_mu * (max_norm / norm)
        return delta_mu

    # ------------------------------------------------------------------
    # Jacobian validation
    # ------------------------------------------------------------------

    def _get_jacobian(self, m):
        """Evaluate Jacobian and validate it is sparse."""
        J = self.inv_problem.jacobian(m)
        _require_sparse(J, "jacobian (returned by problem.jacobian())")
        return J.tocsr()

    # ------------------------------------------------------------------
    # ELBO and convergence
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_elbo(log_joints, logdet_Omega, N):
        """ELBO = E_q[log p(d,m)] + H[q].

        The Gaussian entropy is H[q] = 0.5*N*(1+log(2*pi)) - 0.5*log|Omega|.
        Normalisation constants from the likelihood and prior are omitted
        (they are independent of variational parameters).
        """
        entropy = 0.5 * N * (1 + np.log(2 * np.pi)) - 0.5 * logdet_Omega
        return np.mean(log_joints) + entropy

    @staticmethod
    def _check_convergence(elbo_history, patience, rtol):
        """True if ELBO has not improved significantly in patience iterations."""
        if patience <= 0 or len(elbo_history) < patience + 1:
            return False
        window = elbo_history[-(patience + 1) :]
        if not all(np.isfinite(e) for e in window):
            return False
        baseline = max(elbo_history[: -patience])
        best_recent = max(elbo_history[-patience:])
        improvement = best_recent - baseline
        if improvement < 0:
            return False
        scale = max(abs(baseline), 1.0)
        return improvement / scale < rtol

    # ------------------------------------------------------------------
    # Phase 1: MAP via Gauss-Newton with backtracking line search
    # ------------------------------------------------------------------

    def _run_map(self, m0, d_obs, Cd_inv):
        r"""Phase 1: Gauss-Newton MAP estimation.

        Solves :math:`m_\text{MAP} = \arg\min_m \tfrac{1}{2} (f(m) - d)^\top
        C_d^{-1} (f(m) - d) + \tfrac{1}{2}(m - m_\text{prior})^\top Q_p
        (m - m_\text{prior})` via Newton steps on the Gauss-Newton Hessian
        :math:`H = J^\top C_d^{-1} J + Q_p`, with Armijo backtracking line
        search on the cost.

        Parameters
        ----------
        m0 : numpy.ndarray, shape (N,)
            Initial model.
        d_obs : numpy.ndarray, shape (M,)
            Observations.
        Cd_inv : scipy.sparse, shape (M, M)
            Data precision.

        Returns
        -------
        m : numpy.ndarray, shape (N,)
            MAP estimate.
        H_map : scipy.sparse.csc_matrix, shape (N, N)
            Gauss-Newton Hessian at MAP, used to initialise :math:`\Omega`
            in Phase 2.
        """
        m = m0.copy()
        max_iter = self._params["map_num_iterations"]
        tol = self._params["map_convergence_tol"]
        verbose = self._params["verbose"]
        max_ls = self._params["map_line_search_steps"]
        ls_shrink = self._params["map_line_search_shrink"]
        armijo_c = 1e-4

        for it in range(max_iter):
            f_m = self.inv_problem.forward(m)
            J = self._get_jacobian(m)
            res = f_m - d_obs
            dm_prior = m - self._m_prior

            g = J.T @ (Cd_inv @ res) + self._Qp @ dm_prior
            H = (J.T @ Cd_inv @ J + self._Qp).tocsc()
            delta = self._solve(H, -g)

            f0 = 0.5 * res @ (Cd_inv @ res) + 0.5 * dm_prior @ (self._Qp @ dm_prior)
            slope = g @ delta

            step = 1.0
            for _ in range(max_ls):
                m_trial = m + step * delta
                res_trial = self.inv_problem.forward(m_trial) - d_obs
                dm_trial = m_trial - self._m_prior
                f_trial = (
                    0.5 * res_trial @ (Cd_inv @ res_trial)
                    + 0.5 * dm_trial @ (self._Qp @ dm_trial)
                )
                if f_trial <= f0 + armijo_c * step * slope:
                    break
                step *= ls_shrink

            m = m + step * delta
            step_norm = np.linalg.norm(step * delta)
            if verbose:
                print(
                    f"MAP iteration {it + 1}/{max_iter},"
                    f" cost: {f_trial:.4e}, step: {step_norm:.2e}",
                    flush=True,
                )
            if step_norm < tol:
                if verbose:
                    print(f"MAP converged at iteration {it + 1}", flush=True)
                break

        H_map = self._get_jacobian(m)
        H_map = (H_map.T @ Cd_inv @ H_map + self._Qp).tocsc()
        return m, H_map

    # ------------------------------------------------------------------
    # Phase 2: Gaussian VI
    # ------------------------------------------------------------------

    def _run_gaussian_vi(self, m_map, H_map, d_obs, Cd_inv):
        r"""Phase 2: natural-gradient Gaussian VI on :math:`(\mu, \Omega)`.

        Per iteration:

        1. Factor :math:`\Omega_t` (with PD ridge fallback if needed) and
           draw ``num_samples`` Monte-Carlo samples from
           :math:`\mathcal{N}(\mu_t, \Omega_t^{-1})`.
        2. Apply per-coordinate perturbation clamping with warm-up.
        3. For each sample, compute the gradient
           :math:`g(m) = J^\top C_d^{-1}(f(m) - d) + Q_p(m - m_\text{prior})`
           and Gauss-Newton Hessian :math:`H(m) = J^\top C_d^{-1} J + Q_p`,
           skipping samples that produce non-finite values.
        4. Update :math:`\Omega` (with Hessian-rejection guard) and
           :math:`\mu` (with trust-region step clipping and optional
           diminishing step size); enforce a diagonal floor on
           :math:`\Omega` to keep it well-conditioned.
        5. Record the ELBO; check the patience-based convergence criterion.

        Parameters
        ----------
        m_map : numpy.ndarray, shape (N,)
            MAP estimate from Phase 1; used as the initial mean.
        H_map : scipy.sparse, shape (N, N)
            Gauss-Newton Hessian at MAP; used as the initial precision.
        d_obs : numpy.ndarray, shape (M,)
            Observations.
        Cd_inv : scipy.sparse, shape (M, M)
            Data precision.

        Returns
        -------
        mu : numpy.ndarray, shape (N,)
            Variational posterior mean.
        Omega : scipy.sparse.csc_matrix, shape (N, N)
            Variational posterior precision.
        elbo_history : list of float
            ELBO value at each completed iteration.
        """
        mu = m_map.copy()
        Omega = H_map.copy()
        N = len(mu)

        niter = self._params["num_iterations"]
        nsamp = self._params["num_samples"]
        rho_mu = self._params["learning_rate_mean"]
        rho_omega = self._params["learning_rate_precision"]
        diag_floor = self._params["diagonal_floor"]
        h_diag_floor = self._params["hessian_diagonal_floor"]
        max_pert = self._params["max_perturbation"]
        warmup = self._params["perturbation_warmup"]
        max_step = self._params["max_step_norm"]
        step_decay_tau = self._params["step_decay_timescale"]
        h_reject_ratio = self._params["hessian_rejection_ratio"]
        verbose = self._params["verbose"]
        patience = self._params["convergence_patience"]
        rtol = self._params["convergence_rtol"]

        elbo_history = []

        for it in range(niter):
            factor, logdet_Omega, Omega = self._factor_and_logdet(Omega, diag_floor)
            samples = self._sample_from_factor(mu, factor, nsamp)

            # Perturbation clamping with warmup
            samples = self._clamp_perturbation(samples, mu, max_pert, it, warmup)

            g_acc = np.zeros(N)
            H_list = []
            lls = []
            n_valid = 0

            for m_s in samples:
                f_s = self.inv_problem.forward(m_s)

                # NaN/Inf guard: skip bad samples
                if not np.all(np.isfinite(f_s)):
                    continue

                J_s = self._get_jacobian(m_s)
                if not np.all(np.isfinite(J_s.data)):
                    continue

                res_s = f_s - d_obs
                dm_s = m_s - self._m_prior

                g_s = J_s.T @ (Cd_inv @ res_s) + self._Qp @ dm_s
                if not np.all(np.isfinite(g_s)):
                    continue

                H_s = (J_s.T @ Cd_inv @ J_s + self._Qp).tocsc()

                # Per-sample Hessian diagonal floor
                H_s = self._enforce_hessian_diagonal_floor(H_s, h_diag_floor)

                g_acc += g_s
                H_list.append(H_s)
                n_valid += 1

                lls.append(
                    -0.5 * res_s @ (Cd_inv @ res_s)
                    - 0.5 * dm_s @ (self._Qp @ dm_s)
                )

            if n_valid == 0:
                if verbose:
                    print(f"Gaussian VI iteration {it + 1}: all samples invalid, skipping", flush=True)
                continue

            g_avg = g_acc / n_valid
            H_avg = sum(H_list[1:], H_list[0]) / n_valid

            elbo_history.append(self._compute_elbo(lls, logdet_Omega, N))

            if self._check_convergence(elbo_history, patience, rtol):
                if verbose:
                    print(f"Gaussian VI converged at iteration {it + 1}", flush=True)
                break

            # Diminishing step size
            rho_mu_t = rho_mu
            if step_decay_tau > 0:
                rho_mu_t = rho_mu / (1.0 + it / step_decay_tau)

            # Natural gradient mean update
            delta_mu = rho_mu_t * self._solve_from_factor(factor, g_avg)

            # Trust region: clip step norm
            delta_mu = self._clip_step(delta_mu, max_step, mu)

            mu -= delta_mu

            # Hessian rejection guard: skip Omega update if H_avg is extreme
            omega_norm = splinalg.norm(Omega, "fro")
            h_norm = splinalg.norm(H_avg, "fro")
            if omega_norm > 0 and h_norm / omega_norm > h_reject_ratio:
                if verbose:
                    print(
                        f"Gaussian VI iteration {it + 1}: "
                        f"||H_avg||/||Omega|| = {h_norm / omega_norm:.1f} > "
                        f"{h_reject_ratio}, skipping Omega update",
                        flush=True,
                    )
            else:
                Omega = (1 - rho_omega) * Omega + rho_omega * H_avg
                Omega = self._enforce_diagonal_floor(Omega, diag_floor)

            if verbose:
                print(
                    f"Gaussian VI iteration {it + 1}/{niter},"
                    f" ELBO: {elbo_history[-1]:.4f}",
                    flush=True,
                )

        return mu, Omega, elbo_history

    # ------------------------------------------------------------------
    # Phase 3: Sinh-arcsinh flow VI
    # ------------------------------------------------------------------

    @staticmethod
    def _flow_forward(z, a, b):
        r"""Apply the sinh-arcsinh transform elementwise:
        :math:`T(z) = \sinh(a \cdot \mathrm{arcsinh}(z) + b)`.
        """
        return np.sinh(a * np.arcsinh(z) + b)

    @staticmethod
    def _flow_log_det_jac(z, a, b):
        r"""Log-determinant of the elementwise Jacobian :math:`\partial T/\partial z`:
        :math:`\sum_i \big[\log a_i + \log\cosh(a_i\,\mathrm{arcsinh}(z_i) + b_i)
        - \tfrac{1}{2}\log(1 + z_i^2)\big]`.
        """
        s = a * np.arcsinh(z) + b
        return np.sum(np.log(a) + np.log(np.cosh(s)) - 0.5 * np.log(1 + z**2))

    @staticmethod
    def _flow_grad_logdet(z, a, b):
        r"""Gradient of :math:`\log|\det J_T|` w.r.t. :math:`(a, b)`.

        Returns
        -------
        grad_a : numpy.ndarray
            :math:`\partial / \partial a = 1/a + \mathrm{arcsinh}(z)\,\tanh(s)`.
        grad_b : numpy.ndarray
            :math:`\partial / \partial b = \tanh(s)` where
            :math:`s = a\,\mathrm{arcsinh}(z) + b`.
        """
        arcsinh_z = np.arcsinh(z)
        s = a * arcsinh_z + b
        tanh_s = np.tanh(s)
        return 1.0 / a + arcsinh_z * tanh_s, tanh_s

    @staticmethod
    def _flow_dTdparams(z, a, b):
        r"""Derivatives :math:`\partial T/\partial a` and :math:`\partial T/\partial b`.

        Used in the chain-rule pull-back of the log-joint gradient through
        the flow:
        :math:`\partial T/\partial a = \mathrm{arcsinh}(z)\,\cosh(s)`,
        :math:`\partial T/\partial b = \cosh(s)` with
        :math:`s = a\,\mathrm{arcsinh}(z) + b`.
        """
        arcsinh_z = np.arcsinh(z)
        s = a * arcsinh_z + b
        cosh_s = np.cosh(s)
        return arcsinh_z * cosh_s, cosh_s

    def _run_sas_vi(self, mu_init, Omega_init, d_obs, Cd_inv):
        r"""Phase 3: VI with a per-coordinate sinh-arcsinh normalising flow.

        Augments the Gaussian variational family with the elementwise
        bijection :math:`m_i = T_i(z_i; a_i, b_i) = \sinh(a_i\,
        \mathrm{arcsinh}(z_i) + b_i)`, where :math:`z \sim
        \mathcal{N}(\mu, \Omega^{-1})`.  This adds tunable skewness
        (:math:`b`) and kurtosis (:math:`a`) per coordinate while keeping
        the latent precision sparse.

        Per iteration, gradients are propagated through the flow by chain
        rule (:math:`g_z = g_m \cdot \partial T/\partial z` and similarly
        for the Hessian via :math:`D \cdot H_m \cdot D` with diagonal
        :math:`D`), :math:`(\mu, \Omega)` are updated by natural gradient
        as in Phase 2, and the flow parameters :math:`(a, b)` are updated
        by Adam ascent on the ELBO.  ``a`` is clipped from below at
        ``flow_a_min`` to keep the flow invertible.

        Parameters
        ----------
        mu_init : numpy.ndarray, shape (N,)
            Warm-start mean from Phase 2.
        Omega_init : scipy.sparse, shape (N, N)
            Warm-start precision from Phase 2.
        d_obs : numpy.ndarray, shape (M,)
            Observations.
        Cd_inv : scipy.sparse, shape (M, M)
            Data precision.

        Returns
        -------
        mu : numpy.ndarray, shape (N,)
            Final variational mean for the latent :math:`z`.
        Omega : scipy.sparse.csc_matrix, shape (N, N)
            Final variational precision for :math:`z`.
        a_flow : numpy.ndarray, shape (N,)
            Sinh-arcsinh tail-weight parameters.
        b_flow : numpy.ndarray, shape (N,)
            Sinh-arcsinh skewness parameters.
        elbo_history : list of float
            ELBO value (including the flow log-det term) per iteration.
        """
        N = len(mu_init)
        mu = mu_init.copy()
        Omega = Omega_init.copy()

        a_flow = np.ones(N)
        b_flow = np.zeros(N)

        niter = self._params["flow_num_iterations"]
        nsamp = self._params["flow_num_samples"]
        rho_mu = self._params["learning_rate_mean"]
        rho_omega = self._params["learning_rate_precision"]
        diag_floor = self._params["diagonal_floor"]
        h_diag_floor = self._params["hessian_diagonal_floor"]
        max_pert = self._params["max_perturbation"]
        warmup = self._params["perturbation_warmup"]
        max_step = self._params["max_step_norm"]
        step_decay_tau = self._params["step_decay_timescale"]
        h_reject_ratio = self._params["hessian_rejection_ratio"]
        verbose = self._params["verbose"]
        patience = self._params["convergence_patience"]
        rtol = self._params["convergence_rtol"]

        adam_lr = self._params["flow_learning_rate"]
        adam_b1 = self._params["flow_adam_beta1"]
        adam_b2 = self._params["flow_adam_beta2"]
        adam_eps = 1e-8
        a_min = self._params["flow_a_min"]

        # Adam state
        ma, va = np.zeros(N), np.zeros(N)
        mb, vb = np.zeros(N), np.zeros(N)

        elbo_history = []

        for it in range(niter):
            factor, logdet_Omega, Omega = self._factor_and_logdet(Omega, diag_floor)
            z_samples = self._sample_from_factor(mu, factor, nsamp)

            # Perturbation clamping with warmup
            z_samples = self._clamp_perturbation(z_samples, mu, max_pert, it, warmup)

            g_acc = np.zeros(N)
            H_list = []
            ga_acc, gb_acc = np.zeros(N), np.zeros(N)
            elbo_terms = []
            n_valid = 0

            for z_s in z_samples:
                m_s = self._flow_forward(z_s, a_flow, b_flow)

                f_s = self.inv_problem.forward(m_s)

                # NaN/Inf guard: skip bad samples
                if not np.all(np.isfinite(f_s)):
                    continue

                J_s = self._get_jacobian(m_s)
                if not np.all(np.isfinite(J_s.data)):
                    continue

                res_s = f_s - d_obs
                dm_s = m_s - self._m_prior

                g_m = J_s.T @ (Cd_inv @ res_s) + self._Qp @ dm_s
                if not np.all(np.isfinite(g_m)):
                    continue

                H_m = (J_s.T @ Cd_inv @ J_s + self._Qp).tocsc()

                # Per-sample Hessian diagonal floor
                H_m = self._enforce_hessian_diagonal_floor(H_m, h_diag_floor)

                # Chain rule through flow: dT/dz is diagonal
                arcsinh_z = np.arcsinh(z_s)
                s = a_flow * arcsinh_z + b_flow
                dTdz = a_flow * np.cosh(s) / np.sqrt(1 + z_s**2)

                D = sparse.diags(dTdz)
                g_acc += g_m * dTdz
                H_list.append((D @ H_m @ D).tocsc())

                # Flow parameter gradients
                ga_ent, gb_ent = self._flow_grad_logdet(z_s, a_flow, b_flow)
                dTda, dTdb = self._flow_dTdparams(z_s, a_flow, b_flow)
                ga_acc += ga_ent - g_m * dTda
                gb_acc += gb_ent - g_m * dTdb

                # ELBO log-joint + flow log-det term
                ll = (
                    -0.5 * res_s @ (Cd_inv @ res_s)
                    - 0.5 * dm_s @ (self._Qp @ dm_s)
                )
                ll += self._flow_log_det_jac(z_s, a_flow, b_flow)
                elbo_terms.append(ll)
                n_valid += 1

            if n_valid == 0:
                if verbose:
                    print(
                        f"SAS VI iteration {it + 1}: all samples invalid, skipping",
                        flush=True,
                    )
                continue

            g_avg = g_acc / n_valid
            H_avg = sum(H_list[1:], H_list[0]) / n_valid
            ga_avg = ga_acc / n_valid
            gb_avg = gb_acc / n_valid

            # ELBO: E_q(z)[log p(d,T(z)) + log|det J_T|] + H[q_z]
            entropy = 0.5 * N * (1 + np.log(2 * np.pi)) - 0.5 * logdet_Omega
            elbo_history.append(np.mean(elbo_terms) + entropy)

            if self._check_convergence(elbo_history, patience, rtol):
                if verbose:
                    print(f"SAS VI converged at iteration {it + 1}", flush=True)
                break

            # Diminishing step size
            rho_mu_t = rho_mu
            if step_decay_tau > 0:
                rho_mu_t = rho_mu / (1.0 + it / step_decay_tau)

            # Natural gradient mean update
            delta_mu = rho_mu_t * self._solve_from_factor(factor, g_avg)

            # Trust region: clip step norm
            delta_mu = self._clip_step(delta_mu, max_step, mu)

            mu -= delta_mu

            # Hessian rejection guard: skip Omega update if H_avg is extreme
            omega_norm = splinalg.norm(Omega, "fro")
            h_norm = splinalg.norm(H_avg, "fro")
            if omega_norm > 0 and h_norm / omega_norm > h_reject_ratio:
                if verbose:
                    print(
                        f"SAS VI iteration {it + 1}: "
                        f"||H_avg||/||Omega|| = {h_norm / omega_norm:.1f} > "
                        f"{h_reject_ratio}, skipping Omega update",
                        flush=True,
                    )
            else:
                Omega = (1 - rho_omega) * Omega + rho_omega * H_avg
                Omega = self._enforce_diagonal_floor(Omega, diag_floor)

            # Adam update for flow params (ascend on ELBO)
            t = it + 1
            for param, grad, mst, vst in [
                (a_flow, ga_avg, ma, va),
                (b_flow, gb_avg, mb, vb),
            ]:
                mst[:] = adam_b1 * mst + (1 - adam_b1) * grad
                vst[:] = adam_b2 * vst + (1 - adam_b2) * grad**2
                m_hat = mst / (1 - adam_b1**t)
                v_hat = vst / (1 - adam_b2**t)
                param += adam_lr * m_hat / (np.sqrt(v_hat) + adam_eps)

            a_flow[:] = np.maximum(a_flow, a_min)

            if verbose:
                print(
                    f"SAS VI iteration {it + 1}/{niter},"
                    f" ELBO: {elbo_history[-1]:.4f}",
                    flush=True,
                )

        return mu, Omega, a_flow, b_flow, elbo_history


# CoFI -> Ensemble methods -> Variational inference -> cofi.gaussian_vi -> Gaussian VI
# description: Natural-gradient Gaussian VI with optional sinh-arcsinh normalising flow for Bayesian posterior approximation.
# documentation: https://arxiv.org/abs/1703.04265
