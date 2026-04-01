import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from . import BaseInferenceTool

try:
    from sksparse.cholmod import cholesky as _cholmod_cholesky

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
    """Raise ImportError if CHOLMOD is not available."""
    if not _HAS_CHOLMOD:
        raise ImportError(
            "CoFIGaussianVI requires scikit-sparse (CHOLMOD). "
            "Install with: pip install scikit-sparse"
        )


class VISampler:
    """Lightweight wrapper holding a fitted VI posterior for on-demand sampling.

    Stores the posterior mean, precision, and optional sinh-arcsinh flow
    parameters. Provides a ``sample(n)`` method that draws from the
    approximate posterior.

    Parameters
    ----------
    mu : numpy.ndarray
        Posterior mean (N,).
    omega : scipy.sparse matrix
        Posterior precision matrix (N, N).  Must be sparse.
    flow_a : numpy.ndarray or None
        Sinh-arcsinh tail weight parameters (N,). None if no flow.
    flow_b : numpy.ndarray or None
        Sinh-arcsinh skewness parameters (N,). None if no flow.
    random_state : numpy.random.Generator or None
        Random state for reproducibility.
    """

    def __init__(self, mu, omega, flow_a=None, flow_b=None, random_state=None):
        _require_sparse(omega, "omega")
        _require_cholmod()
        self.mu = mu
        self.omega = omega
        self.flow_a = flow_a
        self.flow_b = flow_b
        self._rng = random_state or np.random.default_rng()
        self._factor = _cholmod_cholesky(omega.tocsc())

    def sample(self, n=1):
        """Draw n samples from the approximate posterior.

        Returns
        -------
        numpy.ndarray
            Samples with shape (n, N).
        """
        N = len(self.mu)
        Z = self._rng.standard_normal((n, N))

        # P A P^T = L L^T  =>  sample = mu + P^T L^{-T} z
        samples = np.empty((n, N))
        for i in range(n):
            y = self._factor.solve_Lt(Z[i], use_LDLt_decomposition=False)
            samples[i] = self.mu + self._factor.apply_Pt(y)

        if self.flow_a is not None and self.flow_b is not None:
            samples = np.sinh(
                self.flow_a[np.newaxis, :] * np.arcsinh(samples)
                + self.flow_b[np.newaxis, :]
            )
        return samples


class CoFIGaussianVI(BaseInferenceTool):
    r"""Natural-gradient Gaussian VI with optional sinh-arcsinh flow.

    All matrices (prior precision, data covariance inverse, Jacobian) must
    be scipy sparse matrices.  Requires scikit-sparse (CHOLMOD) for sparse
    Cholesky factorisation.

    Approximates the posterior :math:`p(m|d)` with a parametric distribution
    :math:`q(m)` by maximising the Evidence Lower Bound (ELBO). Uses
    precision-first parameterisation to preserve sparsity, and natural
    gradient updates for reparameterisation-invariant optimisation.

    The algorithm:

    1. Find the MAP estimate via Gauss-Newton iteration with line search.
    2. Initialise :math:`\mu = m_\text{MAP}`, :math:`\Omega = H_\text{GN}(m_\text{MAP})`.
    3. Iterate natural gradient updates on :math:`(\mu, \Omega)`.
    4. Optionally fit a sinh-arcsinh elementwise flow for skewness/kurtosis,
       warm-started from step 3.

    References
    ----------
    - Khan & Lin (2017). Conjugate-Computation Variational Inference.
    - Khan & Rue (2023). The Bayesian Learning Rule.
    - Jones & Pewsey (2009). Sinh-arcsinh distributions.
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

        Returns (cholmod_factor, logdet, Omega).  Omega may be
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
                factor = _cholmod_cholesky(test.tocsc())
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
            y = factor.solve_Lt(Z[i], use_LDLt_decomposition=False)
            samples[i] = mu + factor.apply_Pt(y)
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

        If max_norm <= 0, use adaptive trust region = ||mu|| (or 1 if mu ~ 0).
        """
        if max_norm <= 0:
            max_norm = max(np.linalg.norm(mu), 1.0)
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
        """Gauss-Newton MAP iteration with Armijo backtracking line search."""
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
            if np.linalg.norm(step * delta) < tol:
                if verbose:
                    print(f"MAP converged at iteration {it}")
                break

        H_map = self._get_jacobian(m)
        H_map = (H_map.T @ Cd_inv @ H_map + self._Qp).tocsc()
        return m, H_map

    # ------------------------------------------------------------------
    # Phase 2: Gaussian VI
    # ------------------------------------------------------------------

    def _run_gaussian_vi(self, m_map, H_map, d_obs, Cd_inv):
        """Core natural-gradient Gaussian VI loop."""
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
            H_acc = sparse.csc_matrix((N, N))
            lls = []
            n_valid = 0

            for m_s in samples:
                f_s = self.inv_problem.forward(m_s)

                # NaN/Inf guard: skip bad samples
                if not np.all(np.isfinite(f_s)):
                    continue

                J_s = self._get_jacobian(m_s)
                res_s = f_s - d_obs
                dm_s = m_s - self._m_prior

                g_acc += J_s.T @ (Cd_inv @ res_s) + self._Qp @ dm_s
                H_s = (J_s.T @ Cd_inv @ J_s + self._Qp).tocsc()

                # Per-sample Hessian diagonal floor
                H_s = self._enforce_hessian_diagonal_floor(H_s, h_diag_floor)

                H_acc = H_acc + H_s
                n_valid += 1

                lls.append(
                    -0.5 * res_s @ (Cd_inv @ res_s)
                    - 0.5 * dm_s @ (self._Qp @ dm_s)
                )

            if n_valid == 0:
                if verbose:
                    print(f"Gaussian VI iteration {it + 1}: all samples invalid, skipping")
                continue

            g_avg = g_acc / n_valid
            H_avg = H_acc / n_valid

            elbo_history.append(self._compute_elbo(lls, logdet_Omega, N))

            if self._check_convergence(elbo_history, patience, rtol):
                if verbose:
                    print(f"Gaussian VI converged at iteration {it + 1}")
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
                        f"{h_reject_ratio}, skipping Omega update"
                    )
            else:
                Omega = (1 - rho_omega) * Omega + rho_omega * H_avg
                Omega = self._enforce_diagonal_floor(Omega, diag_floor)

            if verbose and (it + 1) % max(1, niter // 10) == 0:
                print(
                    f"Gaussian VI iteration {it + 1}/{niter},"
                    f" ELBO: {elbo_history[-1]:.4f}"
                )

        return mu, Omega, elbo_history

    # ------------------------------------------------------------------
    # Phase 3: Sinh-arcsinh flow VI
    # ------------------------------------------------------------------

    @staticmethod
    def _flow_forward(z, a, b):
        """m = sinh(a * arcsinh(z) + b)"""
        return np.sinh(a * np.arcsinh(z) + b)

    @staticmethod
    def _flow_log_det_jac(z, a, b):
        """Sum of log|dT_i/dz_i|."""
        s = a * np.arcsinh(z) + b
        return np.sum(np.log(a) + np.log(np.cosh(s)) - 0.5 * np.log(1 + z**2))

    @staticmethod
    def _flow_grad_logdet(z, a, b):
        """Gradient of log|det J_T| w.r.t. (a, b)."""
        arcsinh_z = np.arcsinh(z)
        s = a * arcsinh_z + b
        tanh_s = np.tanh(s)
        return 1.0 / a + arcsinh_z * tanh_s, tanh_s

    @staticmethod
    def _flow_dTdparams(z, a, b):
        """Derivatives dT/da and dT/db for chain rule through log-joint."""
        arcsinh_z = np.arcsinh(z)
        s = a * arcsinh_z + b
        cosh_s = np.cosh(s)
        return arcsinh_z * cosh_s, cosh_s

    def _run_sas_vi(self, mu_init, Omega_init, d_obs, Cd_inv):
        """VI loop with sinh-arcsinh normalising flow.

        Warm-started from the Gaussian VI posterior (mu_init, Omega_init).
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
            H_acc = sparse.csc_matrix((N, N))
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
                res_s = f_s - d_obs
                dm_s = m_s - self._m_prior

                g_m = J_s.T @ (Cd_inv @ res_s) + self._Qp @ dm_s
                H_m = (J_s.T @ Cd_inv @ J_s + self._Qp).tocsc()

                # Per-sample Hessian diagonal floor
                H_m = self._enforce_hessian_diagonal_floor(H_m, h_diag_floor)

                # Chain rule through flow: dT/dz is diagonal
                arcsinh_z = np.arcsinh(z_s)
                s = a_flow * arcsinh_z + b_flow
                dTdz = a_flow * np.cosh(s) / np.sqrt(1 + z_s**2)

                D = sparse.diags(dTdz)
                g_acc += g_m * dTdz
                H_acc = H_acc + D @ H_m @ D

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
                    print(f"SAS VI iteration {it + 1}: all samples invalid, skipping")
                continue

            g_avg = g_acc / n_valid
            H_avg = H_acc / n_valid
            ga_avg = ga_acc / n_valid
            gb_avg = gb_acc / n_valid

            # ELBO: E_q(z)[log p(d,T(z)) + log|det J_T|] + H[q_z]
            entropy = 0.5 * N * (1 + np.log(2 * np.pi)) - 0.5 * logdet_Omega
            elbo_history.append(np.mean(elbo_terms) + entropy)

            if self._check_convergence(elbo_history, patience, rtol):
                if verbose:
                    print(f"SAS VI converged at iteration {it + 1}")
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
                        f"{h_reject_ratio}, skipping Omega update"
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

            if verbose and (it + 1) % max(1, niter // 10) == 0:
                print(
                    f"SAS VI iteration {it + 1}/{niter},"
                    f" ELBO: {elbo_history[-1]:.4f}"
                )

        return mu, Omega, a_flow, b_flow, elbo_history


# CoFI -> Ensemble methods -> Variational inference -> cofi.gaussian_vi -> Gaussian VI
# description: Natural-gradient Gaussian VI with optional sinh-arcsinh normalising flow for Bayesian posterior approximation.
# documentation: https://arxiv.org/abs/1703.04265
