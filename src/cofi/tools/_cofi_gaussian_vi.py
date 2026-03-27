import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from . import BaseInferenceTool


class VISampler:
    """Lightweight wrapper holding a fitted VI posterior for on-demand sampling.

    Stores the posterior mean, precision, and optional sinh-arcsinh flow
    parameters. Provides a ``sample(n)`` method that draws from the
    approximate posterior.

    Parameters
    ----------
    mu : numpy.ndarray
        Posterior mean (N,).
    omega : scipy.sparse matrix or numpy.ndarray
        Posterior precision matrix (N, N).
    flow_a : numpy.ndarray or None
        Sinh-arcsinh tail weight parameters (N,). None if no flow.
    flow_b : numpy.ndarray or None
        Sinh-arcsinh skewness parameters (N,). None if no flow.
    random_state : numpy.random.Generator or None
        Random state for reproducibility.
    """

    def __init__(self, mu, omega, flow_a=None, flow_b=None, random_state=None):
        self.mu = mu
        self.omega = omega
        self.flow_a = flow_a
        self.flow_b = flow_b
        self._rng = random_state or np.random.default_rng()

    def sample(self, n=1):
        """Draw n samples from the approximate posterior.

        Returns
        -------
        numpy.ndarray
            Samples with shape (n, N).
        """
        N = len(self.mu)
        omega_dense = (
            self.omega.toarray() if sparse.issparse(self.omega) else self.omega
        )
        L = np.linalg.cholesky(omega_dense)
        samples = np.empty((n, N))
        for i in range(n):
            z = self._rng.standard_normal(N)
            v = np.linalg.solve(L.T, z)
            samples[i] = self.mu + v
        if self.flow_a is not None and self.flow_b is not None:
            samples = np.sinh(
                self.flow_a[np.newaxis, :] * np.arcsinh(samples)
                + self.flow_b[np.newaxis, :]
            )
        return samples


class CoFIGaussianVI(BaseInferenceTool):
    r"""Natural-gradient Gaussian VI with optional sinh-arcsinh flow.

    Approximates the posterior :math:`p(m|d)` with a parametric distribution
    :math:`q(m)` by maximising the Evidence Lower Bound (ELBO). Uses
    precision-first parameterisation to preserve sparsity, and natural
    gradient updates for reparameterisation-invariant optimisation.

    The algorithm:

    1. Find the MAP estimate via Gauss-Newton iteration.
    2. Initialise :math:`\mu = m_\text{MAP}`, :math:`\Omega = H_\text{GN}(m_\text{MAP})`.
    3. Iterate natural gradient updates on :math:`(\mu, \Omega)`.
    4. Optionally fit a sinh-arcsinh elementwise flow for skewness/kurtosis.

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
            "enable_flow": False,
            "flow_num_iterations": 300,
            "flow_num_samples": 32,
            "flow_learning_rate": 0.003,
            "flow_adam_beta1": 0.9,
            "flow_adam_beta2": 0.999,
            "flow_a_min": 0.1,
            "map_num_iterations": 30,
            "map_convergence_tol": 1e-8,
            "verbose": True,
            "random_seed": None,
        }

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._Qp = self._params["prior_precision"]
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
        d_obs = np.asarray(self.inv_problem.data)

        # Phase 1: MAP initialisation
        m_map, H_map = self._run_map(m0, d_obs, Cd_inv)

        # Phase 2: Gaussian VI
        mu, Omega, elbo_history = self._run_gaussian_vi(
            m_map, H_map, d_obs, Cd_inv
        )

        flow_params = None

        # Phase 3: Optional SAS flow
        if self._params["enable_flow"]:
            mu, Omega, a_flow, b_flow, elbo_sa = self._run_sas_vi(
                m_map, d_obs, Cd_inv
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
    # Internal: gradient and Hessian
    # ------------------------------------------------------------------

    def _compute_gradient(self, m, d_obs, Cd_inv):
        """Gradient of neg-log-posterior: J^T C_d^{-1} (f(m)-d) + Q_p (m-m_0)."""
        res = self.inv_problem.forward(m) - d_obs
        J = self.inv_problem.jacobian(m)
        return J.T @ (Cd_inv @ res) + self._Qp @ (m - self._m_prior)

    def _compute_gn_hessian(self, m, Cd_inv):
        """Gauss-Newton Hessian: J^T C_d^{-1} J + Q_p (sparse)."""
        J = self.inv_problem.jacobian(m)
        H = J.T @ Cd_inv @ J + self._Qp
        if sparse.issparse(H):
            H = H.tocsc()
        return H

    # ------------------------------------------------------------------
    # Internal: linear algebra helpers
    # ------------------------------------------------------------------

    def _sample(self, mu, Omega, n):
        """Sample from N(mu, Omega^{-1}) via Cholesky factorisation."""
        N = len(mu)
        omega_dense = Omega.toarray() if sparse.issparse(Omega) else Omega
        L = np.linalg.cholesky(omega_dense)
        samples = []
        for _ in range(n):
            z = self._rng.standard_normal(N)
            v = np.linalg.solve(L.T, z)
            samples.append(mu + v)
        return samples

    @staticmethod
    def _solve(Omega, b):
        """Solve Omega x = b via sparse LU."""
        if sparse.issparse(Omega):
            return splinalg.spsolve(Omega.tocsc(), b)
        return np.linalg.solve(Omega, b)

    @staticmethod
    def _logdet(Omega):
        """Exact log|Omega| via Cholesky."""
        omega_dense = Omega.toarray() if sparse.issparse(Omega) else Omega
        L = np.linalg.cholesky(omega_dense)
        return 2.0 * np.sum(np.log(np.diag(L)))

    # ------------------------------------------------------------------
    # Internal: ELBO computation
    # ------------------------------------------------------------------

    def _compute_elbo(self, mu, Omega, samples, d_obs, Cd_inv):
        """ELBO = E_q[log p(d,m)] + H[q]."""
        lls = []
        for m in samples:
            res = self.inv_problem.forward(m) - d_obs
            dm = m - self._m_prior
            ll = -0.5 * res @ (Cd_inv @ res) - 0.5 * dm @ (self._Qp @ dm)
            lls.append(ll)
        return np.mean(lls) - 0.5 * self._logdet(Omega)

    # ------------------------------------------------------------------
    # Phase 1: MAP via Gauss-Newton
    # ------------------------------------------------------------------

    def _run_map(self, m0, d_obs, Cd_inv):
        """Gauss-Newton MAP iteration."""
        m = m0.copy()
        max_iter = self._params["map_num_iterations"]
        tol = self._params["map_convergence_tol"]
        verbose = self._params["verbose"]

        for it in range(max_iter):
            g = self._compute_gradient(m, d_obs, Cd_inv)
            H = self._compute_gn_hessian(m, Cd_inv)
            dm = self._solve(H, -g)
            m += dm
            if np.linalg.norm(dm) < tol:
                if verbose:
                    print(f"MAP converged at iteration {it}")
                break

        H_map = self._compute_gn_hessian(m, Cd_inv)
        return m, H_map

    # ------------------------------------------------------------------
    # Phase 2: Gaussian VI
    # ------------------------------------------------------------------

    def _run_gaussian_vi(self, m_map, H_map, d_obs, Cd_inv):
        """Core natural-gradient Gaussian VI loop."""
        mu = m_map.copy()
        Omega = H_map.copy() if sparse.issparse(H_map) else H_map.copy()
        N = len(mu)

        niter = self._params["num_iterations"]
        nsamp = self._params["num_samples"]
        rho_mu = self._params["learning_rate_mean"]
        rho_omega = self._params["learning_rate_precision"]
        diag_floor = self._params["diagonal_floor"]
        verbose = self._params["verbose"]

        elbo_history = []

        for it in range(niter):
            samples = self._sample(mu, Omega, nsamp)

            g_avg = np.zeros(N)
            H_avg = sparse.csc_matrix((N, N)) if sparse.issparse(Omega) else np.zeros((N, N))
            for m_s in samples:
                g_avg += self._compute_gradient(m_s, d_obs, Cd_inv)
                H_avg = H_avg + self._compute_gn_hessian(m_s, Cd_inv)
            g_avg /= nsamp
            H_avg = H_avg / nsamp

            elbo_history.append(self._compute_elbo(mu, Omega, samples, d_obs, Cd_inv))

            # Natural gradient mean update
            mu -= rho_mu * self._solve(Omega, g_avg)

            # Precision update: convex combination
            Omega = (1 - rho_omega) * Omega + rho_omega * H_avg
            if sparse.issparse(Omega):
                Omega = Omega.tocsc()
                Omega.setdiag(np.maximum(Omega.diagonal(), diag_floor))
            else:
                np.fill_diagonal(Omega, np.maximum(np.diag(Omega), diag_floor))

            if verbose and (it + 1) % max(1, niter // 10) == 0:
                print(f"Gaussian VI iteration {it + 1}/{niter}, ELBO: {elbo_history[-1]:.4f}")

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

    def _run_sas_vi(self, m_map, d_obs, Cd_inv):
        """VI loop with sinh-arcsinh normalising flow."""
        N = len(m_map)
        mu = m_map.copy()
        Omega = self._compute_gn_hessian(m_map, Cd_inv)

        a_flow = np.ones(N)
        b_flow = np.zeros(N)

        niter = self._params["flow_num_iterations"]
        nsamp = self._params["flow_num_samples"]
        rho_mu = self._params["learning_rate_mean"]
        rho_omega = self._params["learning_rate_precision"]
        diag_floor = self._params["diagonal_floor"]
        verbose = self._params["verbose"]

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
            z_samples = self._sample(mu, Omega, nsamp)

            g_avg = np.zeros(N)
            H_avg = sparse.csc_matrix((N, N)) if sparse.issparse(Omega) else np.zeros((N, N))
            ga_avg, gb_avg = np.zeros(N), np.zeros(N)
            elbo_terms = []

            for z_s in z_samples:
                m_s = self._flow_forward(z_s, a_flow, b_flow)
                g_m = self._compute_gradient(m_s, d_obs, Cd_inv)
                H_m = self._compute_gn_hessian(m_s, Cd_inv)

                # Chain rule through flow: dT/dz is diagonal
                arcsinh_z = np.arcsinh(z_s)
                s = a_flow * arcsinh_z + b_flow
                dTdz = a_flow * np.cosh(s) / np.sqrt(1 + z_s**2)

                g_avg += g_m * dTdz
                D = sparse.diags(dTdz) if sparse.issparse(H_m) else np.diag(dTdz)
                H_avg = H_avg + D @ H_m @ D

                # Flow parameter gradients
                ga_ent, gb_ent = self._flow_grad_logdet(z_s, a_flow, b_flow)
                dTda, dTdb = self._flow_dTdparams(z_s, a_flow, b_flow)
                ga_avg += ga_ent - g_m * dTda
                gb_avg += gb_ent - g_m * dTdb

                # ELBO log-joint term
                res = self.inv_problem.forward(m_s) - d_obs
                dm = m_s - self._m_prior
                ll = -0.5 * res @ (Cd_inv @ res) - 0.5 * dm @ (self._Qp @ dm)
                ll += self._flow_log_det_jac(z_s, a_flow, b_flow)
                elbo_terms.append(ll)

            g_avg /= nsamp
            H_avg = H_avg / nsamp
            ga_avg /= nsamp
            gb_avg /= nsamp

            # ELBO before updates
            entropy = -0.5 * self._logdet(Omega)
            elbo_history.append(np.mean(elbo_terms) + entropy)

            # Natural gradient mean update
            mu -= rho_mu * self._solve(Omega, g_avg)

            # Precision update
            Omega = (1 - rho_omega) * Omega + rho_omega * H_avg
            if sparse.issparse(Omega):
                Omega = Omega.tocsc()
                Omega.setdiag(np.maximum(Omega.diagonal(), diag_floor))
            else:
                np.fill_diagonal(Omega, np.maximum(np.diag(Omega), diag_floor))

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
                print(f"SAS VI iteration {it + 1}/{niter}, ELBO: {elbo_history[-1]:.4f}")

        return mu, Omega, a_flow, b_flow, elbo_history


# CoFI -> Ensemble methods -> Variational inference -> cofi.gaussian_vi -> Gaussian VI
# description: Natural-gradient Gaussian VI with optional sinh-arcsinh normalising flow for Bayesian posterior approximation.
# documentation: https://arxiv.org/abs/1703.04265
