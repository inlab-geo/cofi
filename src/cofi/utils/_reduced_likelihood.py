"""Reduced likelihood implementation for various covariance cases."""

from typing import Optional, Callable, Tuple
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

try:
    from ._lik_base import BaseLikelihood, DimensionMismatchError
except ImportError:
    # For standalone testing
    from _lik_base import BaseLikelihood, DimensionMismatchError

EPS = 1e-154


class ReducedLikelihood(BaseLikelihood):
    r"""Reduced likelihood for various data covariance cases.

    This class implements reduced likelihood functions for different
    assumptions about the data covariance structure:

    - **'none'**: Fixed covariance (standard Gaussian likelihood)
    - **'scaled'**: Scaled reference covariance
    - **'kernel'**: Profiled amplitude with parameterized kernel covariance
    - **'kernel_full'**: Explicit amplitude with parameterized kernel covariance
    - **'spherical'**: Spherical covariance (diagonal with same variance)
    - **'diag'**: Diagonal covariance with Student-t likelihood (robust)
    - **'diag_legacy'**: Diagonal covariance (legacy, numerically unstable near zero)
    - **'full'**: Full covariance matrix estimation

    Parameters
    ----------
    data : np.ndarray
        The observed data vector of shape (n_data,)
    forward_func : Callable
        A callable that takes model parameters and returns predicted data
        of shape (n_data,). Should accept **fwd_kwargs if provided.
    fwd_kwargs : dict, optional
        Keyword arguments to pass to forward_func
    G : Optional[np.ndarray], default=None
        Jacobian matrix of shape (n_data, n_params). Required for
        ``gradient()`` and ``hessian()`` but not for ``log_likelihood()``.
        Must be updated before each evaluation for non-linear problems.
    Cd_ref : Optional[np.ndarray], default=None
        Reference covariance matrix. Required for 'scaled' case.
        For 'none' case: if not provided, defaults to identity matrix
        (assumes uncorrelated, unit-variance noise).
        Not used for 'spherical', 'diag', 'diag_legacy', or 'full' cases.
    kernel : optional
        Kernel instance (e.g. ``SquaredExponentialKernel``).  Required for
        ``case='kernel'``.  Must expose ``n_params``, ``evaluate(eta)``,
        ``derivative(eta)``, and ``second_derivative(eta)`` methods.
    case : str, default='none'
        The covariance case: 'none', 'scaled', 'kernel',
        'kernel_full', 'spherical', 'diag', 'diag_legacy', or 'full'.
        The ``'kernel_full'`` case is like ``'kernel'`` but with
        explicit ``phi = log(sigma_d)`` as a parameter instead of profiling
        ``sigma_d`` out. Model vector: ``[m_phys, phi, eta]``.
    nu : float, default=4.0
        Degrees of freedom for Student-t likelihood (used in 'diag' case).
        Smaller values (e.g., 3-5) give heavier tails and more robustness to outliers.
        Larger values approach Gaussian behavior.
    s : float, default=1.0
        Scale parameter for Student-t likelihood (used in 'diag' case).
        Controls the width of the distribution. Should be set to approximate
        the expected noise standard deviation (e.g., via a quick least-squares prefit).
    eps : float, default=1e-154
        Small number for numerical stability
    n_params : int, optional
        Number of physical model parameters (excluding kernel hyperparameters).
        When ``G`` is provided, this is inferred from ``G.shape[1]``. When
        ``G=None``, provide this to enable model shape validation for
        ``log_likelihood()`` without computing the Jacobian. If both ``G``
        and ``n_params`` are provided, they must be consistent
        (``G.shape[1] == n_params``).

    Raises
    ------
    ValueError
        If Cd_ref is not provided for 'scaled' case
        If kernel is not provided for 'kernel' case
        If case is not one of the supported options
        If G is not set when calling gradient() or hessian()

    Examples
    --------
    >>> import numpy as np
    >>> from cofi.utils import ReducedLikelihood
    >>>
    >>> # Define data and forward function
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> def forward_func(m):
    ...     return np.array([m[0], m[1], m[0] + m[1]])  # Linear forward model
    >>>
    >>> # Jacobian matrix (can be computed separately)
    >>> G = np.array([[1, 0], [0, 1], [1, 1]])
    >>>
    >>> # Create likelihood with spherical covariance assumption
    >>> likelihood = ReducedLikelihood(
    ...     data=data,
    ...     forward_func=forward_func,
    ...     G=G,
    ...     case='spherical'
    ... )
    >>>
    >>> # Evaluate at a model
    >>> model = np.array([1.5, 2.5])
    >>> log_p = likelihood.log_likelihood(model)
    >>> grad = likelihood.gradient(model)
    >>> hess = likelihood.hessian(model)
    >>>
    >>> # For non-linear problems, update G before each evaluation
    >>> new_model = np.array([2.0, 3.0])
    >>> # Compute new Jacobian (e.g., using finite differences)
    >>> G_new = compute_jacobian(new_model)  # User-defined
    >>> likelihood.G = G_new
    >>> log_p_new = likelihood.log_likelihood(new_model)

    Derivative-free optimization (no Jacobian needed):

    >>> # For Nelder-Mead or other derivative-free methods,
    >>> # only log_likelihood() is needed — no G required.
    >>> likelihood = ReducedLikelihood(
    ...     data=data,
    ...     forward_func=forward_func,
    ...     case='spherical',
    ...     n_params=2,  # number of model parameters
    ... )
    >>> log_p = likelihood.log_likelihood(model)  # Works without G
    """

    def __init__(
        self,
        data: np.ndarray,
        forward_func: Callable,
        fwd_kwargs: dict = None,
        G: Optional[np.ndarray] = None,
        Cd_ref: Optional[np.ndarray] = None,
        case: str = "none",
        nu: float = 4.0,
        s: float = 1.0,
        eps: float = EPS,
        kernel=None,
        n_params: Optional[int] = None,
    ):
        """Initialize the reduced likelihood."""
        super().__init__()
        self.data = np.asarray(data)
        self.n_data = self.data.size
        self.forward_func = forward_func
        self.fwd_kwargs = fwd_kwargs if fwd_kwargs is not None else {}
        self.n_params = n_params

        # Initialize cache attributes before G setter can access them
        self._model_shape = None
        self._cache = {}
        self._cache_model = None

        # Set G using the property setter (handles type conversion + cache invalidation)
        if G is None:
            self._G = None
        elif sparse.issparse(G):
            self._G = G
        else:
            self._G = np.asarray(G)

        self.Cd_ref = None if Cd_ref is None else np.asarray(Cd_ref)
        self.case = case.lower()
        self.nu = float(nu)  # Student-t degrees of freedom
        self.s = float(s)    # Student-t scale parameter
        self.eps = float(eps)
        self.kernel = kernel

        # Validate n_params consistency with G
        if self._G is not None and self.n_params is not None:
            if self._G.shape[1] != self.n_params:
                raise ValueError(
                    f"Inconsistent n_params={self.n_params} and "
                    f"G.shape[1]={self._G.shape[1]}. These must match."
                )

        # Validate inputs and set defaults
        if self.case in ('kernel', 'kernel_full') and self.kernel is None:
            raise ValueError(
                f"kernel is required for case='{self.case}'. "
                f"Provide a kernel instance (e.g. SquaredExponentialKernel)."
            )
        if self.case == 'scaled' and self.Cd_ref is None:
            raise ValueError(
                f"Cd_ref is required for case='scaled'. "
                f"Provide a reference covariance matrix."
            )
        elif self.case == 'none' and self.Cd_ref is None:
            # Default to identity matrix (uncorrelated, unit variance noise)
            self.Cd_ref = np.eye(self.n_data)

        if self.case not in ['none', 'scaled', 'kernel',
                              'kernel_full', 'spherical',
                              'diag', 'diag_legacy', 'full']:
            raise ValueError(
                f"Unknown case '{self.case}'. Must be one of: "
                "'none', 'scaled', 'kernel', 'kernel_full', "
                "'spherical', 'diag', 'diag_legacy', 'full'"
            )

    @property
    def G(self):
        """Jacobian matrix of shape (n_data, n_params)."""
        return self._G

    @G.setter
    def G(self, value):
        """Set the Jacobian matrix and invalidate cached values."""
        if value is None:
            self._G = None
        elif sparse.issparse(value):
            self._G = value
        else:
            self._G = np.asarray(value)
        # Invalidate caches
        self._model_shape = None
        self._cache_model = None
        self._cache.clear()

    def _get_n_m(self):
        """Determine the number of physical model parameters.

        Returns n_m from G.shape[1] if G is set, otherwise from n_params.
        Returns None if neither is available.
        """
        if self._G is not None:
            return self._G.shape[1]
        if self.n_params is not None:
            return self.n_params
        return None

    @property
    def model_shape(self) -> tuple:
        """Return the model shape."""
        if self._model_shape is None:
            n_m = self._get_n_m()
            if n_m is not None:
                if self.case == 'kernel':
                    self._model_shape = (n_m + self.kernel.n_params,)
                elif self.case == 'kernel_full':
                    # [m_phys, phi, eta]: n_m + 1 + kernel.n_params
                    self._model_shape = (n_m + 1 + self.kernel.n_params,)
                else:
                    self._model_shape = (n_m,)
        return self._model_shape

    def _safe_scalar(self, x: float) -> float:
        """Ensure scalar is above numerical floor."""
        return max(x, self.eps)

    # ------------------------------------------------------------------
    # Tier 1: Log-likelihood (no G required)
    # ------------------------------------------------------------------

    def _evaluate_loglik(self, model: np.ndarray):
        """Compute log-likelihood and Cd_ml (no G required).

        Caches intermediate quantities needed by _evaluate_gradient and _evaluate_hessian.

        Returns
        -------
        tuple
            (log_likelihood, Cd_ml)
        """
        # Check cache — return if already computed for this model
        if (self._cache_model is not None
                and np.array_equal(model, self._cache_model)
                and 'log_likelihood' in self._cache):
            return self._cache['log_likelihood'], self._cache.get('Cd_ml')

        N = self.n_data

        # For kernel cases, split model; forward uses only m_phys
        if self.case == 'kernel':
            n_m = self._get_n_m()
            m_phys = model[:n_m]
            eta = model[n_m:]
            d_model = self.forward_func(np.asarray(m_phys), **self.fwd_kwargs)
        elif self.case == 'kernel_full':
            n_m = self._get_n_m()
            m_phys = model[:n_m]
            phi = model[n_m]           # log(sigma_d)
            eta = model[n_m + 1:]      # log(l)
            d_model = self.forward_func(np.asarray(m_phys), **self.fwd_kwargs)
        else:
            d_model = self.forward_func(np.asarray(model), **self.fwd_kwargs)

        residual = self.data - d_model

        # Start fresh cache for this model
        self._cache_model = model.copy()
        self._cache = {'residual': residual}

        # Case-specific log-likelihood computation
        if self.case == "none":
            Cd = self.Cd_ref
            Cd_inv_r = np.linalg.solve(Cd, residual)
            log_det_Cd = np.linalg.slogdet(Cd)[1]
            log_likelihood = -0.5 * (log_det_Cd + residual.dot(Cd_inv_r))
            Cd_ml = Cd
            # Cache intermediates for derivatives
            self._cache['Cd_inv_r'] = Cd_inv_r

        elif self.case == "scaled":
            Ctilde = self.Cd_ref
            Ctilde_inv = np.linalg.inv(Ctilde)
            Ctilde_inv_r = Ctilde_inv.dot(residual)
            a = residual.dot(Ctilde_inv_r)
            a = self._safe_scalar(a)
            log_likelihood = -0.5 * N * np.log(a)
            Cd_ml = (a / N) * Ctilde
            # Cache intermediates
            self._cache['Ctilde_inv'] = Ctilde_inv
            self._cache['Ctilde_inv_r'] = Ctilde_inv_r
            self._cache['a'] = a

        elif self.case == "kernel":
            eta_scalar = float(eta[0])
            K = self.kernel.evaluate(eta_scalar)
            L, low = cho_factor(K, lower=True)
            alpha = cho_solve((L, low), residual)
            a = residual.dot(alpha)
            a = self._safe_scalar(a)
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            log_likelihood = -0.5 * (log_det_K + N * np.log(a))
            Cd_ml = (a / N) * K
            # Cache intermediates
            self._cache['L'] = L
            self._cache['low'] = low
            self._cache['alpha'] = alpha
            self._cache['a'] = a
            self._cache['eta_scalar'] = eta_scalar
            self._cache['n_m'] = n_m

        elif self.case == "kernel_full":
            # Explicit sigma_d with kernel covariance:
            #   C_d = sigma_d^2 * K(eta) = exp(2*phi) * K(eta)
            # Log-likelihood:
            #   L(m, phi, eta) = -0.5 [2N*phi + log|K| + exp(-2*phi) * a]
            eta_scalar = float(eta[0])
            K = self.kernel.evaluate(eta_scalar)
            L, low = cho_factor(K, lower=True)
            alpha = cho_solve((L, low), residual)
            a = residual.dot(alpha)
            a = self._safe_scalar(a)
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            exp_neg2phi = np.exp(-2.0 * phi)
            log_likelihood = -0.5 * (2.0 * N * phi + log_det_K + exp_neg2phi * a)
            Cd_ml = np.exp(2.0 * phi) * K
            # Cache intermediates
            self._cache['L'] = L
            self._cache['low'] = low
            self._cache['alpha'] = alpha
            self._cache['a'] = a
            self._cache['phi'] = phi
            self._cache['exp_neg2phi'] = exp_neg2phi
            self._cache['eta_scalar'] = eta_scalar
            self._cache['n_m'] = n_m

        elif self.case == "spherical":
            s = residual.dot(residual)
            s = self._safe_scalar(s)
            log_likelihood = -0.5 * N * np.log(s)
            Cd_ml = np.eye(N) * (s / N)
            # Cache intermediates
            self._cache['s'] = s

        elif self.case == "diag":
            nu = self.nu
            s_param = self.s
            if nu <= 0:
                raise ValueError("Student-t parameter nu must be > 0.")
            if s_param <= 0:
                raise ValueError("Student-t scale s must be > 0.")
            a = nu * s_param * s_param
            r2 = residual * residual
            eps = self.eps
            denom = a + r2 + eps
            log_likelihood = -0.5 * (nu + 1) * np.sum(np.log(1.0 + r2 / a))
            sigma2_eff = denom / (nu + 1)
            Cd_ml = np.diag(sigma2_eff)
            # Cache intermediates for derivatives
            w = (nu + 1) * residual / denom
            d_diag = (nu + 1) * (a - r2) / (denom * denom)
            self._cache['w'] = w
            self._cache['d_diag'] = d_diag

        elif self.case == "diag_legacy":
            r_abs = np.maximum(np.abs(residual), self.eps)
            log_likelihood = -np.sum(np.log(r_abs))
            Cd_ml = np.diag(residual * residual)
            # Cache intermediates
            self._cache['r_abs'] = r_abs

        elif self.case == "full":
            s = residual.dot(residual)
            s = self._safe_scalar(s)
            log_likelihood = -0.5 * np.log(s)
            Cd_ml = np.outer(residual, residual)
            # Cache intermediates
            self._cache['s'] = s

        self._cache['log_likelihood'] = log_likelihood
        self._cache['Cd_ml'] = Cd_ml

        return log_likelihood, Cd_ml

    # ------------------------------------------------------------------
    # Tier 2: Gradient and Hessian (requires G)
    # ------------------------------------------------------------------

    def _check_G(self):
        """Raise ValueError if Jacobian G is not set."""
        if self._G is None:
            raise ValueError(
                f"Cannot compute derivatives when G=None (case='{self.case}'). "
                f"Gradient/Hessian require the Jacobian matrix. Either provide G "
                f"during initialization, assign via likelihood.G = <jacobian>, "
                f"or use derivative-free optimization."
            )

    def _evaluate_gradient(self, model: np.ndarray):
        """Compute gradient using cached intermediates from _evaluate_loglik.

        Requires G to be set.

        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        self._check_G()

        # Check cache
        if (self._cache_model is not None
                and np.array_equal(model, self._cache_model)
                and 'gradient' in self._cache):
            return self._cache['gradient']

        # Ensure log-likelihood intermediates are cached
        self._evaluate_loglik(model)

        G = self._G
        N = self.n_data
        residual = self._cache['residual']

        if self.case == "none":
            Cd_inv_r = self._cache['Cd_inv_r']
            gradient = G.T.dot(Cd_inv_r)

        elif self.case == "scaled":
            Ctilde_inv_r = self._cache['Ctilde_inv_r']
            a = self._cache['a']
            numerator = G.T.dot(Ctilde_inv_r)
            gradient = (N / a) * numerator
            self._cache['numerator'] = numerator

        elif self.case == "kernel":
            L = self._cache['L']
            low = self._cache['low']
            alpha = self._cache['alpha']
            a = self._cache['a']
            eta_scalar = self._cache['eta_scalar']

            K_eta = self.kernel.derivative(eta_scalar)
            g0 = G.T.dot(alpha)
            grad_m = (N / a) * g0

            K_inv_K_eta = cho_solve((L, low), K_eta)
            t = np.trace(K_inv_K_eta)
            b = alpha.dot(K_eta.dot(alpha))
            grad_eta = -0.5 * (t - (N / a) * b)

            gradient = np.concatenate([grad_m, np.atleast_1d(grad_eta)])

            # Cache intermediates for hessian
            self._cache['g0'] = g0
            self._cache['K_eta'] = K_eta
            self._cache['K_inv_K_eta'] = K_inv_K_eta
            self._cache['t'] = t
            self._cache['b'] = b

        elif self.case == "kernel_full":
            L = self._cache['L']
            low = self._cache['low']
            alpha = self._cache['alpha']
            a = self._cache['a']
            exp_neg2phi = self._cache['exp_neg2phi']
            eta_scalar = self._cache['eta_scalar']

            K_eta = self.kernel.derivative(eta_scalar)
            g0 = G.T.dot(alpha)
            grad_m = exp_neg2phi * g0

            grad_phi = -N + exp_neg2phi * a

            K_inv_K_eta = cho_solve((L, low), K_eta)
            t = np.trace(K_inv_K_eta)
            b = alpha.dot(K_eta.dot(alpha))
            grad_eta = -0.5 * (t - exp_neg2phi * b)

            gradient = np.concatenate([grad_m, [grad_phi], np.atleast_1d(grad_eta)])

            # Cache intermediates for hessian
            self._cache['g0'] = g0
            self._cache['K_eta'] = K_eta
            self._cache['K_inv_K_eta'] = K_inv_K_eta
            self._cache['t'] = t
            self._cache['b'] = b

        elif self.case == "spherical":
            s = self._cache['s']
            numerator = G.T.dot(residual)
            gradient = (N / s) * numerator
            self._cache['numerator'] = numerator

        elif self.case == "diag":
            w = self._cache['w']
            gradient = G.T.dot(w) if sparse.issparse(G) else (G.T @ w)

        elif self.case == "diag_legacy":
            r_abs = self._cache['r_abs']
            inv_r = np.sign(residual) / r_abs
            gradient = G.T.dot(inv_r)

        elif self.case == "full":
            s = self._cache['s']
            numerator = G.T.dot(residual)
            gradient = numerator / s
            self._cache['numerator'] = numerator

        self._cache['gradient'] = gradient
        return gradient

    def _evaluate_hessian(self, model: np.ndarray):
        """Compute hessian using cached intermediates from _evaluate_loglik
        and _evaluate_gradient.

        Requires G to be set.

        Returns
        -------
        np.ndarray
            Hessian matrix.
        """
        self._check_G()

        # Check cache
        if (self._cache_model is not None
                and np.array_equal(model, self._cache_model)
                and 'hessian' in self._cache):
            return self._cache['hessian']

        # Ensure gradient intermediates are cached
        self._evaluate_gradient(model)

        G = self._G
        N = self.n_data

        if self.case == "none":
            Cd = self.Cd_ref
            Cd_inv_G = np.linalg.solve(Cd, G)
            hessian = -G.T.dot(Cd_inv_G)

        elif self.case == "scaled":
            Ctilde_inv = self._cache['Ctilde_inv']
            a = self._cache['a']
            numerator = self._cache['numerator']
            outer = np.outer(numerator, numerator)
            GTCtildeG = G.T.dot(Ctilde_inv.dot(G))
            hessian = 2.0 * N * outer / (a * a) - N * GTCtildeG / a

        elif self.case == "kernel":
            L = self._cache['L']
            low = self._cache['low']
            alpha = self._cache['alpha']
            a = self._cache['a']
            eta_scalar = self._cache['eta_scalar']
            n_m = self._cache['n_m']
            g0 = self._cache['g0']
            K_eta = self._cache['K_eta']
            K_inv_K_eta = self._cache['K_inv_K_eta']
            b = self._cache['b']

            # H_mm
            outer_mm = np.outer(g0, g0)
            K_inv_G = cho_solve((L, low), G)
            GTKinvG = G.T.dot(K_inv_G)
            H_mm = (2.0 * N / (a * a)) * outer_mm - (N / a) * GTKinvG

            # H_m_eta
            K_eta_alpha = K_eta.dot(alpha)
            K_inv_K_eta_alpha = cho_solve((L, low), K_eta_alpha)
            H_m_eta = N * ((b / (a * a)) * g0 - (1.0 / a) * G.T.dot(K_inv_K_eta_alpha))
            H_m_eta = H_m_eta.reshape(n_m, self.kernel.n_params)

            # H_eta_eta
            K_etaeta = self.kernel.second_derivative(eta_scalar)
            K_inv_K_etaeta = cho_solve((L, low), K_etaeta)
            dt_deta = (np.trace(K_inv_K_etaeta)
                       - np.sum(K_inv_K_eta * K_inv_K_eta.T))
            db_deta = (alpha.dot(K_etaeta.dot(alpha))
                       - 2.0 * alpha.dot(K_eta.dot(K_inv_K_eta.dot(alpha))))
            H_eta_eta = -0.5 * (dt_deta
                                - N * ((1.0 / a) * db_deta + (b * b) / (a * a)))
            H_eta_eta = np.atleast_2d(H_eta_eta)

            # Assemble
            n_total = n_m + self.kernel.n_params
            hessian = np.zeros((n_total, n_total))
            hessian[:n_m, :n_m] = H_mm
            hessian[:n_m, n_m:] = H_m_eta
            hessian[n_m:, :n_m] = H_m_eta.T
            hessian[n_m:, n_m:] = H_eta_eta

        elif self.case == "kernel_full":
            L = self._cache['L']
            low = self._cache['low']
            alpha = self._cache['alpha']
            a = self._cache['a']
            exp_neg2phi = self._cache['exp_neg2phi']
            eta_scalar = self._cache['eta_scalar']
            n_m = self._cache['n_m']
            g0 = self._cache['g0']
            K_eta = self._cache['K_eta']
            K_inv_K_eta = self._cache['K_inv_K_eta']
            b = self._cache['b']

            # H_mm
            K_inv_G = cho_solve((L, low), G)
            GTKinvG = G.T.dot(K_inv_G)
            H_mm = -exp_neg2phi * GTKinvG

            # H_m_phi
            H_m_phi = -2.0 * exp_neg2phi * g0

            # H_m_eta
            K_eta_alpha = K_eta.dot(alpha)
            K_inv_K_eta_alpha = cho_solve((L, low), K_eta_alpha)
            H_m_eta = -exp_neg2phi * G.T.dot(K_inv_K_eta_alpha)
            H_m_eta = H_m_eta.reshape(n_m, self.kernel.n_params)

            # H_phi_phi
            H_phi_phi = -2.0 * exp_neg2phi * a

            # H_phi_eta
            H_phi_eta = -exp_neg2phi * b

            # H_eta_eta
            K_etaeta = self.kernel.second_derivative(eta_scalar)
            K_inv_K_etaeta = cho_solve((L, low), K_etaeta)
            dt_deta = (np.trace(K_inv_K_etaeta)
                       - np.sum(K_inv_K_eta * K_inv_K_eta.T))
            db_deta = (alpha.dot(K_etaeta.dot(alpha))
                       - 2.0 * alpha.dot(K_eta.dot(K_inv_K_eta.dot(alpha))))
            H_eta_eta = -0.5 * (dt_deta - exp_neg2phi * db_deta)
            H_eta_eta = np.atleast_2d(H_eta_eta)

            # Assemble [m, phi, eta]
            n_eta = self.kernel.n_params
            n_total = n_m + 1 + n_eta
            hessian = np.zeros((n_total, n_total))
            hessian[:n_m, :n_m] = H_mm
            hessian[:n_m, n_m] = H_m_phi
            hessian[n_m, :n_m] = H_m_phi
            hessian[:n_m, n_m + 1:] = H_m_eta
            hessian[n_m + 1:, :n_m] = H_m_eta.T
            hessian[n_m, n_m] = H_phi_phi
            hessian[n_m, n_m + 1:] = H_phi_eta
            hessian[n_m + 1:, n_m] = H_phi_eta
            hessian[n_m + 1:, n_m + 1:] = H_eta_eta

        elif self.case == "spherical":
            s = self._cache['s']
            numerator = self._cache['numerator']
            outer = np.outer(numerator, numerator)
            GTG = G.T.dot(G)
            hessian = 2.0 * N * outer / (s * s) - (N / s) * GTG

        elif self.case == "diag":
            d_diag = self._cache['d_diag']
            if sparse.issparse(G):
                D = sparse.diags(d_diag)
                hessian = -(G.T.dot(D.dot(G))).toarray()
            else:
                hessian = -(G.T @ (d_diag[:, None] * G))

        elif self.case == "diag_legacy":
            r_abs = self._cache['r_abs']
            diag_vals = 1.0 / (r_abs**2)
            if sparse.issparse(G):
                D = sparse.diags(diag_vals)
                hessian = (G.T.dot(D.dot(G))).toarray()
            else:
                hessian = G.T.dot(np.diag(diag_vals)).dot(G)

        elif self.case == "full":
            s = self._cache['s']
            numerator = self._cache['numerator']
            outer = np.outer(numerator, numerator)
            GTG = G.T.dot(G)
            hessian = 2.0 * outer / (s * s) - GTG / s

        self._cache['hessian'] = hessian
        return hessian

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_likelihood(self, model: np.ndarray) -> float:
        """Evaluate the log-likelihood at given model parameters.

        This method does **not** require the Jacobian matrix ``G`` to be set.
        """
        flat_m = self._validate_model(model)
        return self._evaluate_loglik(flat_m)[0]

    def gradient(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the gradient at given model parameters.

        Requires the Jacobian matrix ``G`` to be set. If ``G=None``, raises
        ``ValueError`` with guidance on how to proceed.
        """
        flat_m = self._validate_model(model)
        return self._evaluate_gradient(flat_m)

    def hessian(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the hessian at given model parameters.

        Requires the Jacobian matrix ``G`` to be set. If ``G=None``, raises
        ``ValueError`` with guidance on how to proceed.
        """
        flat_m = self._validate_model(model)
        return self._evaluate_hessian(flat_m)

    def get_ml_cov(self, model: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate the covariance matrix at given model parameters.

        This method does **not** require the Jacobian matrix ``G`` to be set.
        """
        flat_m = self._validate_model(model)
        return self._evaluate_loglik(flat_m)[1]

    def _validate_model(self, model):
        flat_m = np.ravel(model)
        # When model_shape is available, validate model size
        if self.model_shape is not None:
            if flat_m.size != self.model_size:
                raise DimensionMismatchError(
                    entered_name="model",
                    entered_dimension=model.shape,
                    expected_source="model_size",
                    expected_dimension=self.model_size,
                )
        # When model_shape is None (neither G nor n_params provided),
        # skip size validation — the forward function will fail if wrong.
        return flat_m
