"""Reduced likelihood implementation for various covariance cases."""

from typing import Optional, Callable, Tuple
import numpy as np
from scipy import sparse

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
    - **'spherical'**: Spherical covariance (diagonal with same variance)
    - **'diag'**: Diagonal covariance with different variances
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
        Jacobian matrix of shape (n_data, n_params). Must be updated
        before each evaluation for non-linear problems.
    Cd_ref : Optional[np.ndarray], default=None
        Reference covariance matrix. Required for 'scaled' case.
        For 'none' case: if not provided, defaults to identity matrix
        (assumes uncorrelated, unit-variance noise).
        Not used for 'spherical', 'diag', or 'full' cases.
    case : str, default='none'
        The covariance case: 'none', 'scaled', 'spherical', 'diag', or 'full'
    eps : float, default=1e-154
        Small number for numerical stability

    Raises
    ------
    ValueError
        If Cd_ref is not provided for 'scaled' case
        If case is not one of the supported options
        If G is not set before evaluation

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
    """

    def __init__(
        self,
        data: np.ndarray,
        forward_func: Callable,
        fwd_kwargs: dict = None,
        G: Optional[np.ndarray] = None,
        Cd_ref: Optional[np.ndarray] = None,
        case: str = "none",
        eps: float = EPS,
    ):
        """Initialize the reduced likelihood."""
        super().__init__()
        self.data = np.asarray(data)
        self.n_data = self.data.size
        self.forward_func = forward_func
        self.fwd_kwargs = fwd_kwargs if fwd_kwargs is not None else {}
        if G is None:
            self.G = None
        elif sparse.issparse(G):
            # keep sparse matrices as-is (so .shape and sparse operations work)
            self.G = G
        else:
            self.G = np.asarray(G)
        self._model_shape = None  # Will be inferred from G when set
        self.Cd_ref = None if Cd_ref is None else np.asarray(Cd_ref)
        self.case = case.lower()
        self.eps = float(eps)

        # Validate inputs and set defaults
        if self.case == 'scaled' and self.Cd_ref is None:
            raise ValueError(
                f"Cd_ref is required for case='scaled'. "
                f"Provide a reference covariance matrix."
            )
        elif self.case == 'none' and self.Cd_ref is None:
            # Default to identity matrix (uncorrelated, unit variance noise)
            self.Cd_ref = np.eye(self.n_data)

        if self.case not in ['none', 'scaled', 'spherical', 'diag', 'full']:
            raise ValueError(
                f"Unknown case '{self.case}'. Must be one of: "
                "'none', 'scaled', 'spherical', 'diag', 'full'"
            )

        # Cache for evaluation results to avoid redundant computation
        self._cache = {}
        self._cache_model = None

    @property
    def model_shape(self) -> tuple:
        """Return the model shape."""
        if self._model_shape is None and self.G is not None:
            self._model_shape = (self.G.shape[1],)
        return self._model_shape

    def _safe_scalar(self, x: float) -> float:
        """Ensure scalar is above numerical floor."""
        return max(x, self.eps)

    def _evaluate(self, model: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Evaluate log-likelihood and its derivatives.

        Returns
        -------
        tuple
            (log_likelihood, gradient, hessian, Cd_ml)
        """
        # Check that G has been set
        if self.G is None:
            raise ValueError("Jacobian matrix G must be set before evaluation")

        # Check cache
        if self._cache_model is not None and np.array_equal(model, self._cache_model):
            return (
                self._cache['log_likelihood'],
                self._cache['gradient'],
                self._cache['hessian'],
                self._cache.get('Cd_ml')
            )

        # Compute forward model and residual
        d_model = self.forward_func(np.asarray(model), **self.fwd_kwargs)
        residual = self.data - d_model
        N = self.n_data
        G = self.G  # Use the Jacobian matrix

        # Compute based on case
        if self.case == "none":
            # Standard Gaussian likelihood with fixed covariance
            Cd = self.Cd_ref
            Cd_inv_r = np.linalg.solve(Cd, residual)

            log_det_Cd = np.linalg.slogdet(Cd)[1]
           ## log_likelihood = -0.5 * (N * np.log(2 * np.pi) + log_det_Cd + residual.dot(Cd_inv_r))
            log_likelihood = -0.5 * (log_det_Cd + residual.dot(Cd_inv_r))
            gradient = G.T.dot(Cd_inv_r)

            Cd_inv_G = np.linalg.solve(Cd, G)
            hessian = -G.T.dot(Cd_inv_G)

            Cd_ml = Cd

        elif self.case == "scaled":
            # Scaled reference covariance: Cd = a * Cd_ref
            Ctilde = self.Cd_ref
            Ctilde_inv = np.linalg.inv(Ctilde)
            Ctilde_inv_r = Ctilde_inv.dot(residual)

            a = residual.dot(Ctilde_inv_r)
            a = self._safe_scalar(a)

            log_likelihood = -0.5 * N * np.log(a)

            numerator = G.T.dot(Ctilde_inv_r)
            gradient = (N / a) * numerator

            outer = np.outer(numerator, numerator)
            GTCtildeG = G.T.dot(Ctilde_inv.dot(G))
            hessian = 2.0 * N * outer / (a * a) - N * GTCtildeG / a

            Cd_ml = (a / N) * Ctilde

        elif self.case == "spherical":
            # Spherical covariance: Cd = s * I
            s = residual.dot(residual)
            s = self._safe_scalar(s)

            log_likelihood = -0.5 * N * np.log(s)

            numerator = G.T.dot(residual)
            gradient = (N / s) * numerator

            outer = np.outer(numerator, numerator)
            GTG = G.T.dot(G)
            hessian = 2.0 * N * outer / (s * s) - (N / s) * GTG

            Cd_ml = np.eye(N) * (s / N)

        elif self.case == "diag":
            # Diagonal covariance with different variances
            r_abs = np.maximum(np.abs(residual), self.eps)

            log_likelihood = -np.sum(np.log(r_abs))

            inv_r = np.sign(residual) / r_abs
            gradient = G.T.dot(inv_r)

            diag_vals = 1.0 / (r_abs**2)
            if sparse.issparse(G):
                D = sparse.diags(diag_vals)
                hessian = (G.T.dot(D.dot(G))).toarray()
            else:
                hessian = G.T.dot(np.diag(diag_vals)).dot(G)

            Cd_ml = np.diag(residual * residual)

        elif self.case == "full":
            # Full covariance estimation
            s = residual.dot(residual)
            s = self._safe_scalar(s)

            log_likelihood = -0.5 * np.log(s)

            numerator = G.T.dot(residual)
            gradient = numerator / s

            outer = np.outer(numerator, numerator)
            GTG = G.T.dot(G)
            hessian = 2.0 * outer / (s * s) - GTG / s

            Cd_ml = np.outer(residual, residual)

        # Cache results
        self._cache_model = model.copy()
        self._cache = {
            'log_likelihood': log_likelihood,
            'gradient': gradient,
            'hessian': hessian,
            'Cd_ml': Cd_ml
        }

        return log_likelihood, gradient, hessian, Cd_ml

    def log_likelihood(self, model: np.ndarray) -> float:
        """Evaluate the log-likelihood at given model parameters."""
        flat_m = self._validate_model(model)
        return self._evaluate(flat_m)[0]

    def gradient(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the gradient at given model parameters."""
        flat_m = self._validate_model(model)
        return self._evaluate(flat_m)[1]

    def hessian(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the hessian at given model parameters."""
        flat_m = self._validate_model(model)
        return self._evaluate(flat_m)[2]

    def get_ml_cov(self, model: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate the covariance matrix at given model parameters."""
        flat_m = self._validate_model(model)
        return self._evaluate(flat_m)[3]

    def _validate_model(self, model):
        flat_m = np.ravel(model)
        # If G wasn't supplied we cannot infer the expected model shape/size.
        # Raise a clear error instead of letting model_size trigger a confusing TypeError.
        if self.model_shape is None:
            raise ValueError("Jacobian matrix G must be set before evaluation")
        if flat_m.size != self.model_size:
            raise DimensionMismatchError(
                entered_name="model",
                entered_dimension=model.shape,
                expected_source="model_size",
                expected_dimension=self.model_size,
            )
        return flat_m