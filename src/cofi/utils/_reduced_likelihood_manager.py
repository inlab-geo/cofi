"""Manager class for efficiently handling multiple ReducedLikelihood instances."""

from typing import List, Tuple, Callable, Optional, Union
import numpy as np
from scipy import sparse

from ._reduced_likelihood import ReducedLikelihood


class ReducedLikelihoodManager:
    r"""Manager for multiple ReducedLikelihood instances with shared Jacobian computation.

    This class manages multiple ReducedLikelihood objects and provides efficient
    evaluation of the combined objective function, gradient, and Hessian by caching
    Jacobian matrices to avoid redundant computation.

    Parameters
    ----------
    fwd_funcs : List[Tuple[Callable, dict]]
        List of (forward_function, fwd_kwargs) tuples for each dataset
    d_obs_list : List[np.ndarray]
        List of observed data arrays, one per forward function
    jacobian_fn : Callable
        Function with signature: jacobian_fn(model, n_data, fwd_func, fwd_kwargs) -> np.ndarray
        Should return Jacobian matrix of shape (n_data, n_params)
    cases : Union[str, List[str]], default='none'
        Covariance case(s). Either a single string applied to all, or a list
        with one case per dataset. Options: 'none', 'scaled', 'kernel',
        'spherical', 'diag', 'full'
    Cd_refs : Optional[Union[np.ndarray, List[np.ndarray]]], default=None
        Reference covariance matrix(ces). Either a single matrix applied to all,
        or a list with one per dataset. Required for 'scaled' case.
        For 'none' case: defaults to identity matrix if not provided.
        Not used for 'spherical', 'diag', or 'full' cases.
    kernels : optional
        Kernel instance(s) for 'kernel' case. Either a single kernel
        applied to all datasets using 'kernel', or a list with one per
        dataset. Required for datasets with ``case='kernel'``.
    copy_G : bool, default=False
        If True, copy Jacobian matrices before assigning to ReducedLikelihood.G.
        Use if jacobian_fn returns views that may be mutated.
    track_stats : bool, default=False
        If True, track statistics on cache hits and Jacobian evaluations

    Attributes
    ----------
    reduced_likelihoods : List[ReducedLikelihood]
        The managed ReducedLikelihood instances
    stats : dict
        Statistics dictionary (if track_stats=True) with keys:
        - 'n_cache_hits': Number of cache hits
        - 'n_jacobian_evals': Total Jacobian evaluations across all datasets

    Examples
    --------
    >>> import numpy as np
    >>> from cofi.utils import ReducedLikelihoodManager
    >>>
    >>> # Define forward functions and data
    >>> def fwd1(m): return np.array([m[0], m[1]])
    >>> def fwd2(m): return np.array([m[0] + m[1]])
    >>> fwd_funcs = [(fwd1, {}), (fwd2, {})]
    >>> d_obs_list = [np.array([1.0, 2.0]), np.array([3.0])]
    >>>
    >>> # Define Jacobian function
    >>> def jacobian(m, n_data, fwd, fwd_kwargs):
    ...     # Compute Jacobian (simplified example)
    ...     if n_data == 2:
    ...         return np.array([[1, 0], [0, 1]])
    ...     else:
    ...         return np.array([[1, 1]])
    >>>
    >>> # Create manager
    >>> manager = ReducedLikelihoodManager(
    ...     fwd_funcs=fwd_funcs,
    ...     d_obs_list=d_obs_list,
    ...     jacobian_fn=jacobian,
    ...     cases=['spherical', 'spherical'],
    ...     track_stats=True
    ... )
    >>>
    >>> # Evaluate at a model
    >>> model = np.array([1.5, 2.5])
    >>> obj = manager.objective(model)
    >>> grad = manager.gradient(model)
    >>> hess = manager.hessian(model)
    >>>
    >>> # Check statistics
    >>> print(manager.stats)
    """

    def __init__(
        self,
        fwd_funcs: List[Tuple[Callable, dict]],
        d_obs_list: List[np.ndarray],
        jacobian_fn: Callable,
        cases: Union[str, List[str]] = 'none',
        Cd_refs: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        kernels=None,
        copy_G: bool = False,
        track_stats: bool = False,
    ):
        """Initialize the ReducedLikelihoodManager."""
        self.fwd_funcs = fwd_funcs
        self.d_obs_list = d_obs_list
        self.jacobian_fn = jacobian_fn
        self.copy_G = copy_G
        self.track_stats = track_stats

        # Validate inputs
        if len(fwd_funcs) != len(d_obs_list):
            raise ValueError(
                f"fwd_funcs and d_obs_list must have same length, "
                f"got {len(fwd_funcs)} and {len(d_obs_list)}"
            )

        n_datasets = len(d_obs_list)

        # Handle cases parameter (single or list)
        if isinstance(cases, str):
            cases_list = [cases] * n_datasets
        else:
            cases_list = list(cases)
            if len(cases_list) != n_datasets:
                raise ValueError(
                    f"cases list length {len(cases_list)} does not match "
                    f"number of datasets {n_datasets}"
                )

        # Handle Cd_refs parameter (single, list, or None)
        if Cd_refs is None:
            Cd_refs_list = [None] * n_datasets
        elif isinstance(Cd_refs, np.ndarray):
            Cd_refs_list = [Cd_refs] * n_datasets
        else:
            Cd_refs_list = list(Cd_refs)
            if len(Cd_refs_list) != n_datasets:
                raise ValueError(
                    f"Cd_refs list length {len(Cd_refs_list)} does not match "
                    f"number of datasets {n_datasets}"
                )
            
        # Handle kernels parameter (single, list, or None)
        if kernels is None:
            kernels_list = [None] * n_datasets
        elif isinstance(kernels, list):
            kernels_list = kernels
            if len(kernels_list) != n_datasets:
                raise ValueError(
                    f"kernels list length {len(kernels_list)} does not match "
                    f"number of datasets {n_datasets}"
                )
        else:
            # Single kernel instance applied to all kernel datasets
            kernels_list = [kernels] * n_datasets

        # Validate that 'scaled' case has Cd_ref and 'kernel' has kernel
        for i, case in enumerate(cases_list):
            if case == 'scaled' and Cd_refs_list[i] is None:
                raise ValueError(
                    f"Case 'scaled' for dataset {i} requires a Cd_ref. "
                    f"Pass Cd_refs argument with a reference covariance matrix."
                )
            if case == 'kernel' and kernels_list[i] is None:
                raise ValueError(
                    f"Case 'kernel' for dataset {i} requires a kernel. "
                    f"Pass kernels argument with a kernel instance."
                )

        # Create ReducedLikelihood instances
        self.reduced_likelihoods = []
        for (fwd, fwd_kwargs), d_obs, case, Cd_ref, kernel in zip(
            fwd_funcs, d_obs_list, cases_list, Cd_refs_list, kernels_list
        ):
            rl = ReducedLikelihood(
                data=d_obs,
                forward_func=fwd,
                fwd_kwargs=fwd_kwargs,
                G=None,  # Will be set by _ensure_jacobians
                Cd_ref=Cd_ref,
                case=case,
                kernel=kernel,
            )
            self.reduced_likelihoods.append(rl)

        # Precompute total kernel hyperparameters for model slicing
        self._n_eta = sum(
            getattr(rl.kernel, 'n_params', 0)
            for rl in self.reduced_likelihoods
            if rl.case == 'kernel'
        )
        self._has_kernel = self._n_eta > 0

        # Cache management
        self._cache_valid = False
        self._last_model = None

        # Statistics tracking
        if self.track_stats:
            self.stats = {
                'n_cache_hits': 0,
                'n_jacobian_evals': 0,
            }

    def _models_equal(self, m1, m2):
        """Check if two models are equal."""
        if m1 is None or m2 is None:
            return False
        return np.array_equal(m1, m2)

    def _get_m_phys(self, m):
        """Return the physical model parameters (strip kernel hyperparameters)."""
        if self._has_kernel:
            return m[:len(m) - self._n_eta]
        return m

    def _get_model_for_rl(self, m, rl):
        """Return the appropriate model slice for a given ReducedLikelihood."""
        if rl.case == 'kernel':
            return m  # full [m_phys, eta]
        else:
            return self._get_m_phys(m)  # physical model only

    def _ensure_jacobians(self, model):
        """Ensure Jacobian matrices are computed and cached for the given model.

        This method checks the cache and only recomputes Jacobians if the model
        has changed. Jacobians are computed in a two-phase process to ensure
        cache validity is only set after all computations succeed.
        """
        m = np.asarray(model).ravel()

        # Check cache
        if self._cache_valid and self._models_equal(self._last_model, m):
            if self.track_stats:
                self.stats['n_cache_hits'] += 1
            return

        # Jacobian is computed w.r.t. physical model only (not kernel hyperparams)
        m_phys = self._get_m_phys(m)

        # Compute all Jacobians first (collect before assigning to protect cache)
        computed = []
        for i,(rl, (fwd, fwd_kwargs), d_obs) in enumerate(zip(
            self.reduced_likelihoods, self.fwd_funcs, self.d_obs_list
        )):
            try:
                G = self.jacobian_fn(m_phys, len(d_obs), fwd, fwd_kwargs)
            except Exception as e:
                raise RuntimeError(f"Jacobian failed for dataset {i}: {e}") from e
            computed.append((rl, G))

        # Assign Jacobians only after all computations succeed
        for rl, G in computed:
            if self.copy_G:
                if sparse.issparse(G):
                    rl.G = G.copy()
                else:
                    rl.G = np.asarray(G).copy()
            else:
                rl.G = G
            rl._cache_model = None
            rl._cache.clear()

        # Update stats and cache only after success
        if self.track_stats:
            self.stats['n_jacobian_evals'] += len(computed)
        self._last_model = m.copy()
        self._cache_valid = True

    def objective(self, model: np.ndarray) -> float:
        """Evaluate the combined objective function (negative log-likelihood).

        Parameters
        ----------
        model : np.ndarray
            Model parameters (``[m_phys, eta]`` when using ``kernel``)

        Returns
        -------
        float
            Negative log-likelihood value
        """
        self._ensure_jacobians(model)
        m = np.asarray(model).ravel()
        neg_log_like = 0.0
        for rl in self.reduced_likelihoods:
            neg_log_like += -rl.log_likelihood(self._get_model_for_rl(m, rl))
        return neg_log_like

    def gradient(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the gradient of the objective function.

        Parameters
        ----------
        model : np.ndarray
            Model parameters (``[m_phys, eta]`` when using ``kernel``)

        Returns
        -------
        np.ndarray
            Gradient vector
        """
        self._ensure_jacobians(model)
        m = np.asarray(model).ravel()
        n_total = m.size
        g = np.zeros(n_total, dtype=float)
        for rl in self.reduced_likelihoods:
            m_rl = self._get_model_for_rl(m, rl)
            grad_rl = -rl.gradient(m_rl)
            if grad_rl.size < n_total:
                # Non-kernel dataset: contributes only to m_phys part
                g[:grad_rl.size] += grad_rl
            else:
                g += grad_rl
        return g

    def hessian(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the Hessian of the objective function.

        Parameters
        ----------
        model : np.ndarray
            Model parameters (``[m_phys, eta]`` when using ``kernel``)

        Returns
        -------
        np.ndarray
            Hessian matrix
        """
        self._ensure_jacobians(model)
        m = np.asarray(model).ravel()
        n_total = m.size
        H = np.zeros((n_total, n_total), dtype=float)
        for rl in self.reduced_likelihoods:
            m_rl = self._get_model_for_rl(m, rl)
            hess_rl = -rl.hessian(m_rl)
            n_rl = hess_rl.shape[0]
            if n_rl < n_total:
                # Non-kernel dataset: contributes only to m_phys block
                H[:n_rl, :n_rl] += hess_rl
            else:
                H += hess_rl
        return H

    def get_ml_covs(self, model: np.ndarray) -> List[Optional[np.ndarray]]:
        """Get maximum likelihood covariance estimates for all datasets.

        Parameters
        ----------
        model : np.ndarray
            Model parameters (``[m_phys, eta]`` when using ``kernel``)

        Returns
        -------
        List[Optional[np.ndarray]]
            List of covariance matrices, one per dataset
        """
        self._ensure_jacobians(model)
        m = np.asarray(model).ravel()
        return [rl.get_ml_cov(self._get_model_for_rl(m, rl))
                for rl in self.reduced_likelihoods]

    def invalidate_cache(self):
        """Manually invalidate the Jacobian cache.

        This forces recomputation on the next evaluation. Useful if the
        forward function or Jacobian computation has changed.
        """
        self._cache_valid = False
        self._last_model = None
        for rl in self.reduced_likelihoods:
            rl._cache_model = None
            rl._cache.clear()

    def get_stats(self) -> dict:
        """Get performance statistics (if track_stats=True).

        Returns
        -------
        dict
            Statistics dictionary with keys 'n_cache_hits' and 'n_jacobian_evals'

        Raises
        ------
        ValueError
            If track_stats was not enabled at initialization
        """
        if not self.track_stats:
            raise ValueError("Statistics tracking was not enabled. Set track_stats=True.")
        return self.stats.copy()
