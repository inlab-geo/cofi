from typing import Optional
import warnings

import numpy as np
from scipy import sparse

from ._reg_lp_norm import QuadraticReg


def _neumann_laplacian_1d(n: int) -> sparse.csr_matrix:
    """1D tridiagonal Laplacian with Neumann (zero-gradient) boundary conditions."""
    L = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="lil")
    L[0, 0] = -1.0
    L[0, 1] = 1.0
    L[-1, -2] = 1.0
    L[-1, -1] = -1.0
    return L.tocsr()


class SPDEMaternReg(QuadraticReg):
    r"""Sparse Matérn ν=1 regularization for 2D spatial fields via the SPDE approach.

    Implements the regularization term

    .. math::

        \mathcal{R}(\mathbf{m}) = \|\mathbf{R}(\mathbf{m} - \mathbf{m}_0)\|^2,
        \quad \mathbf{R} = \frac{1}{\sigma}(\kappa^2 \mathbf{I} - \mathbf{L}),
        \quad \kappa = 1 / L_{\text{corr}}

    where :math:`\mathbf{L}` is the 2D discrete Laplacian. This is the Cholesky-like
    factor of the Matérn ν=1 precision matrix
    :math:`Q = \mathbf{R}^\top \mathbf{R} = \sigma^{-2}(\kappa^2\mathbf{I} - \mathbf{L})^2`
    (Lindgren, Rue & Lindström 2011). The matrix :math:`\mathbf{R}` is sparse
    (:math:`O(n)` non-zeros), unlike the dense precision matrix of a standard
    Gaussian prior.

    Parameters
    ----------
    model_shape : tuple of (int, int)
        Shape of the 2D model grid ``(n_lon, n_lat)`` (or equivalently any two
        positive integers whose product is the number of model parameters).
    L_corr : float
        Correlation length in grid cells. The Matérn ν=1 correlation function
        ``ρ(r) = (r/L_corr) K₁(r/L_corr)`` reaches ρ ≈ 0.60 at r = L_corr.
        Should be smaller than roughly half the shortest grid dimension; larger
        values make the precision matrix near-singular (see Notes).
    sigma : float, optional
        Prior marginal standard deviation of model perturbations, in the same
        units as the model. Default 1.0 (dimensionless).
    reference_model : np.ndarray, optional
        Background model :math:`\mathbf{m}_0`. If provided, the penalty is on
        :math:`\mathbf{m} - \mathbf{m}_0`; if omitted, the penalty is on
        :math:`\mathbf{m}` directly.

    Notes
    -----
    Boundary conditions
        The discrete Laplacian uses **Neumann (zero-flux)** boundary conditions.
        This means no smoothness penalty is imposed across the grid edge, so
        boundary cells are treated identically to interior cells.  Dirichlet
        (zero-endpoint) boundaries would artificially anchor boundary cells to
        the reference model and are not appropriate for most geophysical grids.

    Examples
    --------
    >>> from cofi.utils import SPDEMaternReg
    >>> import numpy as np
    >>> reg = SPDEMaternReg(model_shape=(10, 8), L_corr=3.0, sigma=0.02)
    >>> reg(np.zeros(80))
    0.0
    """

    def __init__(
        self,
        model_shape: tuple,
        L_corr: float,
        sigma: float = 1.0,
        reference_model: Optional[np.ndarray] = None,
    ):
        if len(model_shape) != 2:
            raise ValueError(
                f"SPDEMaternReg requires a 2D model_shape (n_lon, n_lat), "
                f"got {model_shape}"
            )
        n_lon, n_lat = model_shape
        n_params = n_lon * n_lat
        kappa2 = 1.0 / L_corr ** 2

        max_dim = max(n_lon, n_lat)
        if L_corr > 0.5 * max_dim:
            warnings.warn(
                f"L_corr={L_corr} exceeds half the grid dimension ({0.5 * max_dim:.1f}). "
                "The precision matrix R = (κ²I − L)/σ will be near-singular, which may "
                "cause solver convergence issues. Consider reducing L_corr or switching to "
                "a GaussianPrior with an explicit covariance matrix.",
                UserWarning,
                stacklevel=2,
            )

        # 1D tridiagonal Laplacian operators (Neumann boundary conditions)
        L1d_lat = _neumann_laplacian_1d(n_lat)
        L1d_lon = _neumann_laplacian_1d(n_lon)

        # 2D Laplacian via Kronecker products
        L_full = (
            sparse.kron(sparse.eye(n_lon), L1d_lat, format="csr")
            + sparse.kron(L1d_lon, sparse.eye(n_lat), format="csr")
        )

        # Sparse precision factor R = (κ²I − L) / σ
        R = (kappa2 * sparse.eye(n_params, format="csr") - L_full) / sigma

        # Flatten reference model if provided
        ref = np.ravel(reference_model) if reference_model is not None else None

        super().__init__(
            weighting_matrix=R,
            model_shape=(n_params,),
            reference_model=ref,
        )

        # Store parameters for inspection
        self._L_corr = L_corr
        self._sigma = sigma
        self._kappa2 = kappa2
        self._2d_shape = model_shape

    @property
    def L_corr(self) -> float:
        """Correlation length in grid cells."""
        return self._L_corr

    @property
    def sigma(self) -> float:
        """Prior marginal standard deviation."""
        return self._sigma

    @property
    def kappa(self) -> float:
        """Wavenumber parameter κ = 1/L_corr."""
        return 1.0 / self._L_corr

    @property
    def grid_shape(self) -> tuple:
        """Original 2D grid shape (n_lon, n_lat)."""
        return self._2d_shape
