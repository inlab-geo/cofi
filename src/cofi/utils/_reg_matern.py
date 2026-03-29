from numbers import Integral
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


def _validate_model_shape(model_shape: tuple) -> tuple:
    if len(model_shape) != 2:
        raise ValueError(
            f"SPDEMaternReg requires a 2D model_shape (n_lon, n_lat), got {model_shape}"
        )
    if any(not isinstance(n, Integral) or n <= 0 for n in model_shape):
        raise ValueError(
            "SPDEMaternReg requires positive integer grid dimensions in model_shape"
        )
    return tuple(int(n) for n in model_shape)


def _normalize_grid_spacing(grid_spacing) -> tuple:
    if np.isscalar(grid_spacing):
        h_lon = h_lat = float(grid_spacing)
    else:
        try:
            h_lon, h_lat = grid_spacing
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "grid_spacing must be a positive scalar or a length-2 sequence"
            ) from exc
        h_lon = float(h_lon)
        h_lat = float(h_lat)
    if h_lon <= 0 or h_lat <= 0:
        raise ValueError("grid_spacing values must be positive")
    return (h_lon, h_lat)


class SPDEMaternReg(QuadraticReg):
    r"""Sparse Matérn ν=1 regularization for 2D spatial fields via the SPDE approach.

    Implements the regularization term

    .. math::

        \mathcal{R}(\mathbf{m}) = \|\mathbf{R}(\mathbf{m} - \mathbf{m}_0)\|^2,
        \quad \mathbf{R} = \tau \mathbf{M}^{1/2}(\kappa^2 \mathbf{I} - \mathbf{L}_h),
        \quad \kappa = \sqrt{2} / \ell,
        \quad \tau = \frac{1}{2 \sqrt{\pi}\,\kappa\,\sigma}

    where :math:`\mathbf{L}_h` is the 2D grid-spacing-aware Neumann Laplacian,
    :math:`\mathbf{M}` is the lumped mass matrix, :math:`\ell` is the Matérn
    length scale for :math:`\nu = 1`, and :math:`\sigma` is the marginal
    standard deviation in the continuous SPDE model. For the uniform rectangular
    grids implemented here, :math:`\mathbf{M} \approx h_x h_y \mathbf{I}` so
    :math:`\mathbf{R} = \tau \sqrt{h_x h_y}(\kappa^2 \mathbf{I} - \mathbf{L}_h)`.
    The matrix :math:`\mathbf{R}` is a sparse precision factor of the Matérn ν=1
    precision matrix

    .. math::

        \mathbf{Q} = \mathbf{R}^\top \mathbf{R}
        = \tau^2 \mathbf{B}_h^\top \mathbf{M} \mathbf{B}_h,
        \quad \mathbf{B}_h = \kappa^2 \mathbf{I} - \mathbf{L}_h

    which reduces to :math:`\tau^2 (\kappa^2 \mathbf{I} - \mathbf{L})^2` on a
    unit grid. (Lindgren, Rue & Lindström 2011). The matrix :math:`\mathbf{R}`
    is sparse (:math:`O(n)` non-zeros), unlike the dense precision matrix of a
    standard Gaussian prior.

    The Matérn practical range (distance at which correlation drops to ~0.14)
    is :math:`\rho = 2\ell` for :math:`\nu = 1`.

    Parameters
    ----------
    model_shape : tuple of (int, int)
        Shape of the 2D model grid ``(n_lon, n_lat)`` (or equivalently any two
        positive integers whose product is the number of model parameters).
    ell : float
        Matérn length scale in the same physical units as ``grid_spacing``.
        For :math:`\nu=1`, this is related to the SPDE wavenumber by
        :math:`\kappa = \sqrt{2}/\ell`. The practical range (correlation ≈ 0.14)
        is :math:`\rho = 2\ell`.
    sigma : float, optional
        Prior marginal standard deviation of model perturbations. The internal SPDE
        scaling uses :math:`\tau = 1 / (2 \sqrt{\pi}\,\kappa\,\sigma)`.
        Default 1.0.
    grid_spacing : float or tuple of (float, float), optional
        Physical spacing of the regular grid in the longitude/x and latitude/y
        directions. A scalar applies the same spacing to both axes.
        ``grid_spacing=1.0`` reproduces the current unit-grid interpretation.
    reference_model : np.ndarray, optional
        Background model :math:`\mathbf{m}_0`. If provided, the penalty is on
        :math:`\mathbf{m} - \mathbf{m}_0`; if omitted, the penalty is on
        :math:`\mathbf{m}` directly.

    Notes
    -----
    Boundary conditions
        The discrete Laplacian uses **Neumann (zero-flux)** boundary conditions.
        This means no smoothness penalty is imposed across the grid edge, so
        boundary cells are treated identically to interior cells. Dirichlet
        (zero-endpoint) boundaries would artificially anchor boundary cells to
        the reference model and are not appropriate for most geophysical grids.
        As in the SPDE literature, these boundary conditions inflate marginal
        variances near the edges of a finite grid.

    Examples
    --------
    >>> from cofi.utils import SPDEMaternReg
    >>> import numpy as np
    >>> reg = SPDEMaternReg(model_shape=(10, 8), ell=2.0, sigma=0.02)
    >>> reg(np.zeros((10, 8)))
    0.0
    """

    def __init__(
        self,
        model_shape: tuple,
        ell: float,
        sigma: float = 1.0,
        grid_spacing=1.0,
        reference_model: Optional[np.ndarray] = None,
    ):
        model_shape = _validate_model_shape(model_shape)
        if ell <= 0:
            raise ValueError("ell must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        n_lon, n_lat = model_shape
        n_params = n_lon * n_lat
        h_lon, h_lat = _normalize_grid_spacing(grid_spacing)
        kappa = np.sqrt(2.0) / ell
        kappa2 = kappa**2
        tau = 1.0 / (2.0 * np.sqrt(np.pi) * kappa * sigma)

        rho = 2.0 * ell  # practical range (correlation ≈ 0.14)
        max_physical_dim = max(n_lon * h_lon, n_lat * h_lat)
        if rho > 0.5 * max_physical_dim:
            warnings.warn(
                f"ell={ell} (practical range rho=2*ell={rho:.1f}) exceeds half the "
                f"longest physical grid dimension ({0.5 * max_physical_dim:.1f}). "
                "This heuristic indicates that the precision factor will weakly "
                "constrain the longest-wavelength modes, which may cause solver "
                "convergence issues on a finite grid.",
                UserWarning,
                stacklevel=2,
            )

        # 1D tridiagonal Laplacian operators (Neumann boundary conditions)
        L1d_lat = _neumann_laplacian_1d(n_lat)
        L1d_lon = _neumann_laplacian_1d(n_lon)

        # 2D Laplacian via Kronecker products
        L_full = (
            sparse.kron(
                sparse.eye(n_lon, format="csr"),
                L1d_lat / h_lat**2,
                format="csr",
            )
            + sparse.kron(
                L1d_lon / h_lon**2,
                sparse.eye(n_lat, format="csr"),
                format="csr",
            )
        )

        # Lumped mass M ≈ h_lon * h_lat * I for a uniform rectangular grid.
        mass_sqrt = np.sqrt(h_lon * h_lat)

        # Sparse precision factor R = tau * M^{1/2} * (kappa^2 I - L_h)
        R = (
            tau
            * mass_sqrt
            * (kappa2 * sparse.eye(n_params, format="csr") - L_full)
        )

        # Flatten reference model if provided
        ref = np.ravel(reference_model) if reference_model is not None else None

        super().__init__(
            weighting_matrix=R,
            model_shape=(n_params,),
            reference_model=ref,
        )

        # Store parameters for inspection
        self._ell = float(ell)
        self._sigma = sigma
        self._tau = tau
        self._kappa2 = kappa2
        self._2d_shape = model_shape
        self._grid_spacing = (h_lon, h_lat)

    @property
    def ell(self) -> float:
        """Matérn length scale."""
        return self._ell

    @property
    def rho(self) -> float:
        """Matérn practical range (rho = 2 * ell for nu=1)."""
        return 2.0 * self._ell

    @property
    def sigma(self) -> float:
        """Prior marginal standard deviation."""
        return self._sigma

    @property
    def kappa(self) -> float:
        """Wavenumber parameter κ = sqrt(2) / ell for ν = 1."""
        return np.sqrt(self._kappa2)

    @property
    def grid_shape(self) -> tuple:
        """Original 2D grid shape (n_lon, n_lat)."""
        return self._2d_shape

    @property
    def grid_spacing(self) -> tuple:
        """Physical grid spacing (h_lon, h_lat)."""
        return self._grid_spacing
