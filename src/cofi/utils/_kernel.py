"""Kernel functions for parameterized covariance matrices."""

import numpy as np


class SquaredExponentialKernel:
    r"""Squared exponential (Gaussian/RBF) correlation kernel.

    Computes the kernel matrix and its first and second derivatives with
    respect to :math:`\eta = \log \ell`, where :math:`\ell` is the
    correlation length.

    .. math::

        K(\eta)_{ij} = \exp\!\Big(-\frac{d_{ij}^2}{2\ell^2}\Big) + \nu\,\delta_{ij}

    where :math:`\ell = e^\eta`, :math:`d_{ij} = |x_i - x_j|`, and
    :math:`\nu` is a small nugget for numerical stability.

    Parameters
    ----------
    positions : array_like
        1-D array of data point positions (e.g. time samples).
        Shape ``(n_data,)``.
    nugget : float, optional
        Small diagonal addition for positive-definiteness. Default ``1e-10``.

    Attributes
    ----------
    n_params : int
        Number of kernel hyperparameters (always 1 for this kernel).
    n_data : int
        Number of data points.
    D2 : np.ndarray
        Precomputed squared-distance matrix, shape ``(n_data, n_data)``.
    """

    n_params = 1

    def __init__(self, positions, nugget=1e-10):
        positions = np.asarray(positions, dtype=float).ravel()
        self.positions = positions
        self.n_data = positions.size
        self.nugget = float(nugget)
        # Precompute squared-distance matrix (symmetric, zero diagonal)
        self.D2 = (positions[:, None] - positions[None, :]) ** 2

    # ------------------------------------------------------------------
    # Internal helper – shared by evaluate / derivative / second_derivative
    # ------------------------------------------------------------------
    def _U_and_K0(self, eta):
        """Return (U, K0) where U = D2/(2 ell^2) and K0 = exp(-U)."""
        ell2 = np.exp(2.0 * float(eta))
        U = self.D2 / (2.0 * ell2)
        K0 = np.exp(-U)
        return U, K0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, eta):
        r"""Kernel matrix :math:`K(\eta)`.

        Parameters
        ----------
        eta : float
            Log-correlation-length, :math:`\eta = \log \ell`.

        Returns
        -------
        K : np.ndarray, shape (n_data, n_data)
        """
        _, K0 = self._U_and_K0(eta)
        K = K0.copy()
        np.fill_diagonal(K, K.diagonal() + self.nugget)
        return K

    def derivative(self, eta):
        r"""First derivative :math:`\partial K / \partial \eta`.

        .. math::

            \frac{\partial K_{ij}}{\partial\eta}
            = 2\,U_{ij}\,K^0_{ij}

        where :math:`U_{ij} = d_{ij}^2/(2\ell^2)` and
        :math:`K^0_{ij} = e^{-U_{ij}}` (without nugget).

        Parameters
        ----------
        eta : float

        Returns
        -------
        K_eta : np.ndarray, shape (n_data, n_data)
        """
        U, K0 = self._U_and_K0(eta)
        return 2.0 * U * K0

    def second_derivative(self, eta):
        r"""Second derivative :math:`\partial^2 K / \partial \eta^2`.

        .. math::

            \frac{\partial^2 K_{ij}}{\partial\eta^2}
            = 4\,U_{ij}\,(U_{ij} - 1)\,K^0_{ij}

        Parameters
        ----------
        eta : float

        Returns
        -------
        K_etaeta : np.ndarray, shape (n_data, n_data)
        """
        U, K0 = self._U_and_K0(eta)
        return 4.0 * U * (U - 1.0) * K0
