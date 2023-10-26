# Need a Parameterisation class to abstract away imported parameterisations
# and allow for easy switching between them.

import numpy as np
from octo.basis import CosineBasis2D as OctoCosineBasis2D


class Parameterisation:
    def __init__(self) -> None:
        self.basis = None  # Matrix of basis functions. Each column is a basis function

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Expand coefficients X in terms of basis functions

        .. math::

            f(x) = \sum_{i=0}^{N-1} X_i \phi_i(x)

        :param X: vector of N coefficients
        """
        # self.basis is a matrix of shape (N, N) with each column representing a basis function
        return self.basis @ X

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Get the ith basis function

        :param i: index of basis function
        """
        return self.basis[:, i]


class Identity(Parameterisation):
    def __init__(self, N: int) -> None:
        """
        Identity parameterisation

        :param N: number of basis functions
        """
        super().__init__()
        self.basis = np.eye(N)


class CosineBasis2D(OctoCosineBasis2D, Parameterisation):
    # Inheritence order important here.
    # super will prioritise features of OctoCosineBasis2D over Parameterisation
    def __init__(self, Nx, Ny: int) -> None:
        """
        Cosine basis parameterisation

        :param Nx: number of basis functions in x
        :param Ny: number of basis functions in y
        """
        super().__init__(Nx, Ny)
        self._create_basis()
