import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.special import k1

from cofi.utils import SPDEMaternReg


def _neumann_laplacian_1d_dense(n):
    laplacian = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            laplacian[i, i] = -1.0
            laplacian[i, i + 1] = 1.0
        elif i == n - 1:
            laplacian[i, i - 1] = 1.0
            laplacian[i, i] = -1.0
        else:
            laplacian[i, i - 1] = 1.0
            laplacian[i, i] = -2.0
            laplacian[i, i + 1] = 1.0
    return laplacian


def _expected_precision_factor(model_shape, ell, sigma, grid_spacing=(1.0, 1.0)):
    n_lon, n_lat = model_shape
    h_lon, h_lat = grid_spacing
    kappa = np.sqrt(2.0) / ell
    tau = 1.0 / (2.0 * np.sqrt(np.pi) * kappa * sigma)
    l1d_lon = _neumann_laplacian_1d_dense(n_lon)
    l1d_lat = _neumann_laplacian_1d_dense(n_lat)
    laplacian = np.kron(np.eye(n_lon), l1d_lat / h_lat**2) + np.kron(
        l1d_lon / h_lon**2, np.eye(n_lat)
    )
    return (
        tau
        * np.sqrt(h_lon * h_lat)
        * (kappa**2 * np.eye(n_lon * n_lat) - laplacian)
    )


def _index(i, j, n_lat):
    return i * n_lat + j


def test_constructor_and_properties():
    reg = SPDEMaternReg(model_shape=(4, 3), ell=1.5, sigma=0.2)
    assert reg.ell == 1.5
    assert reg.rho == pytest.approx(3.0)
    assert reg.sigma == 0.2
    assert reg.grid_shape == (4, 3)
    assert reg.grid_spacing == (1.0, 1.0)
    assert reg.kappa == pytest.approx(np.sqrt(2.0) / 1.5)

    reg = SPDEMaternReg(
        model_shape=(4, 3),
        ell=1.75,
        sigma=0.4,
        grid_spacing=(2.0, 0.5),
    )
    assert reg.ell == 1.75
    assert reg.rho == pytest.approx(3.5)
    assert reg.grid_spacing == (2.0, 0.5)
    assert reg.kappa == pytest.approx(np.sqrt(2.0) / 1.75)


def test_invalid_inputs():
    with pytest.raises(ValueError, match=r".*2D model_shape.*"):
        SPDEMaternReg(model_shape=(4,), ell=5.0)
    with pytest.raises(ValueError, match=r".*positive integer.*"):
        SPDEMaternReg(model_shape=(0, 3), ell=5.0)
    with pytest.raises(ValueError, match=r".*ell.*positive.*"):
        SPDEMaternReg(model_shape=(4, 3), ell=0.0)
    with pytest.raises(ValueError, match=r".*sigma.*positive.*"):
        SPDEMaternReg(model_shape=(4, 3), ell=5.0, sigma=0.0)
    with pytest.raises(ValueError, match=r".*grid_spacing.*positive.*"):
        SPDEMaternReg(model_shape=(4, 3), ell=5.0, grid_spacing=0.0)
    with pytest.raises(ValueError, match=r".*grid_spacing.*positive.*"):
        SPDEMaternReg(model_shape=(4, 3), ell=5.0, grid_spacing=(1.0, -1.0))


def test_matrix_construction_unit_grid():
    reg = SPDEMaternReg(model_shape=(2, 3), ell=1.0, sigma=0.5)
    expected = _expected_precision_factor((2, 3), ell=1.0, sigma=0.5)
    assert sparse.issparse(reg.matrix)
    assert np.allclose(reg.matrix.toarray(), expected)


def test_matrix_construction_anisotropic_spacing():
    reg = SPDEMaternReg(
        model_shape=(2, 3),
        ell=1.5,
        sigma=0.75,
        grid_spacing=(2.0, 0.25),
    )
    expected = _expected_precision_factor(
        (2, 3),
        ell=1.5,
        sigma=0.75,
        grid_spacing=(2.0, 0.25),
    )
    assert np.allclose(reg.matrix.toarray(), expected)


def test_non_unit_grid_spacing_includes_lumped_mass_factor():
    ell = 1.5
    sigma = 0.75
    grid_spacing = (2.0, 0.25)

    reg = SPDEMaternReg(
        model_shape=(2, 3),
        ell=ell,
        sigma=sigma,
        grid_spacing=grid_spacing,
    )
    expected = _expected_precision_factor(
        (2, 3),
        ell=ell,
        sigma=sigma,
        grid_spacing=grid_spacing,
    )
    legacy = expected / np.sqrt(grid_spacing[0] * grid_spacing[1])

    assert np.allclose(reg.matrix.toarray(), expected)
    assert not np.allclose(reg.matrix.toarray(), legacy)


def test_reference_model_and_quadratic_structure():
    reference_model = np.arange(6.0).reshape(2, 3)
    reg = SPDEMaternReg(
        model_shape=(2, 3),
        ell=1.0,
        sigma=0.5,
        reference_model=reference_model,
    )
    q_matrix = reg.matrix.T @ reg.matrix
    model = reference_model + np.array([[1.0, -1.0, 2.0], [0.5, -0.5, 1.5]])

    assert reg(reference_model) == pytest.approx(0.0)
    assert np.allclose(reg.gradient(reference_model), np.zeros(reference_model.size))
    assert np.allclose(
        reg.hessian(reference_model).toarray(),
        reg.hessian(model).toarray(),
    )
    assert np.allclose(reg.hessian(model).toarray(), (2.0 * q_matrix).toarray())
    assert np.allclose(q_matrix.toarray(), q_matrix.toarray().T)
    assert np.linalg.eigvalsh(q_matrix.toarray()).min() > 0.0


def test_empirical_covariance_matches_sigma_and_correlation_shape():
    ell = 3.0
    sigma = 0.2
    reg = SPDEMaternReg(model_shape=(15, 15), ell=ell, sigma=sigma)
    q_matrix = (reg.matrix.T @ reg.matrix).toarray()
    covariance = np.linalg.inv(q_matrix)
    center = _index(7, 7, 15)
    neighbor = _index(7, 8, 15)
    variance = covariance[center, center]
    correlation = covariance[center, neighbor] / np.sqrt(
        covariance[center, center] * covariance[neighbor, neighbor]
    )
    expected_correlation = reg.kappa * k1(reg.kappa)

    assert variance == pytest.approx(sigma**2, rel=0.2)
    assert correlation == pytest.approx(expected_correlation, abs=0.1)


def test_physical_covariance_is_consistent_under_grid_refinement():
    ell = 1.5
    sigma = 0.2

    coarse = SPDEMaternReg(
        model_shape=(9, 9),
        ell=ell,
        sigma=sigma,
        grid_spacing=1.0,
    )
    fine = SPDEMaternReg(
        model_shape=(17, 17),
        ell=ell,
        sigma=sigma,
        grid_spacing=0.5,
    )

    coarse_cov = np.linalg.inv((coarse.matrix.T @ coarse.matrix).toarray())
    fine_cov = np.linalg.inv((fine.matrix.T @ fine.matrix).toarray())

    coarse_center = _index(4, 4, 9)
    fine_center = _index(8, 8, 17)
    coarse_neighbor = _index(4, 5, 9)
    fine_neighbor = _index(8, 10, 17)

    coarse_variance = coarse_cov[coarse_center, coarse_center]
    fine_variance = fine_cov[fine_center, fine_center]
    coarse_correlation = coarse_cov[coarse_center, coarse_neighbor] / np.sqrt(
        coarse_cov[coarse_center, coarse_center]
        * coarse_cov[coarse_neighbor, coarse_neighbor]
    )
    fine_correlation = fine_cov[fine_center, fine_neighbor] / np.sqrt(
        fine_cov[fine_center, fine_center] * fine_cov[fine_neighbor, fine_neighbor]
    )

    assert fine_variance == pytest.approx(coarse_variance, rel=0.15)
    assert fine_correlation == pytest.approx(coarse_correlation, abs=0.05)


def test_large_ell_warning_uses_physical_grid_extent():
    # rho = 2*ell = 6.0 > 0.5 * max(4*2, 10*1) = 5.0 → warns
    with pytest.warns(UserWarning, match=r".*longest physical grid dimension.*"):
        SPDEMaternReg(
            model_shape=(4, 10),
            ell=3.0,
            sigma=0.2,
            grid_spacing=(2.0, 1.0),
        )

    # rho = 2*ell = 4.0 < 5.0 → no warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SPDEMaternReg(
            model_shape=(4, 10),
            ell=2.0,
            sigma=0.2,
            grid_spacing=(2.0, 1.0),
        )
    assert not caught
