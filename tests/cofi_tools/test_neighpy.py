import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
import pytest

from cofi.tools import Neighpy, NeighpyI, NeighpyII
from cofi import BaseProblem, InversionOptions, Inversion


############### Problem setup #########################################################
np.random.seed(0)

_sample_size = 20
_ndim = 3  # quadratic model


def basis_func(x):
    return np.array([x**i for i in range(_ndim)]).T


_x = np.random.choice(np.linspace(-3.5, 2.5), size=_sample_size)
_forward = lambda model: basis_func(_x) @ model
_sigma = 1e-4  # common noise standard deviation
_true_model = np.random.randint(-5, 5, _ndim)  # don't need to check results
_y = _forward(_true_model) + np.random.normal(0, _sigma, _sample_size)
_Cdinv = np.eye(len(_y)) / (_sigma**2)  # Inverse data covariance matrix


def objective(x: NDArray) -> float:
    data_predicted = _forward(x)
    residual = _y - data_predicted
    return residual @ (_Cdinv @ residual).T


inv_problem = BaseProblem(objective=objective)

bounds = [(-5.0, 5.0)] * _ndim
direct_search_ns = 100
direct_search_nr = 10
direct_search_ni = 10
direct_search_n = 10
_direct_search_total = direct_search_ni + direct_search_n * direct_search_ns
direct_search_serial = False
appraisal_n_resample = 100
appraisal_n_walkers = 1  # only one walker for testing

initial_ensemble = np.random.uniform(-10, 10, (_direct_search_total, _ndim))
log_ppd = -1 * np.apply_along_axis(lambda x: objective(x), 1, initial_ensemble)


@pytest.fixture(scope="module")
def valid_options() -> InversionOptions:
    inv_options = InversionOptions()
    inv_options.set_params(
        bounds=bounds,
        n_samples_per_iteration=direct_search_ns,
        n_cells_to_resample=direct_search_nr,
        n_initial_samples=direct_search_ni,
        n_iterations=direct_search_n,
        serial=direct_search_serial,
        initial_ensemble=initial_ensemble,
        log_ppd=log_ppd,
        n_resample=appraisal_n_resample,
        n_walkers=appraisal_n_walkers,
    )
    return inv_options


@pytest.fixture(scope="module")
def neighpyI_solver(valid_options) -> NeighpyI:
    return NeighpyI(inv_problem, valid_options)


@pytest.fixture(scope="module")
def neighpyII_solver(valid_options) -> NeighpyII:
    return NeighpyII(inv_problem, valid_options)


@pytest.fixture(scope="module")
def neighpy_solver(valid_options) -> Neighpy:
    return Neighpy(inv_problem, valid_options)


def _check_neighpyI_options(solver: NeighpyI):
    assert solver._params["n_samples_per_iteration"] == direct_search_ns
    assert solver._params["n_cells_to_resample"] == direct_search_nr
    assert solver._params["n_initial_samples"] == direct_search_ni
    assert solver._params["n_iterations"] == direct_search_n
    assert solver._params["bounds"] == bounds
    assert solver._params["ndim"] == _ndim


def _check_neighpyI_results(results: dict):
    assert results["model"].shape == (_ndim,)
    assert results["samples"].shape == (_direct_search_total, _ndim)
    assert results["objectives"].shape == (_direct_search_total,)


def _check_neighpyII_options(solver: NeighpyII):
    assert solver._params["n_resample"] == appraisal_n_resample
    assert solver._params["n_walkers"] == appraisal_n_walkers
    assert np.array_equal(solver._params["initial_ensemble"], initial_ensemble)
    assert np.array_equal(solver._params["log_ppd"], log_ppd)
    assert solver._params["bounds"] == bounds
    assert solver._params["ndim"] == _ndim


def _check_neighpyII_results(results: dict):
    assert results["new_samples"].shape == (appraisal_n_resample, _ndim)


def _check_neighpy_options(solver: Neighpy):
    assert solver._params["n_samples_per_iteration"] == direct_search_ns
    assert solver._params["n_cells_to_resample"] == direct_search_nr
    assert solver._params["n_initial_samples"] == direct_search_ni
    assert solver._params["n_resample"] == appraisal_n_resample
    assert solver._params["n_walkers"] == appraisal_n_walkers
    assert solver._params["n_iterations"] == direct_search_n
    assert solver._params["bounds"] == bounds
    assert solver._params["ndim"] == _ndim


def _check_neighpy_results(results: dict):
    assert results["model"].shape == (_ndim,)
    assert results["direct_search_samples"].shape == (_direct_search_total, _ndim)
    assert results["direct_search_objectives"].shape == (_direct_search_total,)
    assert results["appraisal_samples"].shape == (appraisal_n_resample, _ndim)


############### Begin testing #########################################################
@pytest.mark.parametrize("solver", [NeighpyI, NeighpyII, Neighpy])
def test_empty_baseproblem(solver) -> None:
    inv_problem = BaseProblem()
    inv_options = InversionOptions()

    # An empty BaseProblem is not a problem for NeighpyII,
    # so the error is raised by the InversionOptions
    # so we match for BaseProblem|InversionOptions
    with pytest.raises(
        ValueError, match=r".*not enough information.*BaseProblem|InversionOptions.*"
    ):
        _ = solver(inv_problem, inv_options)


@pytest.mark.parametrize("solver", [NeighpyI, NeighpyII, Neighpy])
def test_empty_inversionoptions(solver) -> None:
    inv_problem = BaseProblem(objective=objective)
    inv_options = InversionOptions()

    with pytest.raises(
        ValueError, match=r".*not enough information.*InversionOptions.*"
    ):
        _ = solver(inv_problem, inv_options)


@pytest.mark.parametrize(
    "solver, check_fn",
    [
        ("neighpyI_solver", _check_neighpyI_options),
        ("neighpyII_solver", _check_neighpyII_options),
        ("neighpy_solver", _check_neighpy_options),
    ],
)
def test_valid_inversionoptions(solver, check_fn, request) -> None:
    solver = request.getfixturevalue(solver)
    check_fn(solver)


@pytest.mark.parametrize(
    "initial_ensemble,error_match",
    [
        (
            initial_ensemble[::2],
            r".*number of samples.*",
        ),
        (
            np.zeros((_direct_search_total, _ndim - 1)),
            r".*number of dimensions.*",
        ),
    ],
)
def test_neghipyII_input_shapes(initial_ensemble, error_match, valid_options) -> None:
    # invalid ensemble and log_ppd shapes
    inv_options = deepcopy(valid_options)
    inv_options.set_params(initial_ensemble=initial_ensemble)
    with pytest.raises(ValueError, match=error_match):
        _ = NeighpyII(BaseProblem(), inv_options)


@pytest.mark.parametrize(
    "solver, check_fn",
    [
        ("neighpyI_solver", _check_neighpyI_results),
        ("neighpyII_solver", _check_neighpyII_results),
        ("neighpy_solver", _check_neighpy_results),
    ],
)
def test_call(solver, check_fn, request) -> None:
    solver = request.getfixturevalue(solver)
    results = solver()
    assert results["success"] is True
    check_fn(results)


@pytest.mark.parametrize(
    "tool, check_fn",
    [
        ("neighpyI", _check_neighpyI_results),
        ("neighpyII", _check_neighpyII_results),
        ("neighpy", _check_neighpy_results),
    ],
)
def test_use_in_Inversion(tool, check_fn, valid_options) -> None:
    inv_problem = BaseProblem(objective=objective)
    valid_options.set_tool(tool)

    inv = Inversion(inv_problem, valid_options)
    results = inv.run()
    assert results.success is True
    check_fn(results.res)
