import numpy as np
import pytest

from cofi.tools import Neighpy, NeighpyI, NeighpyII
from cofi import BaseProblem, InversionOptions, Inversion


############### Problem setup #########################################################
_sample_size = 20
_ndim = 4
_x = np.random.choice(np.linspace(-3.5, 2.5), size=_sample_size)
_forward = lambda model: np.vander(_x, N=_ndim, increasing=True) @ model
_y = _forward(np.random.randint(-5, 5, _ndim)) + np.random.normal(0, 1, _sample_size)
_sigma = 1.0  # common noise standard deviation
_Cdinv = np.eye(len(_y)) / (_sigma**2)  # Inverse data covariance matrix


def _objective(model, data_observed, Cdinv):
    data_predicted = _forward(model)
    residual = data_observed - data_predicted
    return residual @ (Cdinv @ residual).T


objective = lambda m: _objective(m, _y, _Cdinv)
inv_problem = BaseProblem(objective=objective)

bounds = [(-10.0, 10.0)] * _ndim
direct_search_ns = 100
direct_search_nr = 10
direct_search_ni = 100
direct_search_n = 10
direct_search_serial = False
appraisal_n_resample = 1000
appraisal_n_walkers = 5


@pytest.fixture(scope="module")
def inversion_options(request):
    inv_options = InversionOptions()
    inv_options.set_tool("neighpy")
    inv_options.set_params(
        bounds=bounds,
        n_samples_per_iteration=direct_search_ns,
        n_cells_to_resample=direct_search_nr,
        n_initial_samples=direct_search_ni,
        n_iterations=direct_search_n,
        serial=request.param,
        n_resample=appraisal_n_resample,
        n_walkers=appraisal_n_walkers,
    )
    return inv_options


############### Begin testing #########################################################
def test_validate_neighpyI():
    inv_problem = BaseProblem()
    inv_options = InversionOptions()
    # 1 - no objective
    with pytest.raises(ValueError, match=r".*not enough information.*BaseProblem.*"):
        neighpy_solver = NeighpyI(inv_problem, inv_options)
    # 2 - no inv_options
    inv_problem.objective = objective
    with pytest.raises(ValueError, match=r".*not enough info.*InversionOptions.*"):
        neighpy_solver = NeighpyI(inv_problem, inv_options)
    # 3 - valid options
    inv_options.set_params(
        n_samples_per_iteration=direct_search_ns,
        n_cells_to_resample=direct_search_nr,
        n_initial_samples=direct_search_ni,
        n_iterations=direct_search_n,
        bounds=bounds,
    )
    neighpy_solver = NeighpyI(inv_problem, inv_options)
    assert neighpy_solver._params["n_samples_per_iteration"] == direct_search_ns
    assert neighpy_solver._params["n_cells_to_resample"] == direct_search_nr
    assert neighpy_solver._params["n_initial_samples"] == direct_search_ni
    assert neighpy_solver._params["n_iterations"] == direct_search_n
    assert neighpy_solver._params["bounds"] == bounds
    assert neighpy_solver._params["ndim"] == _ndim


def test_call_neighpyI():
    inv_problem = BaseProblem(objective=objective)
    inv_options = InversionOptions()
    inv_options.set_params(
        bounds=bounds,
        n_samples_per_iteration=direct_search_ns,
        n_cells_to_resample=direct_search_nr,
        n_initial_samples=direct_search_ni,
        n_iterations=direct_search_n,
    )
    neighpy_solver = NeighpyI(inv_problem, inv_options)
    res = neighpy_solver()
    assert res["success"] is True
    assert res["model"].shape == (_ndim,)
    _direct_search_total = direct_search_ni + direct_search_n * direct_search_ns
    assert res["samples"].shape == (_direct_search_total, _ndim)
    assert res["objectives"].shape == (_direct_search_total,)


def test_validate_neighpyII():
    initial_ensemble = np.random.uniform(-10, 10, (_sample_size, _ndim))
    log_ppd = np.apply_along_axis(lambda x: -1 * objective(x), 1, initial_ensemble)

    inv_problem = BaseProblem()
    inv_options = InversionOptions()
    # 1 - NeighpyII doesn't require anything in BaseProblem
    with pytest.raises(
        ValueError, match=r".*not enough information.*InversionOptions.*"
    ):
        neighpy_solver = NeighpyII(inv_problem, inv_options)
    # 2 - invalid ensemble and log_ppd shapes
    with pytest.raises(ValueError, match=r".*number of samples.*"):
        inv_options.set_params(
            initial_ensemble=initial_ensemble[::2],
            log_ppd=log_ppd,
            n_resample=appraisal_n_resample,
            n_walkers=appraisal_n_walkers,
            bounds=bounds,
        )
        neighpy_solver = NeighpyII(inv_problem, inv_options)

    with pytest.raises(ValueError, match=r".*number of dimensions.*"):
        inv_options.set_params(
            initial_ensemble=np.zeros((_sample_size, _ndim - 1)),
            log_ppd=log_ppd,
            n_resample=appraisal_n_resample,
            n_walkers=appraisal_n_walkers,
            bounds=bounds,
        )
        neighpy_solver = NeighpyII(inv_problem, inv_options)

    # 3 - valid options
    inv_options.set_params(
        initial_ensemble=initial_ensemble,
        log_ppd=log_ppd,
        n_resample=appraisal_n_resample,
        n_walkers=appraisal_n_walkers,
        bounds=bounds,
    )
    neighpy_solver = NeighpyII(inv_problem, inv_options)
    assert neighpy_solver._params["n_resample"] == appraisal_n_resample
    assert neighpy_solver._params["n_walkers"] == appraisal_n_walkers
    assert np.array_equal(neighpy_solver._params["initial_ensemble"], initial_ensemble)
    assert np.array_equal(neighpy_solver._params["log_ppd"], log_ppd)
    assert neighpy_solver._params["bounds"] == bounds
    assert neighpy_solver._params["ndim"] == _ndim


def test_call_neighpyII():
    initial_ensemble = np.random.uniform(-10, 10, (_sample_size, _ndim))
    log_ppd = np.apply_along_axis(lambda x: objective(x), 1, initial_ensemble)
    inv_problem = BaseProblem(objective=objective)
    inv_options = InversionOptions()
    inv_options.set_params(
        bounds=bounds,
        initial_ensemble=initial_ensemble,
        log_ppd=log_ppd,
        n_resample=appraisal_n_resample,
        n_walkers=1,  # parallel hanging for some reason but fine for combined Neighpy class
    )
    neighpy_solver = NeighpyII(inv_problem, inv_options)
    res = neighpy_solver()
    assert res["success"] is True
    assert res["new_samples"].shape == (appraisal_n_resample, _ndim)


def test_validate_neighpy():
    inv_problem = BaseProblem()
    inv_options = InversionOptions()
    # 1
    with pytest.raises(ValueError, match=r".*not enough information.*BaseProblem.*"):
        neighpy_solver = Neighpy(inv_problem, inv_options)
    # 2
    inv_problem.objective = objective
    with pytest.raises(ValueError, match=r".*not enough info.*InversionOptions.*"):
        neighpy_solver = Neighpy(inv_problem, inv_options)
    # 3
    inv_options.set_params(
        bounds=bounds,
        n_samples_per_iteration=direct_search_ns,
        n_cells_to_resample=direct_search_nr,
        n_initial_samples=direct_search_ni,
        n_iterations=direct_search_n,
        n_resample=appraisal_n_resample,
        n_walkers=appraisal_n_walkers,
    )
    neighpy_solver = Neighpy(inv_problem, inv_options)
    assert neighpy_solver._params["n_samples_per_iteration"] == direct_search_ns
    assert neighpy_solver._params["n_cells_to_resample"] == direct_search_nr
    assert neighpy_solver._params["n_initial_samples"] == direct_search_ni
    assert neighpy_solver._params["n_resample"] == appraisal_n_resample
    assert neighpy_solver._params["n_walkers"] == appraisal_n_walkers
    assert neighpy_solver._params["n_iterations"] == direct_search_n
    assert neighpy_solver._params["bounds"] == bounds
    assert neighpy_solver._params["ndim"] == _ndim


# parameterisation to test both parallel and serial direct search
@pytest.mark.parametrize("inversion_options", [True, False], indirect=True)
def test_call_neighpy(inversion_options):
    neighpy_inversion = Inversion(inv_problem, inversion_options)
    res = neighpy_inversion.run()
    assert res.success is True
    assert res.model.shape == (_ndim,)
    _direct_search_total = direct_search_ni + direct_search_n * direct_search_ns
    assert res.direct_search_samples.shape == (_direct_search_total, _ndim)
    assert res.direct_search_objectives.shape == (_direct_search_total,)
    assert res.appraisal_samples.shape == (appraisal_n_resample, _ndim)
