import numpy as np
import pytest

from cofi.tools import Neighpy
from cofi import BaseProblem, InversionOptions, Inversion


############### Problem setup #########################################################
_sample_size = 20
_ndim = 4
_x = np.random.choice(np.linspace(-3.5, 2.5), size=_sample_size)
_forward = lambda model: np.vander(_x, N=_ndim, increasing=True) @ model
_y = _forward(np.random.randint(-5, 5, _ndim)) + np.random.normal(
    0, 1, _sample_size
)
_sigma = 1.0  # common noise standard deviation
_Cdinv = np.eye(len(_y)) / (_sigma**2)  # Inverse data covariance matrix


def _objective(model, data_observed, Cdinv):
    data_predicted = _forward(model)
    residual = data_observed - data_predicted
    return residual @ (Cdinv @ residual).T


objective = lambda m: _objective(m, _y, _Cdinv)
bounds = [(-10.0, 10.0)] * _ndim
direct_search_ns = 100
direct_search_nr = 10
direct_search_ni = 100
direct_search_n = 10
appraisal_n_resample = 1000
appraisal_n_walkers = 5


inv_problem = BaseProblem(objective=objective)
inv_options = InversionOptions()
inv_options.set_tool("neighpy")
inv_options.set_params(
    bounds=bounds,
    direct_search_ns=direct_search_ns,
    direct_search_nr=direct_search_nr,
    direct_search_ni=direct_search_ni,
    direct_search_n=direct_search_n,
    appraisal_n_resample=appraisal_n_resample,
    appraisal_n_walkers=appraisal_n_walkers,
)


@pytest.fixture(scope="module")
def neighpy_inversion():
    return Inversion(inv_problem, inv_options)


############### Begin testing #########################################################
def test_validate():
    inv_problem = BaseProblem()
    inv_options = InversionOptions()
    inv_options.set_tool("neighpy")
    # 1
    with pytest.raises(ValueError, match=r".*not enough information.*BaseProblem.*"):
        neighpy_solver = Neighpy(inv_problem, inv_options)
    # 2
    inv_problem.objective = objective
    with pytest.raises(ValueError, match=r".*not enough info.*InversionOptions.*"):
        neighpy_solver = Neighpy(inv_problem, inv_options)
    # 3
    inv_options.set_params(
        direct_search_ns=direct_search_ns,
        direct_search_nr=direct_search_nr,
        direct_search_ni=direct_search_ni,
        direct_search_n=direct_search_n,
        appraisal_n_resample=appraisal_n_resample,
        appraisal_n_walkers=appraisal_n_walkers,
        bounds=bounds,
    )
    neighpy_solver = Neighpy(inv_problem, inv_options)
    assert neighpy_solver._params["ndim"] == _ndim


def test_call(neighpy_inversion):
    res = neighpy_inversion.run()
    assert res.success is True
    assert res.model.shape == (_ndim,)
    _direct_search_total = direct_search_ni + direct_search_n * direct_search_ns
    assert res.direct_search_samples.shape == (_direct_search_total, _ndim)
    assert res.direct_search_objectives.shape == (_direct_search_total,)
    assert res.appraisal_samples.shape == (appraisal_n_resample, _ndim)