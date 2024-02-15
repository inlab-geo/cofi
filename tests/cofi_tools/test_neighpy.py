import numpy as np
import pytest

from cofi.tools import Neighpy
from cofi import BaseProblem, InversionOptions, Inversion


############### Problem setup #########################################################
_sample_size = 20
_x = np.random.choice(np.linspace(-3.5, 2.5), size=_sample_size)
_forward = lambda model: np.vander(_x, N=4, increasing=True) @ model
_y = _forward(np.array([-6, -5, 2, 1])) + np.random.normal(0, 1, _sample_size)
_sigma = 1.0  # common noise standard deviation
_Cdinv = np.eye(len(_y)) / (_sigma**2)  # Inverse data covariance matrix


def _objective(model, data_observed, Cdinv):
    data_predicted = _forward(model)
    residual = data_observed - data_predicted
    return -0.5 * residual @ (Cdinv @ residual).T


objective = lambda m: _objective(m, _y, _Cdinv)
bounds = [(-10.0, 10.0)] * 4
direct_search_ns = 100
direct_search_nr = 10
direct_search_ni = 100
direct_search_n = 10
appraisal_n_resample = 1000
appraisal_n_walkers = 5


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
    )
    neighpy_solver = Neighpy(inv_problem, inv_options)
    assert neighpy_solver._params["ndim"] == 4