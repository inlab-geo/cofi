from emcee import EnsembleSampler
import numpy as np
import pytest

from cofi.solvers import EmceeSolver
from cofi import BaseProblem, InversionOptions


############### Problem setup #########################################################
_sample_size = 20
_x = np.random.choice(np.linspace(-3.5,2.5), size=_sample_size)
_forward = lambda model: np.vander(_x, N=4, increasing=True) @ model
_y = _forward(np.array([-6,-5,2,1])) + np.random.normal(0,1,_sample_size)
_sigma = 1.0                                # common noise standard deviation
_Cdinv = np.eye(len(_y))/(_sigma**2) # Inverse data covariance matrix


def log_likelihood(model, data_observed, Cdinv):
    data_predicted = _forward(model)
    residual = data_observed - data_predicted
    _log_likelihood = -0.5 * residual @ (Cdinv @ residual).T
    return _log_likelihood

def _log_prior_gauss(model, prior_mu, prior_covariance):
    model_diff = model - prior_mu
    _log_prior = -0.5 * model_diff @ (prior_covariance @ model_diff).T
    return _log_prior

def _log_prior_uniform(model, lower_bound, upper_bound):
    for i in range(len(lower_bound)):
        if(model[i] < lower_bound[i] or model[i] > upper_bound[i]): # if input model lies outside bounds -> return low probability
            return -np.inf
    return 0.0 # model lies within bounds -> return log(1)

def log_prior(model, prior_args):
    prior_option = prior_args[0]
    if prior_option == "Gaussian":
        return _log_prior_gauss(model, *prior_args[1:])
    elif prior_option == "Uniform":
        return _log_prior_uniform(model, *prior_args[1:])

def log_posterior(model, data_observed, Cdinv, prior_args):
    lp = log_prior(model, prior_args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(model, data_observed, Cdinv)


############### Begin testing #########################################################
def test_validate():
    inv_problem = BaseProblem()
    _prior_args_gauss = [
        "Gaussian", 
        np.diag(np.array([1/5.,1/5.,1/2.,1.])), 
        np.zeros(4),
    ]
    _prior_args_uniform = [
        "Uniform",
        np.ones(4) * (-10.),
        np.ones(4) * 10.,
    ]
    _log_prosterior = lambda m: log_posterior(m, _y, _Cdinv, _prior_args_gauss)
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    with pytest.raises(ValueError, match=r".*not enough information.*"):
        emcee_solver = EmceeSolver(inv_problem, inv_options)
    inv_problem.log_posterior = log_posterior
    inv_problem.set_initial_model(np.array([0.,0.,0.,0.]))
    emcee_solver = EmceeSolver(inv_problem, inv_options)

