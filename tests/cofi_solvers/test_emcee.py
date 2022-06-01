import numpy as np
import pytest

from cofi.solvers import EmceeSolver
from cofi import BaseProblem, InversionOptions, Inversion


############### Problem setup #########################################################
_sample_size = 20
_x = np.random.choice(np.linspace(-3.5,2.5), size=_sample_size)
_forward = lambda model: np.vander(_x, N=4, increasing=True) @ model
_y = _forward(np.array([-6,-5,2,1])) + np.random.normal(0,1,_sample_size)
_sigma = 1.0                                # common noise standard deviation
_Cdinv = np.eye(len(_y))/(_sigma**2) # Inverse data covariance matrix
_prior_args_gauss = [
    "Gaussian", 
    np.zeros(4),
    np.diag(np.array([1/5.,1/5.,1/2.,1.])), 
]
_prior_args_uniform = [
    "Uniform",
    np.ones(4) * (-10.),
    np.ones(4) * 10.,
]


def _log_likelihood(model, data_observed, Cdinv):
    data_predicted = _forward(model)
    residual = data_observed - data_predicted
    _log_likelihood = -0.5 * residual @ (Cdinv @ residual).T
    return _log_likelihood
log_likelihood = lambda m: _log_likelihood(m, _y, _Cdinv)

def _log_prior_gauss(model, prior_mu, prior_covariance):
    model_diff = model - prior_mu
    _log_prior = -0.5 * model_diff @ (prior_covariance @ model_diff).T
    return _log_prior
log_prior_gauss = lambda m: _log_prior_gauss(m, *_prior_args_gauss[1:])

def _log_prior_uniform(model, lower_bound, upper_bound):
    for i in range(len(lower_bound)):
        if model[i] < lower_bound[i] or model[i] > upper_bound[i]: return -np.inf
    return 0.0 # model lies within bounds -> return log(1)
log_prior_uniform = lambda m: _log_prior_uniform(m, *_prior_args_uniform[1:])

def _log_posterior(model, data_observed, Cdinv):
    lp = log_prior_gauss(model)
    if not np.isfinite(lp): return -np.inf
    return lp + _log_likelihood(model, data_observed, Cdinv)
log_posterior = lambda m: _log_posterior(m, _y, _Cdinv)

nwalkers = 32
ndim = 4
nsteps = 500
walkers_start = np.array([0.,0.,0.,0.]) + 1e-4 * np.random.randn(nwalkers, ndim)

############### Begin testing #########################################################
def test_validate():
    inv_problem = BaseProblem()
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    # 1
    with pytest.raises(ValueError, match=r".*not enough information.*BaseProblem.*"):
        emcee_solver = EmceeSolver(inv_problem, inv_options)
    # 2
    inv_problem.log_posterior = log_posterior
    inv_problem.set_walkers_starting_pos(walkers_start)
    with pytest.raises(ValueError, match=r".*not enough info.*InversionOptions.*"):
        emcee_solver = EmceeSolver(inv_problem, inv_options)
    # 3
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
    emcee_solver = EmceeSolver(inv_problem, inv_options)
    assert emcee_solver._ndim == 4

def test_run_with_posterior():
    # set up problem
    inv_problem = BaseProblem()
    inv_problem.log_posterior = log_posterior
    inv_problem.set_walkers_starting_pos(walkers_start)
    # set up options
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
    # define solver
    emcee_solver = EmceeSolver(inv_problem, inv_options)
    res = emcee_solver()

def test_run_with_prior_likelihood():
    # set up problem
    inv_problem = BaseProblem()
    inv_problem.set_log_prior(log_prior_uniform)
    inv_problem.set_log_likelihood(log_likelihood)
    inv_problem.set_walkers_starting_pos(walkers_start)
    # set up options
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
    # define solver
    emcee_solver = EmceeSolver(inv_problem, inv_options)
    res = emcee_solver()

def test_with_inversion_prior_likelihood():
    # set up problem
    inv_problem = BaseProblem()
    inv_problem.set_log_prior(log_prior_uniform)
    inv_problem.set_log_likelihood(log_likelihood)
    inv_problem.set_walkers_starting_pos(walkers_start)
    # set up options
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
    # define inversion
    inv = Inversion(inv_problem, inv_options)
    res = inv.run()

def test_with_inversion_posterior():
    # set up problem
    inv_problem = BaseProblem()
    inv_problem.log_posterior = log_posterior
    inv_problem.set_walkers_starting_pos(walkers_start)
    # set up options
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
    # define inversion
    inv = Inversion(inv_problem, inv_options)
    res = inv.run()

def test_no_initial_state():
    # set up problem
    inv_problem = BaseProblem()
    inv_problem.set_log_posterior(log_posterior)
    inv_problem.set_model_shape(ndim)
    # set up options
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
    # define inversion
    with pytest.raises(ValueError):
        inv = Inversion(inv_problem, inv_options)
