import numpy as np
import pytest

from cofi.solvers import EmceeSolver
from cofi import BaseProblem, InversionOptions


def _log_prior(theta):
    a,b,c,d, log_f = theta
    if -10.0 < a < 10.0 and -10.0 < b < 10.0 and  -10.0 < c < 10.0 and -10.0 < d < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def _log_likelihood(theta, x, y):
    a,b,c,d, log_f = theta
    model = a+b*x+c*x**2+d*x**3
    sigma2 =  np.exp(2* log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def _log_probability(theta, x, y):
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta, x, y)

def test_validate():
    inv_problem = BaseProblem()
    inv_problem.log_prior = _log_prior
    inv_problem.log_likelihood = _log_likelihood
    inv_problem.log_probability = _log_probability
    
