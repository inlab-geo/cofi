"""Template test file for a new CoFI tool

This file serves as a starting point only. Feel free to refer to existing test files
as references, and to modify your own test file as needed.
"""

import numpy as np
import pytest

from cofi.tools import MyNewTool
from cofi import BaseProblem, InversionOptions, Inversion


m_true = np.array([0, 0, 1])
x = np.random.choice(np.linspace(-5, 5), 20)
y = np.array([x**i for i in range(3)]).T @ m_true
m0 = np.array([0, 0, 0])

def forward_func(m):
    return np.array([x**i for i in range(3)]).T @ m    

def jacobian(m):
    return np.array([x**i for i in range(3)]).T

def objective(m):
    return np.sum((forward_func(m) - y)**2)

def gradient(m):
    return 2 * (forward_func(m) - y) @ jacobian(m)

def hessian(m):
    return 2 * jacobian(m).T @ jacobian(m)

def log_likelihood(m):
    return -0.5 * np.sum((forward_func(m) - y)**2)

def log_prior(m):
    return 0

@pytest.fixture
def problem_setup():
    # define BaseProblem
    inv_problem = BaseProblem()
    inv_problem.set_data(y)
    inv_problem.set_initial_model(m0)
    inv_problem.set_forward(forward_func)
    inv_problem.set_objective(objective)
    inv_problem.set_gradient(gradient)
    inv_problem.set_hessian(hessian)
    inv_problem.set_jacobian(jacobian)
    inv_problem.set_data_misfit("least squares")
    inv_problem.set_regularization(lambda _: 0, np.zeros((3,3)))
    inv_problem.set_log_likelihood(log_likelihood)
    inv_problem.set_log_prior(log_prior)
    inv_problem.set_data_covariance(np.eye(3))
    inv_problem.set_data_covariance_inv(np.eye(3))
    # define InversionOptions
    inv_options = InversionOptions()
    inv_options.set_tool("TODO: FILL ME")
    inv_options.set_params(
        todo1="FILL ME",
        todo2="FILL ME"
    )
    
def test_validate(problem_setup):       # TODO
    inv_problem, inv_options = problem_setup
    raise NotImplementedError

def test_call(problem_setup):           # TODO
    inv_problem, inv_options = problem_setup
    raise NotImplementedError
