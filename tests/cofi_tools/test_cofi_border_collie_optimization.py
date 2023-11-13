import numpy as np
import pytest

from cofi.tools import CoFIBorderCollieOptimization
from cofi import BaseProblem, InversionOptions, Inversion

def rosen(x):
    a=1.0
    b=10.0
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

@pytest.fixture
def problem_setup():
    # Define the Base Problem
    inv_problem = BaseProblem()
    inv_problem.name = "Rosenbrock Function"
    inv_problem.set_objective(rosen)
    inv_problem.set_initial_model([0,0])
    inv_problem.set_model_shape((2))

    bounds= ((-1.0,3.0),(-1.0,3.0))
    inv_problem.set_bounds(bounds)
    inv_options = InversionOptions()
    inv_options.set_params(number_of_iterations=3)
    inv_options.set_tool("cofi.border_collie_optimization")
    return inv_problem, inv_options

def test_run(problem_setup):
    inv_problem, inv_options = problem_setup
    solver = CoFIBorderCollieOptimization(inv_problem, inv_options)
    res = solver()
    #assert res["model"][0] == 3.0
    #assert inv_problem.initial_model[0] == 9.

