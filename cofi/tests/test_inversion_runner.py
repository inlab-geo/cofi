import numpy as np
import pytest

from cofi import InversionRunner, BaseProblem, InversionOptions


@pytest.fixture
def poly_problem():
    inv_problem = BaseProblem()
    _x = np.array([1,2,3,4,5])
    _G = np.array([_x**i for i in range(3)]).T
    _m_true = np.array([2,1,1])
    _y = _G @ _m_true
    inv_problem.set_dataset(_x, _y)
    inv_problem.set_jacobian(_G)
    inv_problem.set_hessian(_G.T @ _G)
    return inv_problem

def test_solve(poly_problem):
    inv_options = InversionOptions()
    inv_options.set_tool("numpy.linalg.lstsq")
    runner = InversionRunner(poly_problem, inv_options)
    runner.run()
