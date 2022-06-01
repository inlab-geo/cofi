import numpy as np
import pytest

from cofi.solvers import ScipyLstSqSolver
from cofi import BaseProblem, InversionOptions


def test_validate_jac():
    # 1
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(lambda x: np.array([1, 2, 3]))
    inv_problem.set_hessian(lambda x: x)
    inv_problem.set_data(np.array([2]))
    inv_options = InversionOptions()
    inv_solver = ScipyLstSqSolver(inv_problem, inv_options)
    # 2
    inv_problem.set_model_shape((1, 2))
    inv_solver = ScipyLstSqSolver(inv_problem, inv_options)
    # 3
    inv_problem.set_initial_model(np.ones((1, 2)))
    inv_solver = ScipyLstSqSolver(inv_problem, inv_options)
    # 4
    inv_problem.set_jacobian(lambda: np.array([1]))
    with pytest.raises(ValueError, match=".*isn't set properly.*"):
        inv_solver = ScipyLstSqSolver(inv_problem, inv_options)


def test_run():
    x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
    y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])
    M = x[:, np.newaxis] ** [0, 2]
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(M)
    inv_problem.set_data(y)
    inv_options = InversionOptions()
    solver = ScipyLstSqSolver(inv_problem, inv_options)
    res = solver()
    assert res["success"]
