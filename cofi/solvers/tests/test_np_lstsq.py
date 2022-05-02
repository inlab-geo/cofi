import numpy as np
import pytest

from cofi.solvers import NumpyLstSqSolver
from cofi import BaseProblem, InversionOptions


def test_validate_jac():
    # 1
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(lambda x: np.array([1,2,3]))
    inv_problem.set_hessian(lambda x: x)
    inv_problem.set_dataset(np.array([1]), np.array([2]))
    inv_options = InversionOptions()
    inv_solver = NumpyLstSqSolver(inv_problem, inv_options)
    # 2
    inv_problem.set_model_shape((1,2))
    inv_solver = NumpyLstSqSolver(inv_problem, inv_options)
    # 3
    inv_problem.set_initial_model(np.ones((1,2)))
    inv_solver = NumpyLstSqSolver(inv_problem, inv_options)
    # 4
    inv_problem.set_jacobian(lambda: np.array([1]))
    with pytest.raises(ValueError, match=".*isn't set properly.*"):
        inv_solver = NumpyLstSqSolver(inv_problem, inv_options)