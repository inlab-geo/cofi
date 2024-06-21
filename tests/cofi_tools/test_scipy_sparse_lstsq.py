import numpy as np
import scipy
import pytest

from cofi.tools import ScipySparseLstSq
from cofi import BaseProblem, InversionOptions


@pytest.fixture
def problem_setup():
    A = scipy.sparse.csr_matrix([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
    b = np.array([2., 4., -1.])
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(A)
    inv_problem.set_data(b)
    return inv_problem


def test_validate_algorithm(problem_setup):
    inv_options = InversionOptions()
    inv_options.set_tool("scipy.sparse.linalg")
    inv_options.set_params(algorithm="spsolvee")
    with pytest.raises(ValueError, match=".*algorithm.*"):
        solver = ScipySparseLstSq(problem_setup, inv_options)
    inv_options.set_params(algorithm_params="1", algorithm="spsolve")
    with pytest.raises(ValueError, match=".*algorithm_params.*"):
        solver = ScipySparseLstSq(problem_setup, inv_options)

def test_run_all_algorithms(problem_setup):
    for algorithm in ScipySparseLstSq.available_algorithms():
        inv_options = InversionOptions()
        inv_options.set_tool("scipy.sparse.linalg")
        inv_options.set_params(algorithm=algorithm)
        solver = ScipySparseLstSq(problem_setup, inv_options)
        res = solver()
        assert res["success"]
        assert np.allclose(
            problem_setup.jacobian(1) @ res["model"], problem_setup.data, atol=1e-5
        )
