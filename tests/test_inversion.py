import numpy as np
import pytest

from cofi import Inversion, BaseProblem, InversionOptions
from cofi.solvers.base_solver import BaseSolver


@pytest.fixture
def polynomial_problem():
    inv_problem = BaseProblem()
    _x = np.array([1, 2, 3, 4, 5])
    _G = np.array([_x ** i for i in range(3)]).T
    _m_true = np.array([2, 1, 1])
    _y = _G @ _m_true
    inv_problem.set_data(_y)
    inv_problem.set_jacobian(_G)
    inv_problem.set_jacobian_times_vector(lambda m, x: _G @ x)
    inv_problem.set_hessian(_G.T @ _G)
    inv_problem.set_hessian_times_vector(lambda m, x: _G.T @ _G @ x)
    inv_options = InversionOptions()
    inv_options.set_tool("scipy.linalg.lstsq")
    return inv_problem, inv_options


def test_solve(polynomial_problem):
    runner = Inversion(*polynomial_problem)
    inv_result = runner.run()
    assert inv_result.success


def test_runner_result_summary(polynomial_problem, capsys):
    runner = Inversion(*polynomial_problem)
    # 0
    runner.summary()
    console_output = capsys.readouterr()
    assert "scipy.linalg.lstsq" in console_output.out
    assert "Inversion hasn't started" in console_output.out
    # 1
    inv_result = runner.run()
    assert "success" in str(inv_result)
    inv_result.summary()
    console_output = capsys.readouterr()
    assert "SUCCESS" in console_output.out
    assert "model" in console_output.out
    runner.summary()
    console_output = capsys.readouterr()
    assert "SUCCESS" in console_output.out


def test_dispatch_custom_solver(polynomial_problem, capsys):
    inv_problem, inv_options = polynomial_problem
    class CustomSolver(BaseSolver):
        def __call__(self) -> dict:
            return {"successful": True}
    inv_options.set_tool(CustomSolver)
    runner = Inversion(inv_problem, inv_options)
    with pytest.raises(ValueError):
        res = runner.run()
