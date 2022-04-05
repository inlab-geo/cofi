import pytest

from cofi.solvers import ScipyOptMinSolver
from cofi import BaseProblem, InversionOptions


def test_run():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda x: x**2)
    inv_problem.set_initial_model(30)
    inv_options = InversionOptions()
    solver = ScipyOptMinSolver(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert pytest.approx(res["model"], abs=0.1) == 0
