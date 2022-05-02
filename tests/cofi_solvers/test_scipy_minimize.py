from operator import inv
import pytest

from cofi.solvers import ScipyOptMinSolver
from cofi import BaseProblem, InversionOptions


def test_run():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda x: x ** 2)
    inv_problem.set_initial_model(30)
    inv_options = InversionOptions()
    solver = ScipyOptMinSolver(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert pytest.approx(res["model"], abs=0.1) == 0

def test_components_used():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda x:x)
    inv_problem.set_initial_model(1)
    inv_options = InversionOptions()
    solver1 = ScipyOptMinSolver(inv_problem, inv_options)
    assert "initial_model" in solver1.components_used
    assert "objective" in solver1.components_used
    assert "gradient" not in solver1.components_used
    inv_problem.set_gradient(lambda x:x)
    solver2 = ScipyOptMinSolver(inv_problem, inv_options)
    assert "gradient" in solver2.components_used
