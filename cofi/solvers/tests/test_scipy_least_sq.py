import pytest
import numpy as np
import matplotlib.pyplot as plt

from cofi.solvers import ScipyOptLstSqSolver
from cofi import BaseProblem, InversionOptions

def test_run():
    inv_problem = BaseProblem()
    forward_func = lambda m, x: np.array([x**i for i in range(3)]).T @ m
    _m_true = np.array([0,0,1])
    x = np.random.choice(np.linspace(-5,5),20)
    y = forward_func(_m_true, x) + np.random.normal(0,1,20)
    inv_problem.set_dataset(x, y)
    inv_problem.set_forward(lambda m: forward_func(m, x))
    inv_problem.set_initial_model(np.array([0,0,0]))
    inv_options = InversionOptions()
    solver = ScipyOptLstSqSolver(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert pytest.approx(res["model"], abs=0.5) == np.array([0,0,1])
