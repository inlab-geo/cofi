import numpy as np

from cofi.solvers import CoFISimpleNewtonSolver
from cofi import BaseProblem, InversionOptions, Inversion


inv_problem = BaseProblem()
inv_problem.set_objective(lambda x: (x-3)**2)
inv_problem.set_initial_model(30)
inv_problem.set_gradient(lambda x: 2*x - 6)
inv_problem.set_hessian(lambda x: 2)
inv_options = InversionOptions()
inv_options.set_params(max_iterations=4)
inv_options.set_tool("cofi.simple_newton")

def test_run():
    solver = CoFISimpleNewtonSolver(inv_problem, inv_options)
    res = solver()
    assert np.array_equal(res["model"], np.array([[3.]]))

def test_inv_run():
    inv = Inversion(inv_problem, inv_options)
    res = inv.run()
    res.summary()
    assert res.success
    assert np.array_equal(res.model, np.array([[3.]]))
