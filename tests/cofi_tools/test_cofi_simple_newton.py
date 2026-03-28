import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

from cofi.tools import CoFISimpleNewton
from cofi import BaseProblem, InversionOptions, Inversion


@pytest.fixture
def problem_setup():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda x: (x - 3) ** 2)
    inv_problem.set_initial_model(30)
    inv_problem.set_gradient(lambda x: 2 * x - 6)
    inv_problem.set_hessian(lambda x: 2)
    inv_options = InversionOptions()
    inv_options.set_params(num_iterations=4)
    inv_options.set_tool("cofi.simple_newton")
    return inv_problem, inv_options

def test_run(problem_setup):
    inv_problem, inv_options = problem_setup
    solver = CoFISimpleNewton(inv_problem, inv_options)
    res = solver()
    assert res["model"] == 3.0
    assert inv_problem.initial_model == 30


def test_inv_run(problem_setup):
    inv_problem, inv_options = problem_setup
    inv = Inversion(inv_problem, inv_options)
    res = inv.run()
    res.summary()
    assert res.success
    assert res.model == 3.0
    assert inv_problem.initial_model == 30


def test_not_inplace(problem_setup):
    inv_problem, inv_options = problem_setup
    inv_problem.set_initial_model(np.array([[30.0]]))
    solver = CoFISimpleNewton(inv_problem, inv_options)
    res = solver()
    assert res["model"] == 3.0
    assert inv_problem.initial_model == 30.0

def test_with_stopping_criteria_obj(problem_setup):
    inv_problem, inv_options = problem_setup
    inv_problem.set_initial_model(np.array([[30.0]]))
    inv_options.set_params(param_tol=0)
    solver = CoFISimpleNewton(inv_problem, inv_options)
    res = solver()
    assert res["n_obj_evaluations"] == 3
    assert res["n_grad_evaluations"] == 2
    assert res["n_hess_evaluations"] == 2

def test_with_stopping_criteria_param(problem_setup):
    inv_problem, inv_options = problem_setup
    inv_problem.set_initial_model(np.array([[30.0]]))
    inv_options.set_params(obj_tol=0)
    solver = CoFISimpleNewton(inv_problem, inv_options)
    res = solver()
    assert res["n_obj_evaluations"] == 3
    assert res["n_grad_evaluations"] == 2
    assert res["n_hess_evaluations"] == 2

def test_no_stopping_criteria(problem_setup):
    inv_problem, inv_options = problem_setup
    inv_problem.set_initial_model(np.array([[30.0]]))
    inv_options.set_params(obj_tol=0, param_tol=0)
    solver = CoFISimpleNewton(inv_problem, inv_options)
    res = solver()
    assert res["n_obj_evaluations"] == 5
    assert res["n_grad_evaluations"] == 4
    assert res["n_hess_evaluations"] == 4

def test_symmetric_hessian(monkeypatch):
    hess = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda x: 0.5 * x.T @ hess @ x - rhs @ x)
    inv_problem.set_initial_model(np.array([0.0, 0.0]))
    inv_problem.set_gradient(lambda x: hess @ x - rhs)
    inv_problem.set_hessian(lambda x: hess)
    inv_options = InversionOptions()
    inv_options.set_tool("cofi.simple_newton")
    inv_options.set_params(num_iterations=1, hessian_is_symmetric=True)

    solve_kwargs = {}
    real_solve = scipy.linalg.solve

    def wrapped_solve(*args, **kwargs):
        solve_kwargs.update(kwargs)
        return real_solve(*args, **kwargs)

    def fail_minres(*args, **kwargs):
        raise AssertionError("minres should not be called for dense symmetric Hessians")

    monkeypatch.setattr(scipy.linalg, "solve", wrapped_solve)
    monkeypatch.setattr(scipy.sparse.linalg, "minres", fail_minres)

    solver = CoFISimpleNewton(inv_problem, inv_options)
    res = solver()

    expected_model = np.linalg.solve(hess, rhs)

    assert solve_kwargs["assume_a"] == "sym"
    assert np.allclose(res["model"], expected_model)
