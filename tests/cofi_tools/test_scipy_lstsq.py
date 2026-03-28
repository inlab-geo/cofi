import numpy as np
import pytest

from cofi.tools import ScipyLstSq
from cofi import BaseProblem, InversionOptions


@pytest.fixture
def problem_setup():
    x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
    y = np.array([0.8, 2.6, 3.5, 4.1, 4.8, 6.6, 8.6])
    G = x[:, np.newaxis] ** np.arange(2)
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(G)
    inv_problem.set_data(y)
    Cdinv = np.diag(np.array([5, 10, 100, 10, 5, 2.5, 10]))
    lamda = 1
    L = np.eye(2)
    reg_func = lambda m: lamda * (L @ m).T @ (L @ m)
    return inv_problem, Cdinv, lamda, L, reg_func


def test_validate_jac():
    # 1
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(lambda x: np.array([[1, 2, 3]]))
    inv_problem.set_hessian(lambda x: x)
    inv_problem.set_data(np.array([2]))
    inv_options = InversionOptions()
    inv_solver = ScipyLstSq(inv_problem, inv_options)
    # 2
    inv_problem.set_model_shape((1, 2))
    inv_solver = ScipyLstSq(inv_problem, inv_options)
    # 3
    inv_problem.set_initial_model(np.ones((1, 2)))
    inv_solver = ScipyLstSq(inv_problem, inv_options)
    # 4
    inv_problem.set_jacobian(lambda: np.array([1]))
    with pytest.raises(ValueError, match=".*isn't set properly.*"):
        inv_solver = ScipyLstSq(inv_problem, inv_options)


def test_run(problem_setup):
    inv_problem, _, _, _, _ = problem_setup
    inv_options = InversionOptions()
    solver = ScipyLstSq(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert res["model"][0] == pytest.approx(0, abs=0.1)
    assert res["model"][1] == pytest.approx(1, abs=0.1)


def test_uncertainty_cov(problem_setup):
    inv_problem, Cdinv, _, _, _ = problem_setup
    inv_problem.set_data_covariance_inv(Cdinv)
    inv_options = InversionOptions()
    solver = ScipyLstSq(inv_problem, inv_options)
    res = solver()
    _G = inv_problem.jacobian(1)
    _d = inv_problem.data
    assert res["success"]
    assert "model_covariance" in res
    assert np.allclose(solver._a, _G.T @ Cdinv @ _G)
    assert np.allclose(solver._b, _G.T @ Cdinv @ _d)
    assert np.allclose(res["model_covariance"], np.linalg.inv(solver._a))
    assert res["model"][0] == pytest.approx(0, abs=0.05)
    assert res["model"][1] == pytest.approx(1, abs=0.05)


def test_uncertainty_cov_inv(problem_setup):
    inv_problem, Cdinv, _, _, _ = problem_setup
    _G = inv_problem.jacobian(1)
    _d = inv_problem.data

    Cd = np.linalg.inv(Cdinv)
    inv_problem.set_data_covariance(Cd)
    inv_options = InversionOptions()
    solver = ScipyLstSq(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert "model_covariance" in res
    assert np.allclose(solver._a, _G.T @ Cdinv @ _G)
    assert np.allclose(solver._b, _G.T @ Cdinv @ _d)
    assert np.allclose(res["model_covariance"], np.linalg.inv(solver._a))
    assert res["model"][0] == pytest.approx(0, abs=0.05)
    assert res["model"][1] == pytest.approx(1, abs=0.05)

    full_Cdinv = Cdinv.copy()
    full_Cdinv[0, 1] = 1
    full_Cdinv[1, 0] = 1
    full_Cd = np.linalg.inv(full_Cdinv)
    inv_problem.set_data_covariance(full_Cd)
    solver = ScipyLstSq(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert "model_covariance" in res
    assert np.allclose(solver._a, _G.T @ full_Cdinv @ _G)
    assert np.allclose(solver._b, _G.T @ full_Cdinv @ _d)
    assert np.allclose(res["model_covariance"], np.linalg.inv(solver._a))

    Cdinv[0, 1] = 1
    Cdinv[1, 0] = 1
    inv_problem.set_data_covariance_inv(Cdinv)
    solver = ScipyLstSq(inv_problem, inv_options)
    assert np.allclose(solver._a, _G.T @ Cdinv @ _G)
    assert np.allclose(solver._b, _G.T @ Cdinv @ _d)


def test_tikhonov(problem_setup):
    inv_problem, Cdinv, lamda, L, reg_func = problem_setup
    inv_problem.set_data_covariance_inv(Cdinv)
    # 1
    inv_problem.set_regularization(reg_func)
    inv_options = InversionOptions()
    solver = ScipyLstSq(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert "model_covariance" in res
    assert res["model"][0] == pytest.approx(0, abs=0.02)
    assert res["model"][1] == pytest.approx(1, abs=0.02)
    # 2
    inv_problem.set_regularization(reg_func, L)
    inv_options = InversionOptions()
    solver = ScipyLstSq(inv_problem, inv_options)
    res = solver()
    assert res["success"]
    assert "model_covariance" in res
    assert res["model"][0] == pytest.approx(0, abs=0.02)
    assert res["model"][1] == pytest.approx(1, abs=0.02)
    # 3
    inv_problem.set_regularization(reg_func, lambda: np.array([1]))
    with pytest.raises(ValueError, match=r".*isn't set properly.*"):
        inv_solver = ScipyLstSq(inv_problem, inv_options)
