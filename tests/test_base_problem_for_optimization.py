import os
import pytest
import numpy

from cofi import BaseProblem
import cofi.utils as utils
from cofi._exceptions import (
    InvalidOptionError, 
    InvocationError, 
    NotDefinedError, 
)


# ---------------- data ---------------------------------------------------------------
data_files_to_test = [
    "datasets/dummy_test1_comma.txt",
    "datasets/dummy_test2_tab.txt",
    "datasets/dummy_test3_idx.txt",
    "datasets/dummy_test4_npy.npy",
    "datasets/dummy_test5_pickle.pickle",
    "datasets/dummy_test6_pickle.pkl",
]


@pytest.fixture(params=data_files_to_test)
def data_path(request):
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.split(path_to_current_file)[0]
    data_path = os.path.join(current_directory, request.param)
    return data_path


def test_set_data_from_file(data_path):
    inv_problem = BaseProblem()
    if "idx" in data_path:
        inv_problem.set_data_from_file(data_path, 0)
    else:
        inv_problem.set_data_from_file(data_path)


def test_set_data_with_uncertainty():
    inv_problem = BaseProblem()
    inv_problem.set_data(numpy.ones((2,1)), numpy.zeros((2,2)), numpy.zeros((2,2)))
    assert inv_problem.data_defined
    assert inv_problem.data_covariance_defined
    assert inv_problem.data_covariance_inv_defined
    inv_problem.set_data_covariance(numpy.ones((2,2)))
    assert inv_problem.data_covariance_defined
    assert inv_problem.data_covariance[0,0] == 1


# ---------------- objective ----------------------------------------------------------
@pytest.fixture
def problem_objective_setup():
    inv_problem = BaseProblem()
    _x = numpy.array([1, 2, 3, 4, 5])
    _fwd = lambda m: m[0] + m[1] * _x + m[2] * _x**2
    _m_true = numpy.array([2, 1, 1])
    _y = _fwd(_m_true)
    _dt_misfit = lambda m: numpy.sum((_fwd(m) - _y)**2)
    _reg = lambda m: 0.5 * m.T @ m
    return inv_problem, _fwd, _y, _dt_misfit, _reg


def test_set_obj(problem_objective_setup):
    # setup instance
    inv_problem, _, _, _dt_misfit, _ = problem_objective_setup
    inv_problem.set_objective(_dt_misfit)
    # check
    assert inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.data_misfit_defined
    assert not inv_problem.regularization_defined
    assert not inv_problem.forward_defined
    assert not inv_problem.data_defined
    assert len(inv_problem.defined_components()) == 1
    assert inv_problem.objective(numpy.array([2, 1, 1])) == 0
    assert inv_problem.objective(numpy.array([2, 1, 2])) == 979


# ---------------- data misfit --------------------------------------------------------
def test_set_data_misfit(problem_objective_setup, capsys):
    # setup instance
    inv_problem, _, _, _dt_misfit, _ = problem_objective_setup
    inv_problem.set_data_misfit(_dt_misfit)
    # check
    assert inv_problem.data_misfit_defined
    assert inv_problem.objective_defined
    assert len(inv_problem.defined_components()) == 2
    assert inv_problem.objective(numpy.array([2, 1, 1])) == 0
    inv_problem.summary()
    captured = capsys.readouterr()
    assert "you did not set regularization" in captured.out


# ---------------- regularization -----------------------------------------------------
def test_set_data_misfit_regularization(problem_objective_setup):
    # setup instance
    inv_problem, _, _, _dt_misfit, _reg = problem_objective_setup
    inv_problem.set_data_misfit(_dt_misfit)
    inv_problem.set_regularization(_reg)
    # check
    assert inv_problem.data_misfit_defined
    assert inv_problem.regularization_defined
    assert inv_problem.objective_defined
    assert len(inv_problem.defined_components()) == 3
    assert inv_problem.objective(numpy.array([2, 1, 1])) == 3
    assert inv_problem.objective(numpy.array([2, 1, 2])) == 983.5


def test_set_regularization_from_utils_quadratic_reg(problem_objective_setup):
    # setup instance
    inv_problem, _, _, _dt_misfit, _ = problem_objective_setup
    inv_problem.set_data_misfit(_dt_misfit)
    _my_reg_from_utils = utils.QuadraticReg(model_shape=(3,))
    inv_problem.set_regularization(_my_reg_from_utils)
    # check
    assert len(inv_problem.defined_components()) == 4
    assert inv_problem.regularization(numpy.array([2, 1, 1])) == 6
    assert inv_problem.regularization(numpy.array([2, 1, 2])) == 9


def test_set_regularization_from_utils_gaussian_prior(problem_objective_setup):
    # setup instance
    inv_problem, _, _, _dt_misfit, _ = problem_objective_setup
    inv_problem.set_data_misfit(_dt_misfit)
    _my_reg_from_utils = utils.GaussianPrior(((3,), 0.5), numpy.ones(3))
    inv_problem.set_regularization(_my_reg_from_utils)
    # check
    assert len(inv_problem.defined_components()) == 3
    assert numpy.isclose(inv_problem.regularization(numpy.array([2, 1, 1])), 8.2205933)
    assert numpy.isclose(inv_problem.regularization(numpy.array([2, 1, 2])), 16.441187)


# ---------------- forward + data -----------------------------------------------------
def test_set_forward_data(problem_objective_setup):
    # setup instance
    inv_problem, _fwd, _y, _, _ = problem_objective_setup
    inv_problem.set_forward(_fwd)
    inv_problem.set_data(_y)
    inv_problem.set_data_misfit("squared error")
    # check
    assert len(inv_problem.defined_components()) == 5
    assert inv_problem.data_misfit(numpy.array([2, 1, 1])) == 0


def test_set_forward_data_with_Cd(problem_objective_setup):
    # setup instance
    inv_problem, _fwd, _y, _, _ = problem_objective_setup
    inv_problem.set_forward(_fwd)
    inv_problem.set_data(_y)
    inv_problem.set_data_misfit("squared error")
    inv_problem.set_data_covariance(0.5 * numpy.eye(5))
    # check
    assert len(inv_problem.defined_components()) == 6
    assert inv_problem.data_misfit(numpy.array([2, 1, 1])) == 0
    assert inv_problem.data_misfit(numpy.array([2, 1, 2])) == 979 * 2


def test_set_forward_data_with_Cdinv(problem_objective_setup):
    # setup instance
    inv_problem, _fwd, _y, _, _ = problem_objective_setup
    inv_problem.set_forward(_fwd)
    inv_problem.set_data(_y)
    inv_problem.set_data_misfit("squared error")
    inv_problem.set_data_covariance_inv(0.5 * numpy.eye(5))
    # check
    assert len(inv_problem.defined_components()) == 6
    assert inv_problem.data_misfit(numpy.array([2, 1, 1])) == 0
    assert inv_problem.data_misfit(numpy.array([2, 1, 2])) == 979 / 2

def test_set_forward_data_with_Cdinv_non_diagonal(problem_objective_setup):
    # setup instance
    inv_problem, _fwd, _y, _, _ = problem_objective_setup
    inv_problem.set_forward(_fwd)
    inv_problem.set_data(_y)
    inv_problem.set_data_misfit("squared error")
    Cdinv = 0.5 * numpy.eye(5)
    Cdinv[0,1] = 1
    inv_problem.set_data_covariance_inv(Cdinv)
    # check
    assert len(inv_problem.defined_components()) == 6
    assert inv_problem.data_misfit(numpy.array([2, 1, 1])) == 0
    assert inv_problem.data_misfit(numpy.array([2, 1, 2])) == 493.5

def test_set_invalid_misfit_options():
    inv_problem = BaseProblem()
    with pytest.raises(InvalidOptionError):
        inv_problem.set_data_misfit("FOO")
    inv_problem.set_data_misfit("squared error")
    with pytest.raises(InvocationError):
        inv_problem.data_misfit(numpy.array([1, 2, 3]))


# ---------------- model covariance as extra information ------------------------------
def test_model_cov_from_data_cov():
    inv_problem = BaseProblem()
    sigma = 1.0
    Cdinv = numpy.eye(100)/(sigma**2)
    inv_problem.set_data_covariance_inv(Cdinv)
    with pytest.raises(NotDefinedError, match=r".*`jacobian` is called.*"):
        inv_problem.model_covariance_inv(None)
    inv_problem.set_jacobian(numpy.array([[n**i for i in range(2)] for n in range(100)]))
    inv_problem.model_covariance(None)


# ---------------- jac/hess times vector (auto generated) -----------------------------
def test_hess_times_vector():
    inv_problem = BaseProblem()
    # 1
    _hess = numpy.array([[1,0],[0,2]])
    inv_problem.set_hessian(_hess)
    _test_res = inv_problem.hessian_times_vector(0, numpy.array([1,1]))
    assert numpy.array_equal(_test_res, numpy.array([1,2]))
    # 2
    inv_problem.set_hessian(lambda _: _hess)
    _test_res = inv_problem.hessian_times_vector(0, numpy.array([1,1]))
    assert numpy.array_equal(_test_res, numpy.array([1,2]))
