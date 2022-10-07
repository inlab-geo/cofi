import os

import pytest
import numpy as np

from cofi import BaseProblem
from cofi.exceptions import (
    DimensionMismatchError, 
    InvalidOptionError,
    InvocationError, 
    NotDefinedError
)

############### TEST data loader ######################################################
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


############### TEST empty problem ####################################################
def test_non_set():
    inv_problem = BaseProblem()
    with pytest.raises(NotDefinedError):
        inv_problem.objective(1)
    with pytest.raises(NotDefinedError):
        inv_problem.gradient(1)
    with pytest.raises(NotDefinedError):
        inv_problem.hessian(1)
    with pytest.raises(NotDefinedError):
        inv_problem.hessian_times_vector(1, 2)
    with pytest.raises(NotDefinedError):
        inv_problem.residual(1)
    with pytest.raises(NotDefinedError):
        inv_problem.jacobian(1)
    with pytest.raises(NotDefinedError):
        inv_problem.jacobian_times_vector(1, 2)
    with pytest.raises(NotDefinedError):
        inv_problem.data_misfit(1)
    with pytest.raises(NotDefinedError):
        inv_problem.regularization(1)
    with pytest.raises(NotDefinedError):
        inv_problem.regularization_matrix(1)
    with pytest.raises(NotDefinedError):
        inv_problem.forward(1)
    with pytest.raises(NotDefinedError):
        inv_problem.data
    with pytest.raises(NotDefinedError):
        inv_problem.data_covariance
    with pytest.raises(NotDefinedError):
        inv_problem.data_covariance_inv
    with pytest.raises(NotDefinedError):
        inv_problem.initial_model
    with pytest.raises(NotDefinedError):
        inv_problem.model_shape
    with pytest.raises(NotDefinedError):
        inv_problem.bounds
    with pytest.raises(NotDefinedError):
        inv_problem.constraints
    with pytest.raises(NotDefinedError):
        inv_problem.log_posterior(1)
    with pytest.raises(NotDefinedError):
        inv_problem.log_prior(1)
    with pytest.raises(NotDefinedError):
        inv_problem.log_likelihood(1)
    with pytest.raises(NotDefinedError):
        inv_problem.walkers_starting_pos
    with pytest.raises(NotDefinedError):
        inv_problem.log_posterior_with_blobs(1)
    with pytest.raises(NotDefinedError):
        inv_problem.blobs_dtype
    with pytest.raises(NotDefinedError):
        inv_problem.regularization_factor
    assert not inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.hessian_times_vector_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.jacobian_times_vector_defined
    assert not inv_problem.data_misfit_defined
    assert not inv_problem.regularization_defined
    assert not inv_problem.regularization_matrix_defined
    assert not inv_problem.forward_defined
    assert not inv_problem.data_defined
    assert not inv_problem.data_covariance_defined
    assert not inv_problem.data_covariance_inv_defined
    assert not inv_problem.initial_model_defined
    assert not inv_problem.model_shape_defined
    assert not inv_problem.bounds_defined
    assert not inv_problem.constraints_defined
    assert not inv_problem.log_posterior_defined
    assert not inv_problem.log_prior_defined
    assert not inv_problem.log_likelihood_defined
    assert not inv_problem.walkers_starting_pos_defined
    assert not inv_problem.log_posterior_with_blobs_defined
    assert not inv_problem.blobs_dtype_defined
    assert not inv_problem.regularization_factor_defined
    assert len(inv_problem.defined_components()) == 0
    inv_problem.summary()


def test_x_set():
    inv_problem = BaseProblem()
    inv_problem._data_x = np.array([1, 2, 3])
    assert not inv_problem.data_defined


############### TEST set methods Tier 3 ###############################################
def test_set_obj():
    inv_problem = BaseProblem()
    _x = np.array([1, 2, 3, 4, 5])
    _forward = lambda m, x_i: m[0] + m[1] * x_i + m[2] * x_i ** 2
    _forward_true = lambda x_i: 2 + x_i + x_i ** 2
    _y_true = np.vectorize(_forward_true)(_x)
    inv_problem.set_objective(
        lambda m: np.linalg.norm(_y_true - _forward(m, _x)) / _x.shape[0]
    )  # se
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
    assert inv_problem.objective(np.array([2, 1, 1])) == 0
    assert pytest.approx(inv_problem.objective(np.array([2, 1, 2]))) == 6.25779513


############### TEST set methods Tier 2 ###############################################
@pytest.fixture
def inv_problem_with_misfit():
    inv_problem = BaseProblem()
    _x = np.array([1, 2, 3, 4, 5])
    _forward = lambda m, x_i: m[0] + m[1] * x_i + m[2] * x_i ** 2
    _forward_true = lambda x_i: 2 + x_i + x_i ** 2
    _data_misfit = lambda m: np.linalg.norm(_y_true - _forward(m, _x)) / _x.shape[0]
    _y_true = np.vectorize(_forward_true)(_x)
    inv_problem.set_data_misfit(_data_misfit)
    return inv_problem


def check_defined_misfit_reg(inv_problem):
    inv_problem.summary()
    assert inv_problem.data_misfit_defined
    assert inv_problem.regularization_defined
    assert inv_problem.regularization_factor_defined
    assert inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.regularization_matrix_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.data_defined
    assert not inv_problem.forward_defined
    assert len(inv_problem.defined_components()) == 4


def test_set_misfit_reg(inv_problem_with_misfit):
    inv_problem_with_misfit.summary()
    inv_problem_with_misfit.set_regularization(lambda m: m.T @ m, 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2, 1, 1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert inv_problem_with_misfit.regularization(true_model) == (4 + 1 + 1) * 0.5
    assert inv_problem_with_misfit.objective(true_model) == (4 + 1 + 1) * 0.5
    worse_model = np.array([2, 1, 2])
    assert (
        pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2, 1, 2])))
        == 6.25779513
    )
    assert inv_problem_with_misfit.regularization(worse_model) == (4 + 1 + 4) * 0.5
    assert (
        pytest.approx(inv_problem_with_misfit.objective(np.array([2, 1, 2])))
        == 6.25779513 + (4 + 1 + 4) * 0.5
    )


def test_set_misfit_reg_L0(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularization(0, 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2, 1, 1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert inv_problem_with_misfit.regularization(true_model) == 3 * 0.5
    assert inv_problem_with_misfit.objective(true_model) == 3 * 0.5
    worse_model = np.array([2, 1, 2])
    assert (
        pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2, 1, 2])))
        == 6.25779513
    )
    assert inv_problem_with_misfit.regularization(worse_model) == 3 * 0.5
    assert (
        pytest.approx(inv_problem_with_misfit.objective(np.array([2, 1, 2])))
        == 6.25779513 + 3 * 0.5
    )


def test_set_misfit_reg_L1(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularization(1, 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2, 1, 1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert inv_problem_with_misfit.regularization(true_model) == 4 * 0.5
    assert inv_problem_with_misfit.objective(true_model) == 4 * 0.5
    worse_model = np.array([2, 1, 2])
    assert (
        pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2, 1, 2])))
        == 6.25779513
    )
    assert inv_problem_with_misfit.regularization(worse_model) == 5 * 0.5
    assert (
        pytest.approx(inv_problem_with_misfit.objective(np.array([2, 1, 2])))
        == 6.25779513 + 5 * 0.5
    )


def test_set_misfit_reg_l2(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularization(2, 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2, 1, 1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert (
        inv_problem_with_misfit.regularization(true_model) == np.sqrt(4 + 1 + 1) * 0.5
    )
    assert inv_problem_with_misfit.objective(true_model) == np.sqrt(4 + 1 + 1) * 0.5
    worse_model = np.array([2, 1, 2])
    assert (
        pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2, 1, 2])))
        == 6.25779513
    )
    assert (
        inv_problem_with_misfit.regularization(worse_model) == np.sqrt(4 + 1 + 4) * 0.5
    )
    assert (
        pytest.approx(inv_problem_with_misfit.objective(np.array([2, 1, 2])))
        == 6.25779513 + np.sqrt(4 + 1 + 4) * 0.5
    )

def test_set_misfit_reg_inf(inv_problem_with_misfit):
    # inf norm
    inv_problem_with_misfit.set_regularization("inf", 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2, 1, 1])
    assert inv_problem_with_misfit.regularization(true_model) == 1
    worse_model = np.array([2, 1, 2])
    assert inv_problem_with_misfit.regularization(worse_model) == 1
    # -inf norm
    inv_problem_with_misfit.set_regularization("-inf", 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2, 1, 1])
    assert inv_problem_with_misfit.regularization(true_model) == 0.5
    worse_model = np.array([2, 1, 2])
    assert inv_problem_with_misfit.regularization(worse_model) == 0.5
    

def test_invalid_reg_options():
    inv_problem = BaseProblem()
    with pytest.raises(InvalidOptionError, match=r".*the regularization order you've entered.*"):
        inv_problem.set_regularization("FOO")
    with pytest.raises(InvalidOptionError, match=r".*is invalid, please choose from the following:.*"):
        inv_problem.set_regularization(-1)


############### TEST set methods Tier 1 ###############################################
@pytest.fixture
def inv_problem_with_data():
    inv_problem = BaseProblem()
    _x = np.array([1, 2, 3, 4, 5])
    _y = np.vectorize(lambda x_i: 2 + x_i + x_i ** 2)(_x)
    inv_problem.set_data(_y)
    forward = lambda m: np.polynomial.Polynomial(m)(_x)
    return inv_problem, forward


def check_defined_data_fwd_misfit_reg(inv_problem):
    inv_problem.summary()
    assert inv_problem.data_defined
    assert inv_problem.forward_defined
    assert inv_problem.data_misfit_defined
    assert inv_problem.residual_defined
    assert inv_problem.regularization_defined
    assert inv_problem.regularization_factor_defined
    assert inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.jacobian_defined
    assert len(inv_problem.defined_components()) == 7


def check_values_data_fwd_misfit_reg(inv_problem):
    inv_problem.set_regularization(1, 0.5)
    true_model = np.array([2, 1, 1])
    assert inv_problem.data_misfit(true_model) == 0
    assert inv_problem.regularization(true_model) == 4 * 0.5
    assert inv_problem.objective(true_model) == 4 * 0.5
    worse_model = np.array([1, 1, 1])
    assert pytest.approx(inv_problem.data_misfit(worse_model)) == 5
    assert inv_problem.regularization(worse_model) == 3 * 0.5
    assert pytest.approx(inv_problem.objective(worse_model)) == 5 + 3 * 0.5


def test_set_data_fwd_misfit_inbuilt_reg_inbuilt(inv_problem_with_data):
    inv_problem, forward = inv_problem_with_data
    inv_problem.set_forward(forward)
    inv_problem.set_data_misfit("squared error")
    inv_problem.set_regularization(1)
    check_defined_data_fwd_misfit_reg(inv_problem)
    check_values_data_fwd_misfit_reg(inv_problem)

def test_set_data_cov_fwd_mistift_inbuilt(inv_problem_with_data):
    inv_problem, forward = inv_problem_with_data
    inv_problem.set_forward(forward)
    inv_problem.set_data_misfit("squared error")
    inv_problem.set_regularization(1)
    # 1
    inv_problem.set_data_covariance(np.diag(np.array([1,2,1,2,1])))
    assert inv_problem.data_misfit(np.array([2,1,1])) == 0
    assert inv_problem.data_misfit(np.array([1,1,1])) == 3.5 
    # 2
    Cd_inv = np.diag(np.array([1,0.5,1,0.5,1]))
    inv_problem.set_data_covariance_inv(Cd_inv)
    assert inv_problem.data_misfit(np.array([1,1,1])) == 3.5 
    # 3
    Cd_inv[0,3] = 1
    inv_problem.set_data_covariance_inv(Cd_inv)
    assert inv_problem.data_misfit(np.array([1,1,1])) == 5 


def test_invalid_misfit_options():
    inv_problem = BaseProblem()
    with pytest.raises(InvalidOptionError):
        inv_problem.set_data_misfit("FOO")
    inv_problem.set_data_misfit("squared error")
    with pytest.raises(InvocationError):
        inv_problem.data_misfit(np.array([1, 2, 3]))


############### TEST set methods for sampling #########################################
_dummy_dist = lambda m: -np.inf if (m[0]<0 or m[0]>1) else 0
_dummy_dist_with_blobs = lambda m: (_dummy_dist(m), m, 2*m)
_blobs_dtype = [("model", float), ("model_doubled", float)]
def test_prior_likelihood():
    inv_problem = BaseProblem()
    inv_problem.set_log_prior(_dummy_dist)
    inv_problem.set_log_likelihood(_dummy_dist)
    assert inv_problem.log_prior_defined
    assert inv_problem.log_likelihood_defined
    assert inv_problem.log_posterior_defined

def test_posterior():
    inv_problem = BaseProblem()
    inv_problem.set_log_posterior(_dummy_dist)
    assert not inv_problem.log_prior_defined
    assert not inv_problem.log_likelihood_defined
    assert inv_problem.log_posterior_defined

def test_posterior_with_blobs():
    inv_problem = BaseProblem()
    inv_problem.set_log_posterior_with_blobs(_dummy_dist_with_blobs, _blobs_dtype)
    assert not inv_problem.log_prior_defined
    assert not inv_problem.log_likelihood_defined
    assert inv_problem.log_posterior_defined
    assert inv_problem.log_posterior_with_blobs_defined
    assert inv_problem.blobs_dtype_defined


############### TEST properties #######################################################
def test_check_defined():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda a: a + 1)
    assert inv_problem.objective_defined
    assert str(inv_problem) == "BaseProblem"
    inv_problem.name = "AnotherProblem"
    assert str(inv_problem) == "AnotherProblem"
    inv_problem.set_initial_model(np.array([1, 2, 3]))
    assert inv_problem.initial_model_defined
    assert inv_problem.model_shape_defined
    assert inv_problem.model_shape == (3,)
    with pytest.raises(DimensionMismatchError, match=r".*the model shape you've provided.*"):
        inv_problem.set_model_shape((2, 1))
    inv_problem.set_model_shape((3, 1))

def test_set_data():
    inv_problem = BaseProblem()
    inv_problem.set_data(np.ones((2,1)), np.zeros((2,2)), np.zeros((2,2)))
    assert inv_problem.data_defined
    assert inv_problem.data_covariance_defined
    assert inv_problem.data_covariance_inv_defined
    inv_problem.set_data_covariance(np.ones((2,2)))
    assert inv_problem.data_covariance_defined
    assert inv_problem.data_covariance[0,0] == 1


############### TEST suggest_solvers ##################################################
def test_suggest_solvers(capsys):
    inv_problem = BaseProblem()
    # 0
    inv_problem.suggest_solvers()
    console_output = capsys.readouterr().out
    assert "scipy.optimize.minimize" not in console_output
    assert "scipy.linalg.lstsq" not in console_output
    # 1
    inv_problem.set_initial_model(1)
    inv_problem.set_objective(lambda x: x)
    inv_problem.suggest_solvers()
    console_output = capsys.readouterr().out
    assert "scipy.optimize.minimize" in console_output
    assert "scipy.optimize.least_squares" not in console_output
    assert "scipy.linalg.lstsq" not in console_output
    # 2
    inv_problem.set_jacobian(np.array([1]))
    inv_problem.set_data(2)
    inv_problem.suggest_solvers()
    console_output = capsys.readouterr().out
    assert "scipy.linalg.lstsq" in console_output


def test_suggest_solvers_return():
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(np.array([1]))
    inv_problem.set_data(2)
    suggested = inv_problem.suggest_solvers(print_to_console=False)
    assert "scipy.linalg.lstsq" in suggested["matrix solvers"]


############### TEST function wrapper #################################################
def test_wrapping_objective():
    inv_problem = BaseProblem()
    # 1
    inv_problem.set_objective(lambda a,b,c: a+b+c, args=[1], kwargs={"c":2})
    assert inv_problem.objective(1) == 4
    # 2
    inv_problem.set_forward(lambda a,b,c=3,d=2: a+b+c+d, args=[3,2], kwargs={"d":1})
    assert inv_problem.forward(1) == 7

def test_not_overwriting_by_autogen():
    inv_problem = BaseProblem()
    # 1
    inv_problem.set_log_likelihood(lambda _: 1)
    inv_problem.set_log_prior(lambda _: 2)
    assert inv_problem.log_posterior_defined
    assert inv_problem.log_posterior(1) == 3
    # 2
    inv_problem.set_log_posterior(lambda _: 4)
    assert inv_problem.log_posterior(1) == 4
    inv_problem.set_log_prior(lambda _: 5)
    assert inv_problem.log_posterior(1) == 4

def test_set_reg_with_args():
    inv_problem = BaseProblem()
    from scipy.sparse import csr_matrix
    A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    inv_problem.set_regularization(lambda m, A: A @ m.T @ m, 2, args=[A])
    inv_problem.regularization(np.array([1,2,3]))

def test_invalid_func():
    inv_problem = BaseProblem()
    with pytest.raises(InvalidOptionError):
        inv_problem.set_forward(1)

############### TEST regularization (matrix, factor) ##################################
def test_set_reg_with_matrix():
    inv_problem = BaseProblem()
    # test zero info
    assert not inv_problem.regularization_defined
    assert not inv_problem.regularization_factor_defined
    assert not inv_problem.regularization_matrix_defined
    # test set
    inv_problem.set_regularization(2, 0.5, np.array([[2,0],[0,1]]))
    assert inv_problem.regularization_defined
    assert inv_problem.regularization_factor_defined
    assert inv_problem.regularization_matrix_defined
    assert inv_problem.regularization(np.array([1,1])) == 1.118033988749895
    # test unset
    inv_problem.set_regularization(2)
    assert inv_problem.regularization_defined
    assert inv_problem.regularization_factor == 1
    assert not inv_problem.regularization_matrix_defined

def test_set_reg_with_matrix_func():
    inv_problem = BaseProblem()
    inv_problem.set_regularization(2, 0.5, lambda _: np.array([[2,0], [0,1]]))
    assert inv_problem.regularization(np.array([1,1])) == 1.118033988749895


############### TEST model covariance #################################################
def test_model_cov():
    inv_problem = BaseProblem()
    sigma = 1.0
    Cdinv = np.eye(100)/(sigma**2)
    inv_problem.set_data_covariance_inv(Cdinv)
    with pytest.raises(NotDefinedError, match=r".*`jacobian` is required.*"):
        inv_problem.model_covariance_inv(None)
    inv_problem.set_jacobian(np.array([[n**i for i in range(2)] for n in range(100)]))
    inv_problem.model_covariance(None)


############### TEST jac/hess times vector (auto generated) ###########################
def test_hess_times_vector():
    inv_problem = BaseProblem()
    # 1
    _hess = np.array([[1,0],[0,2]])
    inv_problem.set_hessian(_hess)
    _test_res = inv_problem.hessian_times_vector(0, np.array([1,1]))
    assert np.array_equal(_test_res, np.array([1,2]))
    # 2
    inv_problem.set_hessian(lambda _: _hess)
    _test_res = inv_problem.hessian_times_vector(0, np.array([1,1]))
    assert np.array_equal(_test_res, np.array([1,2]))


############### TEST auto generated methods ###########################################
def test_obj_from_dm_reg():
    inv_problem = BaseProblem()
    # test valid
    inv_problem.set_data_misfit(lambda x: x**2)
    inv_problem.set_regularization(lambda x: x)
    assert inv_problem.objective(1) == 2
    # test invalid
    inv_problem.set_data_misfit(lambda x: x[2])
    with pytest.raises(InvocationError, match=r".*exception while calling auto-generated objective.*"):
        inv_problem.objective(1)

def test_obj_from_dm():
    inv_problem = BaseProblem()
    # test invalid
    inv_problem.set_data_misfit(lambda x: x[2])
    with pytest.raises(InvocationError):
        inv_problem.objective(1)

def test_lp_from_ll_lp():
    inv_problem = BaseProblem()
    # test valid
    inv_problem.set_log_likelihood(lambda x: x**2)
    inv_problem.set_log_prior(lambda x: x)
    assert inv_problem.log_posterior(1) == 2
    # test invalid
    inv_problem.set_log_likelihood(lambda x: x[2])
    with pytest.raises(InvocationError):
        inv_problem.log_posterior(1)

def test_hessp_from_hess():
    inv_problem = BaseProblem()
    # test invalid
    inv_problem.set_hessian(np.array([1,2]))
    with pytest.raises(InvocationError):
        inv_problem.hessian_times_vector(0, np.array([1,2,3]))

def test_jacp_from_jac():
    inv_problem = BaseProblem()
    # test invalid
    inv_problem.set_jacobian(np.array([1,2]))
    with pytest.raises(InvocationError):
        inv_problem.jacobian_times_vector(0, np.array([1,2,3]))

def test_res_from_fwd_dt():
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda x: x**2)
    inv_problem.set_data(np.array([1,4,9]))
    with pytest.raises(InvocationError):
        inv_problem.residual(np.array([1,2]))
