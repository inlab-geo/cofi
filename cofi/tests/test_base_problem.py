import os

import pytest
import numpy as np

from cofi import BaseProblem


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

def test_set_dataset_from_file(data_path):
    inv_problem = BaseProblem()
    if "idx" in data_path:
        inv_problem.set_dataset_from_file(data_path, 0)
    else:
        inv_problem.set_dataset_from_file(data_path)


############### TEST empty problem ####################################################
def test_non_set():
    inv_problem = BaseProblem()
    with pytest.raises(NotImplementedError): inv_problem.objective(1)
    with pytest.raises(NotImplementedError): inv_problem.gradient(1)
    with pytest.raises(NotImplementedError): inv_problem.hessian(1)
    with pytest.raises(NotImplementedError): inv_problem.hessian_times_vector(1,2)
    with pytest.raises(NotImplementedError): inv_problem.residual(1)
    with pytest.raises(NotImplementedError): inv_problem.jacobian(1)
    with pytest.raises(NotImplementedError): inv_problem.jacobian_times_vector(1,2)
    with pytest.raises(NotImplementedError): inv_problem.data_misfit(1)
    with pytest.raises(NotImplementedError): inv_problem.regularisation(1)
    with pytest.raises(NotImplementedError): inv_problem.forward(1)
    with pytest.raises(NameError): inv_problem.data_x
    with pytest.raises(NameError): inv_problem.data_y
    with pytest.raises(NameError): inv_problem.initial_model
    with pytest.raises(NameError): inv_problem.model_shape
    with pytest.raises(NameError): inv_problem.bounds
    with pytest.raises(NameError): inv_problem.constraints
    assert not inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.hessian_times_vector_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.jacobian_times_vector_defined
    assert not inv_problem.data_misfit_defined
    assert not inv_problem.regularisation_defined
    assert not inv_problem.forward_defined
    assert not inv_problem.dataset_defined
    assert not inv_problem.initial_model_defined
    assert not inv_problem.model_shape_defined
    assert not inv_problem.bounds_defined
    assert not inv_problem.constraints_defined
    assert len(inv_problem.defined_components()) == 0
    inv_problem.summary()

def test_x_set():
    inv_problem = BaseProblem()
    inv_problem._data_x = np.array([1,2,3])
    assert not inv_problem.dataset_defined


############### TEST set methods Tier 3 ###############################################
def test_set_obj():
    inv_problem = BaseProblem()
    _x = np.array([1,2,3,4,5])
    _forward = lambda m, x_i: m[0] + m[1]*x_i + m[2]*x_i**2
    _forward_true = lambda x_i: 2 + x_i + x_i**2
    _y_true = np.vectorize(_forward_true)(_x)
    inv_problem.set_objective(lambda m: np.linalg.norm(_y_true-_forward(m,_x))/_x.shape[0]) # mse
    assert inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.data_misfit_defined
    assert not inv_problem.regularisation_defined
    assert not inv_problem.forward_defined
    assert not inv_problem.dataset_defined
    assert len(inv_problem.defined_components()) == 1
    assert inv_problem.objective(np.array([2,1,1])) == 0
    assert pytest.approx(inv_problem.objective(np.array([2,1,2]))) == 6.25779513
    # TODO - test suggest_solvers()


############### TEST set methods Tier 2 ###############################################
@pytest.fixture
def inv_problem_with_misfit():
    inv_problem = BaseProblem()
    _x = np.array([1,2,3,4,5])
    _forward = lambda m, x_i: m[0] + m[1]*x_i + m[2]*x_i**2
    _forward_true = lambda x_i: 2 + x_i + x_i**2
    _data_misfit = lambda m: np.linalg.norm(_y_true-_forward(m,_x))/_x.shape[0]
    _y_true = np.vectorize(_forward_true)(_x)
    inv_problem.set_data_misfit(_data_misfit)
    return inv_problem

def check_defined_misfit_reg(inv_problem):
    inv_problem.summary()
    assert inv_problem.data_misfit_defined
    assert inv_problem.regularisation_defined
    assert inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.dataset_defined
    assert not inv_problem.forward_defined
    assert len(inv_problem.defined_components()) == 3

def test_set_misfit_reg(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularisation(lambda m: m.T@m, 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2,1,1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert inv_problem_with_misfit.regularisation(true_model) == (4+1+1)*0.5
    assert inv_problem_with_misfit.objective(true_model) == (4+1+1)*0.5
    worse_model = np.array([2,1,2])
    assert pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2,1,2]))) == 6.25779513
    assert inv_problem_with_misfit.regularisation(worse_model) == (4+1+4)*0.5
    assert pytest.approx(inv_problem_with_misfit.objective(np.array([2,1,2]))) == 6.25779513+(4+1+4)*0.5

def test_set_misfit_reg_L0(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularisation("L0", 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2,1,1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert inv_problem_with_misfit.regularisation(true_model) == 3*0.5
    assert inv_problem_with_misfit.objective(true_model) == 3*0.5
    worse_model = np.array([2,1,2])
    assert pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2,1,2]))) == 6.25779513
    assert inv_problem_with_misfit.regularisation(worse_model) == 3*0.5
    assert pytest.approx(inv_problem_with_misfit.objective(np.array([2,1,2]))) == 6.25779513+3*0.5

def test_set_misfit_reg_L1(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularisation("L1", 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2,1,1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert inv_problem_with_misfit.regularisation(true_model) == 4*0.5
    assert inv_problem_with_misfit.objective(true_model) == 4*0.5
    worse_model = np.array([2,1,2])
    assert pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2,1,2]))) == 6.25779513
    assert inv_problem_with_misfit.regularisation(worse_model) == 5*0.5
    assert pytest.approx(inv_problem_with_misfit.objective(np.array([2,1,2]))) == 6.25779513+5*0.5

def test_set_misfit_reg_L2(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularisation("L2", 0.5)
    check_defined_misfit_reg(inv_problem_with_misfit)
    true_model = np.array([2,1,1])
    assert inv_problem_with_misfit.data_misfit(true_model) == 0
    assert inv_problem_with_misfit.regularisation(true_model) == np.sqrt(4+1+1)*0.5
    assert inv_problem_with_misfit.objective(true_model) == np.sqrt(4+1+1)*0.5
    worse_model = np.array([2,1,2])
    assert pytest.approx(inv_problem_with_misfit.data_misfit(np.array([2,1,2]))) == 6.25779513
    assert inv_problem_with_misfit.regularisation(worse_model) == np.sqrt(4+1+4)*0.5
    assert pytest.approx(inv_problem_with_misfit.objective(np.array([2,1,2]))) == 6.25779513+np.sqrt(4+1+4)*0.5

def test_invalid_reg_options():
    inv_problem = BaseProblem()
    with pytest.raises(NotImplementedError):
        inv_problem.set_regularisation("FOO")


############### TEST set methods Tier 1 ###############################################
@pytest.fixture
def inv_problem_with_data():
    inv_problem = BaseProblem()
    _x = np.array([1,2,3,4,5])
    _y = np.vectorize(lambda x_i: 2 + x_i + x_i**2)(_x)
    inv_problem.set_dataset(_x, _y)
    forward = lambda m: np.polynomial.Polynomial(m)(_x)
    return inv_problem, forward

def check_defined_data_fwd_misfit_reg(inv_problem):
    inv_problem.summary()
    assert inv_problem.dataset_defined
    assert inv_problem.forward_defined
    assert inv_problem.data_misfit_defined
    assert inv_problem.residual_defined
    assert inv_problem.regularisation_defined
    assert inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.jacobian_defined
    assert len(inv_problem.defined_components()) == 6

def check_values_data_fwd_misfit_reg(inv_problem):
    inv_problem.set_regularisation("L1", 0.5)
    true_model = np.array([2,1,1])
    assert inv_problem.data_misfit(true_model) == 0
    assert inv_problem.regularisation(true_model) == 4*0.5
    assert inv_problem.objective(true_model) == 4*0.5
    worse_model = np.array([2,1,2])
    assert pytest.approx(inv_problem.data_misfit(worse_model)) == 6.25779513
    assert inv_problem.regularisation(worse_model) == 5*0.5
    assert pytest.approx(inv_problem.objective(worse_model)) == 6.25779513+5*0.5

def test_set_data_fwd_misfit_inbuilt_reg_inbuilt(inv_problem_with_data):
    inv_problem, forward = inv_problem_with_data
    inv_problem.set_forward(forward)
    inv_problem.set_data_misfit("L2")
    inv_problem.set_regularisation("L1")
    check_defined_data_fwd_misfit_reg(inv_problem)
    check_values_data_fwd_misfit_reg(inv_problem)

def test_set_data_fwd_misfit_reg_all_inbuilt(inv_problem_with_data):
    inv_problem, _ = inv_problem_with_data
    inv_problem.set_forward("polynomial")
    inv_problem.set_data_misfit("L2")
    inv_problem.set_regularisation("L1")
    check_defined_data_fwd_misfit_reg(inv_problem)
    check_values_data_fwd_misfit_reg(inv_problem)

def test_invalid_fwd_options():
    inv_problem = BaseProblem()
    with pytest.raises(NotImplementedError):    # dataset not provided
        inv_problem.set_forward("polynomial")
    _x = np.array([1,2,3,4,5])
    _y = np.vectorize(lambda x_i: 2 + x_i + x_i**2)(_x)
    inv_problem.set_dataset(_x, _y)
    with pytest.raises(NotImplementedError):    # forward name not supported
        inv_problem.set_forward("FOO")

def test_invalid_misfit_options():
    inv_problem = BaseProblem()
    with pytest.raises(NotImplementedError):
        inv_problem.set_data_misfit("FOO")
    inv_problem.set_data_misfit("mse")
    with pytest.raises(NotImplementedError):
        inv_problem.data_misfit(np.array([1,2,3]))


############### TEST properties #######################################################
def test_check_defined():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda a: a+1)
    assert inv_problem.objective_defined
    assert str(inv_problem) == "BaseProblem"
    inv_problem.name = "AnotherProblem"
    assert str(inv_problem) == "AnotherProblem"
    inv_problem.set_initial_model(np.array([1,2,3]))
    assert inv_problem.initial_model_defined
    assert inv_problem.model_shape_defined
    assert inv_problem.model_shape == (3,)
    with pytest.raises(ValueError):
        inv_problem.set_model_shape((2,1))
    inv_problem.set_model_shape((3,1))
