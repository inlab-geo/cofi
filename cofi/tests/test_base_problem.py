import os

import pytest
import numpy as np

from cofi import BaseProblem, inv_problems


############### TEST data loader ######################################################
data_files_to_test = [
    "datasets/dummy_test1_comma.txt",
    "datasets/dummy_test2_tab.txt",
]

@pytest.fixture(params=data_files_to_test)
def data_path(request):
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.split(path_to_current_file)[0]
    data_path = os.path.join(current_directory, request.param)
    return data_path

def test_set_dataset_from_file(data_path):
    inv_problem = BaseProblem()
    inv_problem.set_dataset_from_file(data_path)


############### TEST empty problem ####################################################
def test_non_set():
    inv_problem = BaseProblem()
    with pytest.raises(NotImplementedError): inv_problem.objective(1)
    with pytest.raises(NotImplementedError): inv_problem.gradient(1)
    with pytest.raises(NotImplementedError): inv_problem.hessian(1)
    with pytest.raises(NotImplementedError): inv_problem.residual(1)
    with pytest.raises(NotImplementedError): inv_problem.jacobian(1)
    with pytest.raises(NotImplementedError): inv_problem.data_misfit(1)
    with pytest.raises(NotImplementedError): inv_problem.regularisation(1)
    with pytest.raises(NotImplementedError): inv_problem.forward(1)
    with pytest.raises(NameError): inv_problem.data_x
    with pytest.raises(NameError): inv_problem.data_y
    assert not inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.data_misfit_defined
    assert not inv_problem.regularisation_defined
    assert not inv_problem.dataset_defined
    assert len(inv_problem.defined_list()) == 0

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
    inv_problem.set_objective(lambda m: np.sum((_y_true-_forward(m,_x))**2)/_x.shape[0]) # mse
    assert inv_problem.objective_defined
    assert not inv_problem.gradient_defined
    assert not inv_problem.hessian_defined
    assert not inv_problem.residual_defined
    assert not inv_problem.jacobian_defined
    assert not inv_problem.data_misfit_defined
    assert not inv_problem.regularisation_defined
    assert not inv_problem.dataset_defined
    assert len(inv_problem.defined_list()) == 1
    assert inv_problem.objective(np.array([2,1,1])) == 0
    assert inv_problem.objective(np.array([2,1,2])) == 195.8
    # TODO - test suggest_solvers()


############### TEST set methods Tier 2 ###############################################
@pytest.fixture
def inv_problem_with_misfit():
    inv_problem = BaseProblem()
    _x = np.array([1,2,3,4,5])
    _forward = lambda m, x_i: m[0] + m[1]*x_i + m[2]*x_i**2
    _forward_true = lambda x_i: 2 + x_i + x_i**2
    _data_misfit = lambda m: np.sum((_y_true-_forward(m,_x))**2)/_x.shape[0]
    _y_true = np.vectorize(_forward_true)(_x)
    inv_problem.set_data_misfit(_data_misfit)
    return inv_problem

def test_set_misfit_reg(inv_problem_with_misfit):
    inv_problem_with_misfit.set_regularisation(lambda m: m.T@m, 0.5)
    

def test_set_misfit_reg_L0():
    inv_problem = BaseProblem()

def test_set_misfit_reg_L1():
    inv_problem = BaseProblem()

def test_set_misfit_reg_L2():
    inv_problem = BaseProblem()


############### TEST set methods Tier 1 ###############################################
def test_set_data_fwd_misfit_reg():
    inv_problem = BaseProblem()
    
def test_set_data_fwd_inbuilt_misfit_reg():
    inv_problem = BaseProblem()
    
def test_set_data_fwd_misfit_inbuilt_reg():
    inv_problem = BaseProblem()

def test_set_data_fwd_misfit_reg_inbuilt():
    inv_problem = BaseProblem()

def test_set_data_fwd_misfit_inbuilt_reg_inbuilt():
    inv_problem = BaseProblem()

def test_set_data_fwd_misfit_reg_all_inbuilt():
    inv_problem = BaseProblem()


############### TEST properties #######################################################
def test_check_defined():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda a: a+1)
    assert inv_problem.objective_defined



