import numpy
import pytest

from cofi import BaseProblem
from cofi._exceptions import (
    InvalidOptionError, 
    DimensionMismatchError, 
    InvocationError, 
)


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
    inv_problem.set_hessian(numpy.array([1,2]))
    with pytest.raises(InvocationError):
        inv_problem.hessian_times_vector(0, numpy.array([1,2,3]))


def test_jacp_from_jac():
    inv_problem = BaseProblem()
    # test invalid
    inv_problem.set_jacobian(numpy.array([1,2]))
    with pytest.raises(InvocationError):
        inv_problem.jacobian_times_vector(0, numpy.array([1,2,3]))


def test_res_from_fwd_dt():
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda x: x**2)
    inv_problem.set_data(numpy.array([1,4,9]))
    with pytest.raises(InvocationError):
        inv_problem.residual(numpy.array([1,2]))


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
    from scipy import sparse
    A = sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    inv_problem.set_regularization(lambda m, A: A @ m.T @ m, args=[A])
    inv_problem.regularization(numpy.array([1,2,3]))

def test_invalid_func():
    inv_problem = BaseProblem()
    with pytest.raises(InvalidOptionError, match=".*invalid, please choose from the following.*"):
        inv_problem.set_forward(1)


############### TEST set from BaseProblem.__init__ ####################################
def test_init_set_fwd():
    _fwd = lambda x: x**2
    _fwd_w_args = lambda x, a: x**a
    p1 = BaseProblem(forward=_fwd)
    assert p1.forward_defined
    assert p1.forward(2) == 4
    p2 = BaseProblem(forwardd=_fwd)
    assert not p2.forward_defined
    p3 = BaseProblem(forward={"forward":_fwd_w_args, "args":[2]}, model_shape=(1,2))
    assert p3.forward_defined
    assert p3.forward(2) == 4
    assert p3.model_shape_defined
    assert p3.model_shape[1] == 2


############### TEST check_defined ####################################################
def test_check_defined():
    inv_problem = BaseProblem()
    # function defined by set_*
    inv_problem.set_objective(lambda a: a + 1)
    assert inv_problem.objective_defined
    # function defined by attaching
    inv_problem.forward = lambda x: x**2
    assert inv_problem.forward_defined
    # repr
    assert "BaseProblem" in str(inv_problem)
    inv_problem.name = "AnotherProblem"
    assert "AnotherProblem" in str(inv_problem)
    # initial model and model shape
    inv_problem.set_initial_model(numpy.array([1, 2, 3]))
    assert inv_problem.initial_model_defined
    assert inv_problem.model_shape_defined
    assert inv_problem.model_shape == (3,)
    with pytest.raises(DimensionMismatchError, match=r".*the model shape you've provided.*"):
        inv_problem.set_model_shape((2, 1))
    inv_problem.set_model_shape((3, 1))


############### TEST suggest_tools ##################################################
def test_suggest_tools(capsys):
    inv_problem = BaseProblem()
    # 0
    inv_problem.suggest_tools()
    console_output = capsys.readouterr().out
    assert "scipy.optimize.minimize" not in console_output
    assert "scipy.linalg.lstsq" not in console_output
    # 1
    inv_problem.set_initial_model(1)
    inv_problem.set_objective(lambda x: x)
    inv_problem.suggest_tools()
    console_output = capsys.readouterr().out
    assert "scipy.optimize.minimize" in console_output
    assert "scipy.optimize.least_squares" not in console_output
    assert "scipy.linalg.lstsq" not in console_output
    # 2
    inv_problem.set_jacobian(numpy.array([1]))
    inv_problem.set_data(2)
    inv_problem.suggest_tools()
    console_output = capsys.readouterr().out
    assert "scipy.linalg.lstsq" in console_output


def test_suggest_tools_return():
    inv_problem = BaseProblem()
    inv_problem.set_jacobian(numpy.array([1]))
    inv_problem.set_data(2)
    suggested = inv_problem.suggest_tools(print_to_console=False)
    assert "scipy.linalg.lstsq" in suggested["matrix solvers"]
