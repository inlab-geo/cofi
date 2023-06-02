import numpy
import pytest

from cofi.utils import GaussianPrior
from cofi._exceptions import DimensionMismatchError


def test_byo_matrix():
    Cminv = numpy.ones((3,3))
    Cminv[1,1] = 2
    mean_model = numpy.ones((3,))
    reg = GaussianPrior(Cminv, mean_model)
    test_model = numpy.array([1,2,3])
    reg_out = reg(test_model)
    assert reg_out == 10
    grad_out = reg.gradient(test_model)
    assert numpy.array_equal(grad_out, numpy.array([6,8,6]))
    hess_out = reg.hessian(test_model)
    assert numpy.array_equal(hess_out, 2*Cminv)

def test_matrix_construction_1D():
    corr_lengths = (1.0,)
    sigma = 0.01
    gp = GaussianPrior((corr_lengths, sigma), numpy.zeros((5,)))
    model = numpy.ones(5)
    # check regularization term
    reg_term = gp.reg(model)
    assert numpy.isclose(reg_term, 284846862.90400386)
    # check gradient
    gradient = gp.gradient(model)
    assert numpy.allclose(
        gradient, 
        numpy.array([1.46211716e+08, 9.24234315e+07, 9.24234315e+07, 9.24234315e+07,
                    1.46211716e+08])
    )
    # check hessian
    hessian = gp.hessian(model)
    assert numpy.allclose(hessian, 2 * gp.gaussian_model_covariance_inv)

def test_matrix_construction_2D():
    corr_lengths = (1.0, 1.0)
    sigma = 0.01
    gp = GaussianPrior((corr_lengths, sigma), numpy.zeros((3,3)))
    model = numpy.ones(9)
    # check regularization term
    reg_term = gp.reg(model)
    assert numpy.isclose(reg_term, 335633686.7151643)
    # check gradient
    gradient = gp.gradient(model)
    assert numpy.allclose(
        gradient, 
        numpy.array([1.06763168e+08, 5.85483062e+07, 1.06763168e+08, 5.85483062e+07,
                    1.00214762e+07, 5.85483062e+07, 1.06763168e+08, 5.85483062e+07,
                    1.06763168e+08])
    )
    # check hessian
    hessian = gp.hessian(model)
    assert numpy.allclose(hessian, 2 * gp.gaussian_model_covariance_inv)

def test_mismatch_dimension():
    # mismatch in mean_model and model_covariance_inv
    Cminv = numpy.ones((3,3))
    mean_model = numpy.array([[1,2],[3,4]])
    with pytest.raises(ValueError, match=r".*expected for the shape of model_covariance_inv.*"):
        GaussianPrior(Cminv, mean_model)
    # mismatch in mean_model and corr_lengths
    with pytest.raises(ValueError, match=".*should have the same length.*"):
        GaussianPrior(((4,), 1), mean_model)
    # mismatch in mean_model and input model
    Cminv = numpy.ones((3,3))
    Cminv[1,1] = 2
    mean_model = numpy.ones((3,))
    reg = GaussianPrior(Cminv, mean_model)
    with pytest.raises(DimensionMismatchError):
        reg(numpy.array([1,2]))

def test_wrong_type():
    with pytest.raises(TypeError, match=".*but got 4 of type <class 'str'>.*"):
        GaussianPrior("4", numpy.ones((3,)))
