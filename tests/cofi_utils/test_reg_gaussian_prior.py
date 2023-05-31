import numpy
import pytest

from cofi.utils import GaussianPrior
from cofi._exceptions import DimensionMismatchError


def test_reg_gradient_hessian():
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

def test_mismatch_dimension():
    Cminv = numpy.ones((3,3))
    mean_model = numpy.array([[1,2],[3,4]])
    with pytest.raises(ValueError, match=r".*expected for the shape of model_covariance_inv.*"):
        GaussianPrior(Cminv, mean_model)

def test_invalid_input_model():
    Cminv = numpy.ones((3,3))
    Cminv[1,1] = 2
    mean_model = numpy.ones((3,))
    reg = GaussianPrior(Cminv, mean_model)
    with pytest.raises(DimensionMismatchError):
        reg(numpy.array([1,2]))
