import pytest
import numpy as np

from cofi.utils import BaseRegularization, QuadraticReg
from cofi._exceptions import DimensionMismatchError


def test_base_reg():
    class subclass_reg(BaseRegularization):
        def __init__(self):
            super().__init__()
        @property
        def model_size(self):
            return super().model_size
        @property
        def model_shape(self):
            return super().model_shape
        def reg(self, model):
            if isinstance(model, np.ndarray):
                return model
            else:
                return super().reg(model)
        def gradient(self, model):
            return super().gradient(model)
        def hessian(self, model):
            return super().hessian(model)
    test_reg = subclass_reg()
    assert test_reg(np.array([1]))[0] == 1
    with pytest.raises(NotImplementedError): test_reg.model_size
    with pytest.raises(NotImplementedError): test_reg.model_shape
    with pytest.raises(NotImplementedError): test_reg(1)
    with pytest.raises(NotImplementedError): test_reg.gradient(1)
    with pytest.raises(NotImplementedError): test_reg.hessian(1)

def test_add_regs():
    reg1 = QuadraticReg(model_shape=(9,))
    reg2 = QuadraticReg(model_shape=(3,3), weighting_matrix="flattening")
    reg3 = reg1 + reg2
    test_model = np.array([[1,2,3],[1,2,3],[2,3,4]])
    assert reg1(test_model) + reg2(test_model) == reg3(test_model)
    assert np.array_equal(reg1.gradient(test_model) + reg2.gradient(test_model), reg3.gradient(test_model))
    assert np.array_equal(reg1.hessian(test_model) + reg2.hessian(test_model), reg3.hessian(test_model))
    assert reg1.model_size == reg2.model_size
    assert reg1.model_size == reg3.model_size

def test_add_regs_invalid():
    reg1 = QuadraticReg(model_shape=(3,))
    reg2 = QuadraticReg(model_shape=(3,3), weighting_matrix="flattening")
    with pytest.raises(DimensionMismatchError):
        reg1 + reg2
    with pytest.raises(TypeError):
        reg1 + 1

def test_mul_reg_with_constant():
    reg = QuadraticReg(model_shape=(3,), weighting_matrix="damping")
    with pytest.raises(TypeError):
        "1" * reg
    new_reg = 10 * reg
    test_model = np.array([1,2,3])
    assert new_reg(test_model) == 10 * reg(test_model)
    assert np.array_equal(new_reg.gradient(test_model), 10 * reg.gradient(test_model))
    assert np.array_equal(new_reg.hessian(test_model), 10 * reg.hessian(test_model))
    assert reg.model_shape == new_reg.model_shape
