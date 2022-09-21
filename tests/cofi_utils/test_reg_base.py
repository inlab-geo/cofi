import pytest
import numpy as np

from cofi.utils import BaseRegularisation, RegularisationType


def test_reg_type():
    d = RegularisationType.damping
    f = RegularisationType.flattening
    r = RegularisationType.roughening
    s = RegularisationType.smoothing
    assert d.value == 0
    assert f.value == 1
    assert r.value == 1
    assert f == r
    assert s.value == 2

def test_reg_type_invalid():
    with pytest.raises(ValueError):
        RegularisationType(3)
    with pytest.raises(ValueError):
        RegularisationType(-1)

def test_base_reg():
    class subclass_reg(BaseRegularisation):
        def __init__(self):
            super().__init__()
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
    with pytest.raises(NotImplementedError): test_reg(1)
    with pytest.raises(NotImplementedError): test_reg.gradient(1)
    with pytest.raises(NotImplementedError): test_reg.hessian(1)
    
