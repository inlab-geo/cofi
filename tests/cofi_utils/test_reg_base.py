import pytest
import numpy as np

from cofi.utils import BaseRegularisation


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
