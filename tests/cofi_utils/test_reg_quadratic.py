import pytest
import numpy as np

from cofi.utils import QuadraticReg


def test_damping():
    reg = QuadraticReg(1, 3)
    reg_mat = reg.matrix
    assert reg_mat.shape[0] == 3
    assert reg_mat.shape[0] == reg_mat.shape[1]
    assert (reg_mat == np.eye(reg_mat.shape[0])).all()

def test_damping_invalid():
    with pytest.raises(ValueError): QuadraticReg(1, 10, "dampingnf")
    with pytest.raises(ValueError): QuadraticReg(-1, 3)
    with pytest.raises(ValueError): QuadraticReg(None, 3)
    with pytest.raises(ValueError): QuadraticReg("hello", 3)    
    with pytest.raises(ValueError): QuadraticReg(1, (1,2), "damping")

def test_flattening():
    reg = QuadraticReg(factor=1, model_size=(3,3), reg_type="flattening")
    # 1
    assert reg(np.zeros((3,3))) == 0
    # 2
    assert reg(np.ones((3,3))) == 0
    # 3
    reg_val1 = reg(np.array([[1,2,3],[1,2,3],[2,3,4]]))
    reg_val2 = reg(np.array([1,2,3,1,2,3,2,3,4]))
    assert reg_val1 == reg_val2
    # 4
    test_model = np.array([[1,1,1],[1,1,1],[1,1,2]])
    reg_val = reg(test_model)
    assert reg_val == 5.5
    reg_grad = reg.gradient(test_model)
    assert reg_grad.shape == (9,)
    reg_hess = reg.hessian(test_model)
    assert reg_hess.shape == (9,9)

def test_flattening_invalid():
    with pytest.raises(NotImplementedError): QuadraticReg(1, 1, reg_type="flattening")

def test_smoothing():
    pass

def test_smoothing_invalid():
    pass

def test_byo():
    # 1
    reg = QuadraticReg(1, 3, None)
    reg_mat = reg.matrix
    assert reg_mat.shape[0] == 3
    assert reg_mat.shape[0] == reg_mat.shape[1]
    assert (reg_mat == np.eye(reg_mat.shape[0])).all()
    # 2
    reg = QuadraticReg(1, 2, None, byo_matrix=np.array([[1,2],[3,4]]))

def test_byo_invalid():
    with pytest.raises(ValueError, match=r".*must be 2-dimensional*"):
        QuadraticReg(1, 3, None, byo_matrix=np.array([1,2,3]))
    with pytest.raises(ValueError, match=r".*must be in shape (_, M)*"):
        QuadraticReg(1, 3, None, byo_matrix=np.array([[1,2],[3,4]]))
