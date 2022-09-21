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

def test_flattening():
    pass

def test_flattening_invalid():
    pass

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
