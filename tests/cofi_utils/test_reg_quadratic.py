import pytest
import numpy as np

from cofi.utils import QuadraticReg
from cofi.exceptions import DimensionMismatchError


def test_damping():
    # no ref model
    reg = QuadraticReg(1, 3)
    reg_mat = reg.matrix
    assert reg_mat.shape[0] == 3
    assert reg_mat.shape[0] == reg_mat.shape[1]
    assert (reg_mat == np.eye(reg_mat.shape[0])).all()
    reg_val = reg(np.array([1,2,3]))
    assert reg_val == 14
    grad = reg.gradient(np.array([1,2,3]))
    assert grad[0] == 2
    assert grad[1] == 4
    assert grad[2] == 6
    hess = reg.hessian(np.array([1,2,3]))
    assert hess[0,0] == 2
    assert hess[1,1] == 2
    assert hess[2,2] == 2
    # ref model
    reg = QuadraticReg(1, 3, ref_model=np.array([1,1,1]))
    reg_val = reg(np.array([1,2,3]))
    assert reg_val == 5
    grad = reg.gradient(np.array([1,2,3]))
    assert grad[0] == 0
    assert grad[1] == 2
    assert grad[2] == 4
    hess = reg.hessian(np.array([1,2,3]))
    assert hess[0,0] == 2
    assert hess[1,1] == 2
    assert hess[2,2] == 2

def test_damping_invalid():
    with pytest.raises(ValueError): QuadraticReg(1, 10, "dampingnf")
    with pytest.raises(ValueError): QuadraticReg(-1, 3)
    with pytest.raises(ValueError): QuadraticReg(None, 3)
    with pytest.raises(ValueError): QuadraticReg("hello", 3)    
    with pytest.raises(ValueError): QuadraticReg(1, (1,2), "damping")
    reg = QuadraticReg(1, 3)
    with pytest.raises(DimensionMismatchError): 
        reg(np.array([1,1]))

def test_flattening():
    reg = QuadraticReg(factor=1, model_size=(3,3), reg_type="flattening")
    reg = QuadraticReg(factor=1, model_size=(3,3), reg_type="roughening")
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
    # 5 - test matrix
    reg = QuadraticReg(1, 3, reg_type="flattening")
    assert np.allclose(reg.matrix, np.array([[-1.5, 2, -0.5], [-0.5, 0, 0.5], [0.5, -2, 1.5]]))

def test_flattening_invalid():
    # 1
    with pytest.raises(NotImplementedError, match=r".*only 2D derivative.*"): 
        QuadraticReg(factor=1, model_size=(1,2,3), reg_type="roughening")
    # 2
    with pytest.raises(ValueError, match=r".*at least \(>=3, >=3\).*'roughening'.*"):
        QuadraticReg(factor=1, model_size=(2,3), reg_type="roughening")
    # 3
    with pytest.raises(ValueError, match=r".*at least >=3 for.*flattening.*"): 
        QuadraticReg(factor=1, model_size=2, reg_type="flattening")

def test_smoothing():
    reg = QuadraticReg(factor=1, model_size=(4,4), reg_type="smoothing")
    # 1
    assert pytest.approx(reg(np.zeros((4,4)))) == 0
    # 2
    assert pytest.approx(reg(np.ones((4,4)))) == 0
    # 3
    reg_val1 = reg(np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[2,3,4,5]]))
    reg_val2 = reg(np.array([1,2,3,4,1,2,3,4,1,2,3,4,2,3,4,5]))
    assert pytest.approx(reg_val1) == reg_val2
    # 4
    test_model = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,2]])
    reg_val = reg(test_model)
    assert pytest.approx(reg_val) == 12
    reg_grad = reg.gradient(test_model)
    assert reg_grad.shape == (16,)
    reg_hess = reg.hessian(test_model)
    assert reg_hess.shape == (16,16)
    # 5 - test matrix
    reg = QuadraticReg(1, 4, reg_type="smoothing")
    assert np.allclose(reg.matrix, np.array([[2.0,-5,4,-1],[1,-2,1,0],[0,1,-2,1],[-1,4,-5,2]]))

def test_smoothing_invalid():
    # 1
    with pytest.raises(NotImplementedError, match=r".*only 2D derivative.*"): 
        QuadraticReg(1, (1,2,3), reg_type="smoothing")
    # 2
    with pytest.raises(ValueError, match=r".*at least \(>=4, >=4\).*'smoothing'.*"):
        QuadraticReg(factor=1, model_size=(2,3), reg_type="smoothing")
    # 3
    reg = QuadraticReg(factor=1, model_size=(4,4), reg_type="smoothing")
    with pytest.raises(ValueError): reg(np.zeros((3,3)))
    # 4
    with pytest.raises(ValueError, match=r".*at least >=4 for.*smoothing.*"): 
        QuadraticReg(factor=1, model_size=3, reg_type="smoothing")

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
    with pytest.raises(ValueError, match=r".*provide a number for 'model_size'.*"):
        QuadraticReg(1, (1,2), None, byo_matrix=np.array([[1,2],[3,4]]))
