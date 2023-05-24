import pytest
import numpy
import scipy

from cofi.utils import LpNormRegularization
from cofi._exceptions import DimensionMismatchError


def _is_sparse_matrix(mat):
    sparse_types = [
        getattr(scipy.sparse, name)
        for name in scipy.sparse.__all__
        if name.endswith('_matrix')
    ]
    return any(isinstance(mat, cls) for cls in sparse_types)

def test_p_vals_valid():
    # p = 2
    reg = LpNormRegularization(p=2, model_shape=(2,))
    assert reg(numpy.array([1,2])) == 5
    # p = 1
    reg = LpNormRegularization(p=1, model_shape=(2,))
    assert reg(numpy.array([1,2])) == 3
    # p = 0
    reg = LpNormRegularization(p=0, model_shape=(2,))
    assert reg(numpy.array([1,2])) == 2
    # p = 0.5
    reg = LpNormRegularization(p=0.5, model_shape=(2,))
    assert reg(numpy.array([1,4])) == 3

def test_p_vals_invalid():
    # p = None
    with pytest.raises(ValueError, match=".*number expected.*got None of type.*"):
        LpNormRegularization(p=None, model_shape=(2,))
    # p = "1"
    with pytest.raises(ValueError, match=".*number expected.*got 1 of type <class 'str'>.*"):
        LpNormRegularization(p="1", model_shape=(2,))
    # p = "jump jump"
    with pytest.raises(ValueError, match=".*number expected.*got jump jump of type <class 'str'>.*"):
        LpNormRegularization(p="jump jump", model_shape=(2,))

def test_weighting_damping():
    reg = LpNormRegularization(model_shape=(2,))
    assert numpy.array_equal(reg.matrix, numpy.array([[1,0],[0,1]]))
    assert _is_sparse_matrix(reg.matrix)

def test_weighting_flattening():
    # ensure matrix is sparse
    pass

def test_weighting_smoothing():
    # ensure matrix is sparse
    pass

def test_weighting_none():
    # ensure matrix is sparse
    pass

def test_weighting_byo():
    # ensure matrix is sparse
    pass

def test_weighting_invalid():
    pass

def test_reference_model():
    pass

def test_model_size_must_be_given():
    # through model_size
    # through reference_model
    # error if neither is supplied
    pass

def test_reference_model_matching_shape():
    # ensure shape inferred from reference_model
    # ensure given model_shape matches reference_model
    pass


# integration test
@pytest.mark.parametrize(
    (
        "input_p,input_weighting_matrix,input_model_shape,input_reference_model,"
        "expected_matrix,test_model_valid,expected_reg,expected_gradient,"
        "expected_hessian,expected_model_size,test_model_invalid"
    ),
    [
        # (),
        # (),
    ]
)
def test_valid_cases(
    input_p, 
    input_weighting_matrix, 
    input_model_shape, 
    input_reference_model, 
    expected_matrix, 
    expected_model_size, 
    test_model_valid, 
    expected_reg,
    expected_gradient, 
    expected_hessian, 
    test_model_invalid, 
):
    reg = LpNormRegularization(
        input_p, 
        input_weighting_matrix, 
        input_model_shape, 
        input_reference_model, 
    )
    assert numpy.array_equal(reg.matrix, expected_matrix)
    assert _is_sparse_matrix(reg.matrix)
    assert reg.model_size == expected_model_size
    assert reg(test_model_valid) == expected_reg
    assert reg.reg(test_model_valid) == expected_reg
    assert numpy.array_equal(reg.gradient(test_model_valid), expected_gradient)
    assert numpy.array_equal(reg.hessian(test_model_valid), expected_hessian)
    with pytest.raises(DimensionMismatchError):
        reg(test_model_invalid)
