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

def test_weighting_none():
    reg = LpNormRegularization(model_shape=(2,), weighting_matrix=None)
    assert reg.matrix[0,0] == reg.matrix[1,1] == 1
    assert reg.matrix[0,1] == reg.matrix[1,0] == 0
    assert _is_sparse_matrix(reg.matrix)

def test_weighting_damping():
    reg = LpNormRegularization(model_shape=(2,), weighting_matrix="damping")
    assert reg.matrix[0,0] == reg.matrix[1,1] == 1
    assert reg.matrix[0,1] == reg.matrix[1,0] == 0
    assert _is_sparse_matrix(reg.matrix)

def test_weighting_flattening():
    reg = LpNormRegularization(model_shape=(5,), weighting_matrix="flattening")
    assert reg.matrix[0,0] == -1.5
    assert reg.matrix[0,1] == 2
    assert reg.matrix[0,2] == reg.matrix[1,0] == reg.matrix[3,2] == -0.5
    assert reg.matrix[-1,-1] == 1.5
    assert _is_sparse_matrix(reg.matrix)

def test_weighting_smoothing():
    reg = LpNormRegularization(model_shape=(5,), weighting_matrix="smoothing")
    assert reg.matrix[0,0] == 2
    assert reg.matrix[0,1] == -5
    assert reg.matrix[1,0] == reg.matrix[-2,-1] == 1
    assert pytest.approx(reg.matrix[-1,-1]) == 2
    assert pytest.approx(reg.matrix[-1,-2]) == -5
    assert _is_sparse_matrix(reg.matrix)

def test_weighting_byo():
    my_matrix = numpy.array([[2,0],[0,2]])
    reg = LpNormRegularization(model_shape=(2,), weighting_matrix=my_matrix)
    assert reg.matrix[0,0] == reg.matrix[1,1] == 2
    assert reg.matrix[0,1] == reg.matrix[1,0] == 0
    assert _is_sparse_matrix(reg.matrix)

def test_weighting_invalid():
    # not matrix
    my_matrix = numpy.array([1,2])
    with pytest.raises(ValueError, match=".*must be 2-dimensional.*"):
        LpNormRegularization(model_shape=(2,), weighting_matrix=my_matrix)
    # shape mismatch
    my_matrix = numpy.array([[1,2],[0,1]])
    with pytest.raises(ValueError, match=".*in shape \(_, M\).*"):
        LpNormRegularization(model_shape=(3,), weighting_matrix=my_matrix)
    # wrong type
    my_matrix = "matrix"
    with pytest.raises(ValueError, match=".*specify the weighting matrix either.*"):
        LpNormRegularization(model_shape=(3,), weighting_matrix=my_matrix)

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
