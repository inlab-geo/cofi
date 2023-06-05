import pytest
import numpy
import scipy
from scipy import sparse

from cofi.utils import LpNormRegularization
from cofi._exceptions import DimensionMismatchError


def _is_sparse_matrix(mat):
    sparse_types = [
        getattr(sparse, name)
        for name in sparse.__all__
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
    # p = 0.5
    reg = LpNormRegularization(p=0.5, model_shape=(2,))
    assert reg(numpy.array([1,4])) == 3

def test_p_vals_invalid():
    # p = -1
    with pytest.raises(ValueError, match=".*positive number expected.*got -1.*"):
        LpNormRegularization(p=-1, model_shape=(2,))
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
    # zero reference model == default
    ref_model = numpy.array([0,0])
    reg_with_m0 = LpNormRegularization(2, reference_model=ref_model)
    reg_without_m0 = LpNormRegularization(2, model_shape=(2,))
    assert reg_with_m0(ref_model) == reg_without_m0(ref_model)
    # nonzero reference model
    ref_model = numpy.array([0,1])
    reg = LpNormRegularization(reference_model=ref_model)
    assert reg(ref_model) == 0
    assert reg(numpy.array([0,0])) == 1

def test_model_shape_must_be_given():
    # through model_size
    reg = LpNormRegularization(model_shape=(2,))
    assert reg.model_shape == (2,)
    assert reg.model_size == 2
    # through reference_model
    reg = LpNormRegularization(reference_model=numpy.array([1,2]))
    assert reg.model_shape == (2,)
    assert reg.model_size == 2
    # error if neither is supplied
    with pytest.raises(ValueError, match=".*please provide the model shape.*"):
        reg = LpNormRegularization()
    # error if model_shape not in tuple
    with pytest.raises(TypeError, match=".*expected model shape in tuple.*"):
        reg = LpNormRegularization(model_shape=100)

def test_reference_model_matching_shape():
    # ensure given model_shape matches reference_model
    LpNormRegularization(model_shape=(2,), reference_model=numpy.array([1,2]))
    with pytest.raises(DimensionMismatchError, match=".*doesn't match and cannot be reshaped.*"):
        LpNormRegularization(model_shape=(3,), reference_model=numpy.array([1,2]))
    # ensure given model_shape matches reshaped reference_model
    LpNormRegularization(model_shape=(2,2), reference_model=numpy.array([1,2,3,4]))

def test_gradient_hessian_1():
    # None, p=1, without m0
    reg_none = LpNormRegularization(p=1, model_shape=(3,))
    grad_none = reg_none.gradient(numpy.array([1,2,3]))
    assert numpy.array_equal(grad_none, numpy.array([1,1,1]))
    hess_none = reg_none.hessian(numpy.array([1,2,3]))
    assert not numpy.any(hess_none)

def test_gradient_hessian_2():
    # damping, p=1, without m0
    reg_damping = LpNormRegularization(p=1, weighting_matrix="damping", model_shape=(3,))
    grad_damping = reg_damping.gradient(numpy.array([1,2,3]))
    assert numpy.array_equal(grad_damping, numpy.array([1,1,1]))
    hess_damping = reg_damping.hessian(numpy.array([1,2,3]))
    assert not numpy.any(hess_damping)

def test_gradient_hessian_3():
    # flattening, p=1, without m0
    reg_flattening = LpNormRegularization(p=1, weighting_matrix="flattening", model_shape=(3,))
    grad_flattening = reg_flattening.gradient(numpy.array([1,2,3]))
    assert numpy.array_equal(grad_flattening, numpy.array([-1.5,0,1.5]))
    hess_flattening = reg_flattening.hessian(numpy.array([1,2,3]))
    assert not numpy.any(hess_flattening)

def test_gradient_hessian_4():
    # smoothing, p=2, with m0
    reg_smoothing = LpNormRegularization(
        p=2, 
        weighting_matrix="smoothing", 
        reference_model=numpy.ones((4,4))
    )
    test_model_4x4 = numpy.array([[0., 0., 5., 0.],
                                [0., 0., 0., 2.],
                                [0., 3., 0., 3.],
                                [0., 0., 0., 0.]])
    grad_smoothing = reg_smoothing.gradient(test_model_4x4)
    expected_grad = numpy.array([ 140., -356.,  520., -140.,  -16., -208., -224.,  
                                 -56., -120., 636., -220.,  220.,    0.,  -96.,  -40.,  
                                 -40.])
    assert numpy.allclose(grad_smoothing, expected_grad)
    hess_smoothing = reg_smoothing.hessian(test_model_4x4)
    assert pytest.approx(hess_smoothing[0,0]) == 24
    assert pytest.approx(hess_smoothing[-1,-1]) == 24
    assert pytest.approx(hess_smoothing[0,1]) == -32
    assert pytest.approx(hess_smoothing[1,0]) == -32
    assert pytest.approx(hess_smoothing[1,5]) == -32

def test_gradient_hessian_5():
    # byo, p=2, with m0
    test_model_4x4 = numpy.array([[0., 0., 5., 0.],
                                [0., 0., 0., 2.],
                                [0., 3., 0., 3.],
                                [0., 0., 0., 0.]])
    reg_byo = LpNormRegularization(
        p=2,
        weighting_matrix=test_model_4x4,
        reference_model=numpy.array([1,2,1,2])
    )
    grad_byo = reg_byo.gradient(numpy.zeros((4,)))
    expected_grad = numpy.array([0, -72, -50, -88])
    assert numpy.allclose(grad_byo, expected_grad)
    hess_byo = reg_byo.hessian(numpy.zeros((4,)))
    assert pytest.approx(hess_byo[1,1]) == 18
    assert pytest.approx(hess_byo[1,3]) == 18
    assert pytest.approx(hess_byo[2,2]) == 50
    assert pytest.approx(hess_byo[3,1]) == 18
    assert pytest.approx(hess_byo[3,3]) == 26

def test_gradient_finite_diff():
    reg = LpNormRegularization(p=2, model_shape=(3,))
    model = numpy.array([0.1, 0.2, 0.3])
    delta_m = 0.00001
    gradient_actual = reg.gradient(model)
    # Finite difference approximation
    gradient_approx = numpy.zeros_like(model)
    for i in range(len(model)):
        model_perturbed = model.copy()
        model_perturbed[i] += delta_m
        gradient_approx[i] = (reg.reg(model_perturbed) - reg.reg(model)) / delta_m
    numpy.testing.assert_almost_equal(gradient_actual, gradient_approx, decimal=5)

def test_hessian_finite_diff():
    reg = LpNormRegularization(p=2, model_shape=(3,))
    model = numpy.array([0.1, 0.2, 0.3])
    delta_m = 0.05
    hessian_actual = reg.hessian(model)
    # Finite difference approximation
    hessian_approx = numpy.zeros((len(model), len(model)))
    for j in range(len(model)):
        model_perturbed_plus = model.copy()
        model_perturbed_plus[j] += delta_m
        model_perturbed_minus = model.copy()
        model_perturbed_minus[j] -= delta_m
        gradient_plus = reg.gradient(model_perturbed_plus)
        gradient_minus = reg.gradient(model_perturbed_minus)
        hessian_approx[:, j] = (gradient_plus - gradient_minus) / (2 * delta_m)
    numpy.testing.assert_almost_equal(hessian_actual, hessian_approx, decimal=5)
