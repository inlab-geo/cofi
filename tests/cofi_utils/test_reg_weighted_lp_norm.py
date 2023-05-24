import numpy

from cofi.utils import LpNormRegularization


def test_p_vals_valid():
    # p = 2
    reg = LpNormRegularization(model_shape=(2,))
    assert reg(numpy.array([1,2])) == 5
    # p = 1
    # p = 0
    # p = 0.5
    # p = inf

def test_p_vals_invalid():
    pass

def test_weighting_damping():
    reg = LpNormRegularization(model_shape=(2,))
    assert numpy.array_equal(reg.matrix, numpy.array([[1,0],[0,1]]))
    # ensure matrix is sparse
    pass

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
