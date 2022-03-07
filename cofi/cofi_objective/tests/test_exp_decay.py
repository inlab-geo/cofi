from cofi.cofi_objective import ExpDecay
from cofi import Model

import numpy as np
import pytest


# ---------- define utilities --------------------------------------------------
def predict(x, t):
    yhat = np.zeros_like(t)
    for i in range(int(np.shape(x)[0] / 2)):
        yhat += x[i * 2] * np.exp(-x[i * 2 + 1] * t)
    return yhat


def _test_exp(data):
    x, t, y, x0, y0 = data
    exp_decay_obj = ExpDecay(t, y, x0)
    assert np.array_equal(exp_decay_obj.residual(x), np.zeros(y.shape))
    assert np.array_equal(
        exp_decay_obj.residual(x), exp_decay_obj.residual_mpi(x, 0, t.shape[0])
    )
    assert exp_decay_obj.misfit(x) == exp_decay_obj.misfit_mpi(x, 0, t.shape[0])
    assert np.array_equal(
        exp_decay_obj.jacobian(x), exp_decay_obj.jacobian_mpi(x, 0, t.shape[0])
    )
    assert np.array_equal(
        exp_decay_obj.gradient(x), exp_decay_obj.gradient_mpi(x, 0, t.shape[0])
    )
    assert np.array_equal(
        exp_decay_obj.hessian(x), exp_decay_obj.hessian_mpi(x, 0, t.shape[0])
    )


# ---------- one exponential -------------------------------------------------
@pytest.fixture
def one_exp_data():
    # generate data
    x = np.array([1, 0.01])
    t = np.linspace(0, 100, 20)
    y = predict(x, t)
    x0 = np.array([1.0, 0.012])
    y0 = predict(x0, t)
    return x, t, y, x0, y0


def test_one_exp(one_exp_data):
    _test_exp(one_exp_data)


# ---------- two exponentials -------------------------------------------------
@pytest.fixture
def two_exp_data():
    # generate data
    x = np.array([1, 0.01, 2, 0.03])
    t = np.linspace(0, 100, 20)
    y = predict(x, t)
    x0 = np.array([1.0, 0.012, 3, 0.02])
    y0 = predict(x0, t)
    return x, t, y, x0, y0


def test_two_exp(two_exp_data):
    _test_exp(two_exp_data)


# ---------- test init & invalid input -----------------------------------------
def test_init(one_exp_data):
    x, t, y, x0, y0 = one_exp_data
    x0 = Model(a0=x0[0], x1=x0[1])
    x = Model(a0=x[0], a1=x[1])
    exp_decay_obj = ExpDecay(t, y, x0)
    assert exp_decay_obj.misfit(x) == 0


def test_invalid_init(one_exp_data):
    x, t, y, x0, y0 = one_exp_data
    x0 = Model(a0=x0[0])
    with pytest.raises(ValueError):
        exp_decay_obj = ExpDecay(t, y, x0)


def test_invalid_model(one_exp_data):
    x, t, y, x0, y0 = one_exp_data
    exp_decay_obj = ExpDecay(t, y, x0)
    with pytest.raises(ValueError):
        exp_decay_obj.misfit(x[0])
