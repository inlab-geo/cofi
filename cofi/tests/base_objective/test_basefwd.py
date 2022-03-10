from cofi import (
    BaseForward,
    LinearForward,
    PolynomialForward,
    Model,
)

import matplotlib.pyplot as plt
import numpy as np
import pytest


# ------------------------------ initialise params ------------------------------
params = [3, 2, 5]
model = Model(m1=3, m2=2, m3=5)


# ------------------------------ test BaseForward ------------------------------
def test_base_foward():
    x_basefwd = np.linspace(0, 1, 100)
    X_basefwd = np.array([x_basefwd**o for o in range(3)]).T
    base_fwd = BaseForward(forward=(lambda m, X: X @ m.values()), nparams=3)
    y_basefwd = base_fwd.calc(model, X_basefwd)
    y_basefwd_true = X_basefwd @ params
    print(y_basefwd)
    assert np.array_equal(y_basefwd, y_basefwd_true)

    pytest.raises(NotImplementedError, base_fwd.basis_function, x_basefwd)

    plt.figure(1, figsize=(10, 3))
    a = plt.subplot(1, 3, 1)
    a.plot(x_basefwd, y_basefwd)
    a.set_title("Using BaseForward")
    # plt.show()


# ------------------------------ test LinearForward ------------------------------
def test_linear_fwd():
    linear_fwd = LinearForward(3)
    x_linear = np.linspace(0, 1, 100)
    X_linear = np.array([x_linear**o for o in range(3)]).T
    y_linear = linear_fwd.calc(model, X_linear)

    pytest.raises(ValueError, linear_fwd.calc, np.array([1, 2, 3, 4]), X_linear)

    b = plt.subplot(1, 3, 2)
    b.plot(x_linear, y_linear)
    b.set_title("Using LinearForward")
    # plt.show()


# ------------------------------ test PolynomialForward ------------------------------
def test_poly_fwd():
    poly_fwd = PolynomialForward()
    x_poly = np.linspace(0, 1, 100)
    pytest.raises(ValueError, poly_fwd.basis_function, x_poly)
    y_poly = poly_fwd.calc(model, x_poly)

    pytest.raises(ValueError, poly_fwd.basis_function, np.array([x_poly, x_poly]))

    c = plt.subplot(1, 3, 3)
    c.plot(x_poly, y_poly)
    c.set_title("Using PolynomialForward")
    # plt.show()
