from cofi.cofi_objective import BaseForward, LinearFittingFwd, PolynomialFittingFwd, Model

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------ initialise params ------------------------------
params = [3, 2, 5]
model = Model(m1=3, m2=2, m3=5)


# ------------------------------ test BaseForward ------------------------------
x_basefwd = np.linspace(0, 1, 100)
X_basefwd = np.array([x_basefwd ** o for o in range(3)]).T
base_fwd = BaseForward(forward=(lambda m, X: X @ m.values()))
y_basefwd = base_fwd.solve(model, X_basefwd)

plt.figure(1, figsize=(10, 3))
a = plt.subplot(1, 3, 1)
a.plot(x_basefwd, y_basefwd)
a.set_title("Using BaseForward")
# plt.show()


# ------------------------------ test LinearFittingFwd ------------------------------
linear_fwd = LinearFittingFwd()
x_linear = np.linspace(0, 1, 100)
X_linear = np.array([x_linear ** o for o in range(3)]).T
y_linear = linear_fwd.solve(model, X_linear)

b = plt.subplot(1, 3, 2)
b.plot(x_linear, y_linear)
b.set_title("Using LinearFittingFwd")
# plt.show()


# ------------------------------ test PolynomialFittingFwd ------------------------------
poly_fwd = PolynomialFittingFwd()
x_poly = np.linspace(0, 1, 100)
y_poly = poly_fwd.solve(model, x_poly)

c = plt.subplot(1, 3, 3)
c.plot(x_poly, y_poly)
c.set_title("Using PolynomialFittingFwd")
plt.show()
