from cofi.cofi_objective import PolynomialFittingFwd, Model
from cofi.cofi_solvers import LinearRegression

import numpy as np
import matplotlib.pyplot as plt


initial_model = Model(m1=1, m2=1, m3=1)

# --------------------------- generate data ---------------------------
true_model = Model(m1=3, m2=2, m3=5)
npts = 25
xpts = np.random.uniform(0, 1, npts)

poly_fwd = PolynomialFittingFwd()
ypts = poly_fwd.solve(true_model, xpts)
ypts += np.random.normal(0, 1, size=npts)

plt.plot(xpts, ypts, "x")
plt.plot(np.linspace(0, 1, 100), poly_fwd.solve(true_model, np.linspace(0, 1, 100)))
plt.xlabel("x")
plt.ylabel("y")
# plt.show()
