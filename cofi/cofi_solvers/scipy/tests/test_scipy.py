from cofi.cofi_solvers import ScipyOptimizerSolver
from cofi.cofi_objective import ExpDecay

import numpy as np
import matplotlib.pyplot as plt


# ---------- generate data --------------------------------------------------
def predict(x, t):
    yhat = np.zeros_like(t)
    for i in range(int(np.shape(x)[0] / 2)):
        yhat += x[i * 2] * np.exp(-x[i * 2 + 1] * t)
    return yhat


x = np.array([1, 0.01])
t = np.linspace(0, 100, 20)
y = predict(x, t)
x0 = np.array([1.0, 0.012])
y0 = predict(x0, t)

# plt.plot(t,y)
# plt.plot(t,y0)


# ---------- define problem -------------------------------------------------
exp_decay_objective = ExpDecay(t, y, x0)
scipy_solver = ScipyOptimizerSolver(exp_decay_objective)


# ---------- start solving --------------------------------------------------
scipy_solver.solve()
