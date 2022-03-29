from cofi.optimisers import ScipyOptimiserSolver, ScipyOptimiserLSSolver
from cofi.cofi_objective import ExpDecay

import numpy as np
import matplotlib.pyplot as plt


# ---------- define forward --------------------------------------------------
def predict(x, t):
    yhat = np.zeros_like(t)
    for i in range(int(np.shape(x)[0] / 2)):
        yhat += x[i * 2] * np.exp(-x[i * 2 + 1] * t)
    return yhat


# ---------- one exponential -------------------------------------------------
# generate data
x = np.array([1, 0.01])
t = np.linspace(0, 100, 20)
y = predict(x, t)
x0 = np.array([1.0, 0.012])
y0 = predict(x0, t)
# plt.plot(t,y)
# plt.plot(t,y0)

# define problem (objective) and solver
exp_decay_objective = ExpDecay(t, y, x0)
scipy_solver = ScipyOptimiserSolver(exp_decay_objective)

# solve with Nelder-Mead
print("-------------- 1 exp, Nelder-Mead ----------------------")
model_nm = scipy_solver.solve(method="Nelder-Mead", options={"disp": True})
print(model_nm.values())

# solve with Newton-CG
print("-------------- 1 exp, Newton-CG ------------------------")
exp_decay_objective.hessian = None
model_ncg = scipy_solver.solve(method="Newton-CG", options={"disp": True})
print(model_ncg.values())


# ---------- two exponentials -------------------------------------------------
# generate data
x_2 = np.array([1, 0.01, 2, 0.2])
t_2 = np.linspace(0, 100, 100)
y_2 = predict(x_2, t_2)
x0_2 = np.array([2, 0.001, 5, 0.1])
y0_2 = predict(x0_2, t_2)

# define problem (objective) and solver
exp_decay_objective_2 = ExpDecay(t_2, y_2, x0_2)
scipy_solver_2 = ScipyOptimiserSolver(exp_decay_objective_2)

# solve with Nelder-Mead
print("-------------- 2 exps, Newton-CG ------------------------")
model_2_nm = scipy_solver_2.solve(method="Nelder-Mead", options={"disp": True})
print(model_2_nm.values())

# solve with BFGS
print("-------------- 2 exps, BFGS -----------------------------")
model_2_bfgs = scipy_solver_2.solve(method="BFGS", options={"disp": True})
print(model_2_bfgs.values())


# ---------- three exponentials, Levenberg-Marquardt ---------------------------
# generate data
x_3 = np.array([1, 0.01, 2, 0.2, 3, 0.3])
t_3 = np.linspace(0, 100, 100)
y_3 = predict(x_3, t_3)
x0_3 = np.array([2, 0.001, 5, 0.1, 5, 1])
y0_3 = predict(x0_3, t_3)

# define problem (objective) and solver
exp_decay_objective_3 = ExpDecay(t_3, y_3, x0_3)
scipy_solver_3 = ScipyOptimiserLSSolver(exp_decay_objective_3)
scipy_solver_3.set_method("lm")

# solve with lm
print("-------------- 3 exps, Levenberg-Marquardt with Jacobian ------------")
model_3_lm = scipy_solver_3.solve()
print(model_3_lm.values())

print("------------ 3 exps, Levenberg-Marquardt without Jacobian -----------")
exp_decay_objective_3.jacobian = None
model_3_lm = scipy_solver_3.solve()
print(model_3_lm.values())
