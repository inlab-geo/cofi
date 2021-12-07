
from cofi.cofi_solvers import TAOSolver
from cofi.cofi_objective import ExpDecay

import numpy as np


# ---------- generate data --------------------------------------------------
def predict(x, t):
    yhat = np.zeros_like(t)
    for i in range(int(np.shape(x)[0] / 2)):
        yhat += x[i * 2] * np.exp(-x[i * 2 + 1] * t)
    return yhat

# set data as np array, -> will be translated into petsc objects later
# all processes see the following
x_ = np.array([1, 0.1, 2, 0.2, 3, 0.3])
t_ = np.linspace(0, 10)
y_ = predict(x_, t_)
x0_ = np.array([2, 0.2, 3, 0.3, 4, 0.1])
y0_ = predict(x0_, t_)


# ---------- define problem & solve ------------------------------------------
exp_decay_objective_for_mpi = ExpDecay(t_, y_, x0_)
tao_solver_mpi = TAOSolver(exp_decay_objective_for_mpi, True)
tao_solver_mpi.set_options("-tao_monitor -tao_brgn_regularization_type lm")
tao_solver_mpi.solve('brgn')
