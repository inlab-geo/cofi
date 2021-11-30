from cofi.cofi_solvers import TAOSolver
from cofi.cofi_objective import ExpDecay

import numpy as np
import matplotlib.pyplot as plt


# ---------- generate data --------------------------------------------------
def predict(x,t):
    yhat = np.zeros_like(t)
    for i in range(int(np.shape(x)[0]/2)):
        yhat += x[i*2]*np.exp(-x[i*2+1]*t)
    return yhat

x=np.array([1,0.1])
t=np.linspace(0,10)
y=predict(x,t)
x0=np.array([2,0.2])
y0=predict(x0,t)

plt.plot(t,y)
plt.plot(t,y0)
# plt.show()


# ---------- define problem -------------------------------------------------
exp_decay_objective = ExpDecay(t, y, x0)
tao_solver = TAOSolver(exp_decay_objective)


# ---------- start solving --------------------------------------------------
# some of the methods available in PETSc 
# https://petsc.org/main/docs/manualpages/Tao/TaoSetType.html
methods=["nm","lmvm","nls","ntr","cg","blmvm","tron"]

tao_solver.pre_solve()
for method in methods:
    tao_solver.solve(method)
