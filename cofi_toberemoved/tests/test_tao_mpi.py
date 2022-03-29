### To run MPI, use the mpiexec provided by PETSc:
# - if PETSc is installed from source, then use the `mpiexec` in
#   $PETSC_DIR/$PETSC_ARCH/bin
# - if PETSc is installed from `pip`, then use the `mpiexec` in the
#   same path that contains your python (use `which python` to find out)


from cofi.optimisers import TAOSolver
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
# x_ = np.array([1, 0.2])
t_ = np.linspace(0, 10)
y_ = predict(x_, t_)
x0_ = np.array([2, 0.2, 3, 0.3, 4, 0.1])
# x0_ = np.array([2, 0.1])
y0_ = predict(x0_, t_)


# ---------- define problem & solve ------------------------------------------
exp_decay_objective_for_mpi = ExpDecay(t_, y_, x0_)
tao_solver_mpi = TAOSolver(exp_decay_objective_for_mpi, True)
tao_solver_mpi.set_options(
    "-tao_type brgn -tao_monitor -tao_brgn_regularization_type lm"
)
tao_solver_mpi.solve("brgn")

# ---------- other methods ---------------------------------------------------
# THIS IS NOT FULLY IMPLEMENTED YET, since it's not decided yet how to split computation
# tao_solver_mpi_2 = TAOSolver(exp_decay_objective_for_mpi, True)
# tao_solver_mpi_2.set_options("-tao_monitor")
# methods = [
#     "nm",
#     "lmvm",
#     "cg",
#     "blmvm"
# ]

# for method in methods:
#     model = tao_solver_mpi_2.solve(method)
