from cofi import LinearObjective, PolynomialForward
import cofi.linear_reg as solvers
import cofi.optimisers as optim

import numpy as np
import matplotlib.pyplot as plt
import pytest


# ------------ #0 generate data -----------------------------------------
true_model = [3, 2, 5]
npts = 25
xpts = np.random.uniform(0, 1, npts)
forward = PolynomialForward(2)
ypts = forward.calc(true_model, xpts) + np.random.normal(0, 0.5, size=npts)

print(f"--> ground truth model: {np.array(true_model)}\n")

# uncomment plt.show() in the end to display the plot
plt.figure(figsize=(10, 8))
plt.plot(xpts, ypts, "x")
plt.plot(np.linspace(0, 1, 100), forward.calc(true_model, np.linspace(0, 1, 100)))
# plt.show()


# ------------ #1.1 define objective from pre-defined forward ------------
objective_1 = LinearObjective(xpts, ypts, forward.model_dimension(), forward=forward)


# ------------ #1.2 pure Python solver -----------------------------------
with pytest.warns(UserWarning):
    solver_1_pure = solvers.LRNormalEquation(objective_1)
    model_1_pure = solver_1_pure.solve()
    print(f"--> model predicted by pure Python solver: {model_1_pure.values()}\n")

ypts_predicted = forward.calc(model_1_pure, xpts)

# uncomment plt.show() in the end to display the plot
plt.figure(figsize=(10, 8))
plt.plot(xpts, ypts, "x", label="Data")
plt.plot(
    np.linspace(0, 1, 100),
    forward.calc(true_model, np.linspace(0, 1, 100)),
    label="Input",
)
plt.plot(
    np.linspace(0, 1, 100),
    forward.calc(model_1_pure, np.linspace(0, 1, 100)),
    label="Predicted",
)
plt.legend()
# plt.show()


# ------------ #1.3 scipy.optimize.minimize solver -----------------------------------
solver_1_scipy_minimize = optim.ScipyOptimiserSolver(objective_1)
solver_1_scipy_minimize.set_options({"tol": 1e-6})
model_1_scipy_minimize = solver_1_scipy_minimize.solve()
print(
    "--> model predicted by scipy.optimize.minimize:"
    f" {model_1_scipy_minimize.values()}\n"
)
solver_1_scipy_minimize.set_method("Newton-CG")
model_1_scipy_minimize_newtoncg = solver_1_scipy_minimize.solve()
print(
    "--> model predicted by scipy.optimize.minimize(Newton-CG):"
    f" {model_1_scipy_minimize_newtoncg.values()}\n"
)


# ------------ #1.4 scipy.optimize.least_squares solver -----------------------------------
solver_1_scipy_ls = optim.ScipyOptimiserLSSolver(objective_1)
model_1_scipy_ls = solver_1_scipy_ls.solve()
print(
    "--> model predicted by scipy.optimize.least_squares:"
    f" {model_1_scipy_ls.values()}\n"
)


# ------------ #1.5 TAO "nm" solver -----------------------------------
solver_1_tao_nm = optim.TAOSolver(objective_1)
model_1_tao_nm = solver_1_tao_nm.solve()
print(f"--> model predicted by TAO 'nm': {model_1_tao_nm.values()}\n")


# ------------ #1.6 TAO "brgn" solver -----------------------------------
solver_1_tao_brgn = optim.TAOSolver(objective_1)
model_1_tao_brgn = solver_1_tao_brgn.solve("brgn")
print(f"--> model predicted by TAO 'brgn': {model_1_tao_brgn.values()}\n")


# ------------ #2.1 define objective another way ---------------------------
nparams = 3
basis_function = lambda x: np.array([x**o for o in range(nparams)]).T
objective_2 = LinearObjective(xpts, ypts, nparams, basis_function)
print("--------- objective defined another way -------------------")


# ------------ #2.2 pure Python solver -----------------------------------
with pytest.warns(UserWarning):
    solver_2_pure = solvers.LRNormalEquation(objective_2)
    model_2_pure = solver_2_pure.solve()
    print(f"--> model predicted by pure Python solver: {model_2_pure.values()}\n")

    # uncomment plt.show() in the end to display the plot
    plt.figure(figsize=(10, 8))
    plt.plot(xpts, ypts, "x", label="Data")
    plt.plot(
        np.linspace(0, 1, 100),
        forward.calc(true_model, np.linspace(0, 1, 100)),
        label="Input",
    )
    plt.plot(
        np.linspace(0, 1, 100),
        forward.calc(model_1_pure, np.linspace(0, 1, 100)),
        label="Predicted 1",
        linewidth=3,
    )
    plt.plot(
        np.linspace(0, 1, 100),
        forward.calc(model_2_pure, np.linspace(0, 1, 100)),
        label="Predicted 2",
    )
    plt.legend()
    # plt.show()


# ------------ #2.2 C solver -----------------------------------
with pytest.warns(UserWarning):
    solver_2_c = solvers.LRNormalEquationC(objective_2)
    model_2_c = solver_2_c.solve()
    print(f"--> model predicted by C/Cython solver: {model_2_c.values()}\n")


# ------------ #2.3 C++ solver -----------------------------------
with pytest.warns(UserWarning):
    solver_2_cpp = solvers.LRNormalEquationCpp(objective_2)
    model_2_cpp = solver_2_cpp.solve()
    print(f"--> model predicted by C++/PyBind11 solver: {model_2_cpp.values()}\n")


# ------------ #2.4 Fortran 77 solver -----------------------------------
with pytest.warns(UserWarning):
    solver_2_f77 = solvers.LRNormalEquationF77(objective_2)
    model_2_f77 = solver_2_f77.solve()
    print(f"--> model predicted by Fortran77/f2py solver: {model_2_f77.values()}\n")


# ------------ #2.5 Fortran 90 solver -----------------------------------
with pytest.warns(UserWarning):
    solver_2_f90 = solvers.LRNormalEquationF90(objective_2)
    model_2_f90 = solver_2_f90.solve()
    print(f"--> model predicted by Fortran90/f2py solver: {model_2_f90.values()}\n")
