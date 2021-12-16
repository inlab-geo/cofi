from cofi.cofi_objective.examples import LinearFitting
from cofi.cofi_objective import PolynomialFittingFwd
from cofi.cofi_solvers import SimpleLinearRegression

import numpy as np
import matplotlib.pyplot as plt


# ------------ #0 generate data -----------------------------------------
true_model = [3,2,5]
npts = 25
xpts = np.random.uniform(0, 1, npts)
forward = PolynomialFittingFwd(2)
ypts = forward.solve(true_model, xpts) + np.random.normal(0,0.5,size=npts)

print("--> ground truth model:", np.array(true_model))

plot = False
if plot:
    plt.figure(figsize=(10, 8))
    plt.plot(xpts, ypts, 'x')
    plt.plot(np.linspace(0,1,100), forward.solve(true_model, np.linspace(0,1,100)))
    plt.show()


# ------------ #1.1 define objective from pre-defined forward ------------
objective = LinearFitting(xpts, ypts, forward.model_dimension(), forward=forward)


# ------------ #1.2 pure Python solver -----------------------------------
solver = SimpleLinearRegression(objective)
model = solver.solve()
print("--> model predicted by pure Python solver:", model.values())

ypts_predicted = forward.solve(model, xpts)
# plot = True
if plot:
    plt.figure(figsize=(10, 8))
    plt.plot(xpts, ypts, 'x', label="Data")
    plt.plot(np.linspace(0,1,100), forward.solve(true_model, np.linspace(0,1,100)), label="Input")
    plt.plot(np.linspace(0,1,100), forward.solve(model, np.linspace(0,1,100)), label="Predicted")
    plt.legend()
    plt.show()


# ------------ #2.1 define objective another way ---------------------------
params_count = 3
basis_transform = lambda x: np.array([x ** o for o in range(params_count)]).T 
objective_2 = LinearFitting(xpts, ypts, params_count, basis_transform)

# ------------ #2.2 pure Python solver -----------------------------------
solver_2 = SimpleLinearRegression(objective_2)
model_2 = solver_2.solve()
print("--------- objective defined another way -------------------")
print("--> model predicted by pure Python solver:", model.values())
# plot = True
if plot:
    plt.figure(figsize=(10, 8))
    plt.plot(xpts, ypts, 'x', label="Data")
    plt.plot(np.linspace(0,1,100), forward.solve(true_model, np.linspace(0,1,100)), label="Input")
    plt.plot(np.linspace(0,1,100), forward.solve(model, np.linspace(0,1,100)), label="Predicted 1", linewidth=3)
    plt.plot(np.linspace(0,1,100), forward.solve(model_2, np.linspace(0,1,100)), label="Predicted 2")
    plt.legend()
    plt.show()

