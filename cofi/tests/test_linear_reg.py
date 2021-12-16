from cofi.cofi_objective.examples import LinearFitting
from cofi.cofi_objective import PolynomialFittingFwd
from cofi.cofi_solvers import SimpleLinearRegression

import numpy as np
import matplotlib.pyplot as plt


# -------------------- generate data --------------------------------
true_model = [3,2,5]
npts = 25
xpts = np.random.uniform(0, 1, npts)
forward = PolynomialFittingFwd(2)
ypts = forward.solve(true_model, xpts) + np.random.normal(0,0.5,size=npts)

print("--> ground truth model:")
print("-->", np.array(true_model))

plot = False
if plot:
    plt.figure(figsize=(10, 8))
    plt.plot(xpts, ypts, 'x')
    plt.plot(np.linspace(0,1,100), forward.solve(true_model, np.linspace(0,1,100)))
    plt.show()


# -------------------- define objective --------------------------------
objective = LinearFitting(xpts, ypts, forward)


# -------------------- pure Python solver --------------------------------
solver = SimpleLinearRegression(objective)
model = solver.solve()
print("--> model predicted by pure Python solver:")
print("-->", model.values())

ypts_predicted = forward.solve(model, xpts)
plot = True
if plot:
    plt.figure(figsize=(10, 8))
    plt.plot(xpts, ypts, 'x', label="Data")
    plt.plot(np.linspace(0,1,100), forward.solve(true_model, np.linspace(0,1,100)), label="Input")
    plt.plot(np.linspace(0,1,100), forward.solve(model, np.linspace(0,1,100)), label="Predicted")
    plt.legend()
    plt.show()

