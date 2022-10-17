import numpy as np

from cofi.solvers import PyTorchOptim
from cofi import BaseProblem, InversionOptions, Inversion


inv_problem = BaseProblem()
inv_problem.set_objective(lambda x: (x-3)**2)
inv_problem.set_initial_model(30)
inv_problem.set_gradient(lambda x: 2*x - 6)
inv_problem.set_hessian(lambda x: 2)
inv_options = InversionOptions()
inv_options.set_params(algorithm="SGD", lr=1, num_iterations=10)

def test_run():
    solver = PyTorchOptim(inv_problem, inv_options)
    res = solver()
    print(res)
    assert 0
