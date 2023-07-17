import pytest
import numpy

import cofi


def problem_setup_objective(m, i):
    return m**(i+2)

def problem_setup_gradient(m, i):
    return (i+2)*m**(i+1)

def problem_setup_hessian(m, i):
    return (i+2) * (i+1) * m**i

def problem_setup(i):
    inv_problem = cofi.BaseProblem()
    inv_problem.set_initial_model(numpy.array([[10]]))
    inv_problem.set_objective(problem_setup_objective, args=(i,))
    inv_problem.set_gradient(problem_setup_gradient, args=(i,))
    inv_problem.set_hessian(problem_setup_hessian, args=(i,))
    return inv_problem

class CallBack:
    def __init__(self):
        self.objective_values = []

    def __call__(self, inv_result, i):
        self.objective_values.append(problem_setup_objective(inv_result.model, i))

@pytest.fixture
def problems_and_options():
    # problems
    problems = [problem_setup(i) for i in range(2)]
    # options
    inv_options = cofi.InversionOptions()
    inv_options.set_tool("cofi.simple_newton")
    inv_options.set_params(num_iterations=10)
    return problems, inv_options, CallBack()

def test_run_multiple_sequential(problems_and_options):
    problems, inv_options, callback = problems_and_options
    results = cofi.utils.run_multiple_inversions(
        problems, 
        inv_options, 
        callback, 
        False
    )
    for res in results:
        assert isinstance(res, cofi.InversionResult)
        assert pytest.approx(res.model.item(), abs=1e-2) == 0
    for obj_val in callback.objective_values:
        assert pytest.approx(obj_val.item(), abs=1e-6) == 0

def test_run_multiple_parallel(problems_and_options):
    problems, inv_options, callback = problems_and_options
    results = cofi.utils.run_multiple_inversions(
        problems, 
        inv_options, 
        callback, 
        True
    )
    print(callback.objective_values)
    assert False
    for res in results:
        assert isinstance(res, cofi.InversionResult)
        assert pytest.approx(res.model.item(), abs=1e-2) == 0
    for obj_val in callback.objective_values:
        assert pytest.approx(obj_val.item(), abs=1e-6) == 0
