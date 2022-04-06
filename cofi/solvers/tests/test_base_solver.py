import numpy as np
import pytest

from cofi import BaseProblem, InversionOptions
from cofi.solvers import BaseSolver


@pytest.fixture
def empty_setup():
    inv_problem = BaseProblem()
    inv_options = InversionOptions()
    return inv_problem, inv_options

@pytest.fixture
def subclass_soler_empty():
    class MyOwnSolverEmpty(BaseSolver):
        pass
    return MyOwnSolverEmpty

@pytest.fixture
def subclass_solver1():
    class MyOwnSolverRequiringProblemDef(BaseSolver):
        required_in_problem = {"gradient"}
        def __call__(self) -> dict:
            return super().__call__()
    return MyOwnSolverRequiringProblemDef

@pytest.fixture
def subclass_solver2():
    class MyOwnSolverRequiringOptionsDef(BaseSolver):
        required_in_options = {"tol"}
        def __call__(self) -> dict:
            return super().__call__()
    return MyOwnSolverRequiringOptionsDef


def test_abstract_methods(empty_setup, subclass_soler_empty):
    inv_prob, inv_opt = empty_setup
    with pytest.raises(TypeError):
        inv_solver = subclass_soler_empty(inv_prob, inv_opt)

def test_validation_problem(empty_setup, subclass_solver1):
    inv_prob, inv_opt = empty_setup
    # 1
    with pytest.raises(ValueError, match=".*not enough information is provided in the BaseProblem.*"):
        inv_solver = subclass_solver1(inv_prob, inv_opt)
    # 2
    inv_prob.set_gradient(lambda x: x)
    inv_solver = subclass_solver1(inv_prob, inv_opt)
    with pytest.raises(NotImplementedError):
        inv_solver()
    # 3
    assert str(inv_solver) == "MyOwnSolverRequiringProblemDef"

def test_validation_options(empty_setup, subclass_solver2):
    inv_prob, inv_opt = empty_setup
    # 1
    with pytest.raises(ValueError, match=".*not enough information is provided in the InversionOptions.*"):
        inv_solver = subclass_solver2(inv_prob, inv_opt)
    # 2
    inv_opt.set_params(tol=1000)
    inv_solver = subclass_solver2(inv_prob, inv_opt)
    with pytest.raises(NotImplementedError):
        inv_solver()
    # 3
    assert str(inv_solver) == "MyOwnSolverRequiringOptionsDef"
