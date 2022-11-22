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
def subclass_solver_empty():
    class MyOwnSolverEmpty(BaseSolver):
        pass

    return MyOwnSolverEmpty


@pytest.fixture
def subclass_solver0():
    class MyOwnSolverImplementingNothing(BaseSolver):
        def __call__(self) -> dict:
            return super().__call__()
        @classmethod
        def required_in_problem(cls) -> set:
            return super().required_in_problem()
        @classmethod
        def optional_in_problem(cls) -> dict:
            return super().optional_in_problem()
        @classmethod
        def required_in_options(cls) -> set:
            return super().required_in_options()
        @classmethod
        def optional_in_options(cls) -> dict:
            return super().optional_in_options()

    return MyOwnSolverImplementingNothing


@pytest.fixture
def subclass_solver1():
    class MyOwnSolverRequiringProblemDef(BaseSolver):
        def __call__(self) -> dict: return super().__call__()
        @classmethod
        def required_in_problem(cls) -> set: return {"gradient"}
        @classmethod
        def optional_in_problem(cls) -> dict: return dict()
        @classmethod
        def required_in_options(cls) -> set: return set()
        @classmethod
        def optional_in_options(cls) -> dict: return dict()

    return MyOwnSolverRequiringProblemDef


@pytest.fixture
def subclass_solver2():
    class MyOwnSolverRequiringOptionsDef(BaseSolver):
        def __call__(self) -> dict:
            return super().__call__()
        @classmethod
        def required_in_problem(cls) -> set: return set()
        @classmethod
        def optional_in_problem(cls) -> dict: return dict()
        @classmethod
        def required_in_options(cls) -> set: return {"tol"}
        @classmethod
        def optional_in_options(cls) -> dict: return dict()

    return MyOwnSolverRequiringOptionsDef


def test_abstract_methods(empty_setup, subclass_solver_empty):
    inv_prob, inv_opt = empty_setup
    with pytest.raises(TypeError):
        inv_solver = subclass_solver_empty(inv_prob, inv_opt)

def test_to_make_coverage_happy(empty_setup, subclass_solver0):
    inv_prob, inv_opt = empty_setup
    inv_solver = subclass_solver0(inv_prob, inv_opt)
    with pytest.raises(NotImplementedError):
        inv_solver()
    assert not inv_solver.required_in_problem()
    assert not inv_solver.optional_in_problem()
    assert not inv_solver.required_in_options()
    assert not inv_solver.optional_in_options()


def test_validation_problem(empty_setup, subclass_solver1):
    inv_prob, inv_opt = empty_setup
    # 1
    with pytest.raises(
        ValueError, match=".*not enough information is provided in the BaseProblem.*"
    ):
        inv_solver = subclass_solver1(inv_prob, inv_opt)
    # 2
    inv_prob.set_gradient(lambda x: x)
    inv_solver = subclass_solver1(inv_prob, inv_opt)
    assert str(inv_solver) == "MyOwnSolverRequiringProblemDef"


def test_validation_options(empty_setup, subclass_solver2):
    inv_prob, inv_opt = empty_setup
    # 1
    with pytest.raises(
        ValueError,
        match=".*not enough information is provided in the InversionOptions.*",
    ):
        inv_solver = subclass_solver2(inv_prob, inv_opt)
    # 2
    inv_opt.set_params(tol=1000)
    inv_solver = subclass_solver2(inv_prob, inv_opt)
    with pytest.raises(NotImplementedError):
        inv_solver()
    # 3
    assert str(inv_solver) == "MyOwnSolverRequiringOptionsDef"
    # 4
    inv_solver._assign_options()


# ############################### TEST ALL SOLVERS ###############################
# 1. The four class methods are implemented and they return correct data types
# 2. __call__ method returns a dictionary with at least "success" and "model"
#    as keys
# ################################################################################




