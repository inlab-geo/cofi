import numpy as np

from cofi import BaseProblem
from cofi.utils import BaseParameterisation, Identity


rng = np.random.default_rng(42)
N = 10


def test_usage_in_BaseProblem():
    # The Identity parameterisation multiplies by 1, i.e. it basically does nothing
    # This test checks that the output for BaseProblem.forward is the same with and
    # without the Identity parameterisation

    model = rng.random(N)
    inv_problem = BaseProblem()
    inv_problem.set_forward(lambda x: x**2)

    # Without parameterisation
    expected = inv_problem.forward(model)

    # With parameterisation
    parameterisation = Identity(N)
    inv_problem.set_parameterisation(parameterisation)
    actual = inv_problem.forward(model)

    assert np.allclose(expected, actual)


def test_alternate_usage_in_BaseProblem():
    # This test is the same as above but in the case where the user declares
    # the parameterisation before the forward operator

    model = rng.random(N)
    inv_problem = BaseProblem()
    parameterisation = Identity(N)
    inv_problem.set_parameterisation(parameterisation)
    inv_problem.set_forward(lambda x: x**2)

    inv_problem2 = BaseProblem()
    inv_problem2.set_forward(lambda x: x**2)

    assert np.allclose(inv_problem.forward(model), inv_problem2.forward(model))