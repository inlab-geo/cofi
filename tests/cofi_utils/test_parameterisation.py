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
