import numpy as np
import bayesbay as bb
from cofi.tools import BayesBay
from cofi import BaseProblem, InversionOptions


############### Problem setup #########################################################
_sample_size = 20
_x = np.random.choice(np.linspace(-3.5, 2.5), size=_sample_size)
_forward = lambda model: np.vander(_x, N=3, increasing=True) @ model
_m_true = np.array([-5, 2, 1])
_sigma = 0.01  # common noise standard deviation
_y = _forward(_m_true) + np.random.normal(0, 0.1, _sample_size)

m = bb.prior.UniformPrior("m", -7, 3, 1)
ps = bb.parameterization.ParameterSpace("ps", 3, parameters=[m])
p = bb.parameterization.Parameterization(ps)

def my_fwd(state: bb.State) -> np.ndarray:
    return _forward(state["ps"]["m"])

t = bb.Target("dt", _y, 1/_sigma**2)

n_chains = 1
ndim = 3
walkers_starting_states = []
for i in range(n_chains):
    walkers_starting_states.append(p.initialize())

log_like_ratio_func = bb.LogLikelihood([t], [my_fwd])
perturbation_funcs = p.perturbation_functions
n_iterations = 2000
burnin_iterations = 1500


############### Begin testing #########################################################
def test_run():
    inv_problem = BaseProblem()
    inv_options = InversionOptions()
    inv_options.set_tool("bayesbay")
    inv_options.set_params(
        log_like_ratio_func=log_like_ratio_func, 
        perturbation_funcs=perturbation_funcs, 
        walkers_starting_states=walkers_starting_states, 
        n_chains=n_chains, 
        n_iterations=n_iterations, 
        burnin_iterations=burnin_iterations, 
        verbose=False, 
    )
    runner = BayesBay(inv_problem, inv_options)
    res = runner()
    m_mean = np.mean(np.array(res["models"]["ps.m"]), axis=0)
    np.testing.assert_allclose(_m_true, m_mean, atol=4)
