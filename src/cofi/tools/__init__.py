from ._base_inference_tool import BaseInferenceTool, error_handler

from ._scipy_opt_min import ScipyOptMin
from ._scipy_opt_lstsq import ScipyOptLstSq
from ._scipy_lstsq import ScipyLstSq
from ._emcee import Emcee
from ._cofi_simple_newton import CoFISimpleNewton
from ._pytorch_optim import PyTorchOptim
from ._cofi_border_collie_optimization import CoFIBorderCollieOptimization
from ._bayes_bay import BayesBay


__all__ = [
    "BaseInferenceTool",  # public API, for advanced usage (own solver)
    "ScipyOptMin",
    "ScipyOptLstSq",
    "ScipyLstSq",
    "Emcee",
    "CoFISimpleNewton",
    "PyTorchOptim",
    "BayesBay",
    "CoFIBorderCollieOptimization",
]

# inference tools table grouped by method:
# {inv_options.method -> {inv_options.tool -> BaseInferenceTool}}
inference_tools_table = {
    "optimization": {
        "scipy.optimize.minimize": ScipyOptMin,
        "scipy.optimize.least_squares": ScipyOptLstSq,
        "torch.optim": PyTorchOptim,
        "cofi.border_collie_optimization": CoFIBorderCollieOptimization,
    },
    "matrix solvers": {
        "scipy.linalg.lstsq": ScipyLstSq,
        "cofi.simple_newton": CoFISimpleNewton,
    },
    "sampling": {
        "emcee": Emcee,
        "bayesbay": BayesBay,
    },
}

# inference tools suggest table grouped by method: {inv_options.method -> inv_options.tool}
# NOTE: the default backend solver is from this table, set the first one manually when necessary
# e.g. {'optimization': ['scipy.optimize.minimize'], 'matrix solvers': ['scipy.linalg.lstsq']}
tool_suggest_table = {k: list(val.keys()) for k, val in inference_tools_table.items()}

# tools dispatch table grouped by tool: {inv_options.tool -> BaseInferenceTool}
# e.g. {'scipy.optimize.minimize':
#           <class 'cofi.tools.scipy_opt_min.ScipyOptMinSolver'>,
#       'scipy.linalg.lstsq':
#           <class 'cofi.tools.scipy_lstsq.ScipyLstSqSolver'>}
tool_dispatch_table = {
    k: val for values in inference_tools_table.values() for k, val in values.items()
}

# all solving methods: {inv_options.method}
# e.g. {'optimization', 'matrix solvers'}
solving_methods = set(inference_tools_table.keys())

# alias for deprecated API
BaseSolver = BaseInferenceTool


# Extra information for InLab Explorer
# link_doc: CoFI -> https://cofi.readthedocs.io
# link_git: CoFI -> https://github.com/inlab-geo/cofi
# description: CoFI -> Common Framework for Inference
# description: Parameter estimation -> Parameter estimation is the process of determining the specific numerical values that define a parametrised (mathematical) model, often using methods like Maximum Likelihood Estimation or Least Squares to best fit the observed data.
# description: Ensemble methods -> Ensemble methods are a class of inference technique that result in multiple models, rather than a single model through parameter estimation, these might be driven by data fitting or Bayesian sampling.
# description: Optimization -> Optimization involves finding the best solution from a set of possible solutions, usually by minimizing or maximizing a certain function.
# description: Matrix based solvers -> Matrix-based solvers are computational algorithms that solve systems of equations, which often arise in linear or iteratively linear parameter estimation problems.
# description: Bayesian sampling -> Bayesian sampling is a technique for drawing samples (models) that follow a probability distribution of unknown parameters based on observed data and prior beliefs.
# description: Non linear -> Non-linear optimization focuses on finding the maximum, or minimum, of a function that is not necessarily quadratic over its parameters.
# description: Linear -> Optimization on linear problems involves finding the best solution from a set of possible solutions, where the function to be optimized is typically quadratic.
# description: Linear system solvers -> Linear system solvers are algorithms designed to find the values of unknowns in a set of linear equations.
# description: McMC samplers -> Markov chain Monte Carlo (McMC) samplers are algorithms for generating samples from complex probability distributions, often used in Bayesian inference.
# description: Direct search -> In the context of ensemble methods, direct search involves a search of the parameter space, e.g. to reduce a data misfit function, without requiring gradient information, These typically make use of exploration of the space and exploitation of previous sampling to guide search.
# description: Monte Carlo -> Monte Carlo methods use random sampling to obtain numerical results for problems that might be deterministic in principle.
# description: Deterministic -> Deterministic methods in direct search use specific rules, rather than randomness, to explore the parameter space for the best solution.
# description: Trans-D McMC -> Trans-Dimensional Markov chain Monte Carlo (Trans-D McMC) is a specialized form of McMC that allows for model selection by transitioning between different dimensional spaces.
