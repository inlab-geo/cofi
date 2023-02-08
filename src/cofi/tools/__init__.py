from ._base_inference_tool import BaseInferenceTool, error_handler

from ._scipy_opt_min import ScipyOptMin
from ._scipy_opt_lstsq import ScipyOptLstSq
from ._scipy_lstsq import ScipyLstSq
from ._emcee import Emcee
from ._cofi_simple_newton import CoFISimpleNewton
from ._pytorch_optim import PyTorchOptim


__all__ = [
    "BaseInferenceTool",  # public API, for advanced usage (own solver)
    "ScipyOptMin",
    "ScipyOptLstSq",
    "ScipyLstSq",
    "Emcee",
    "CoFISimpleNewton",
    "PyTorchOptim",
]

# inference tools table grouped by method:
# {inv_options.method -> {inv_options.tool -> BaseInferenceTool}}
inference_tools_table = {
    "optimization": {
        "scipy.optimize.minimize": ScipyOptMin,
        "scipy.optimize.least_squares": ScipyOptLstSq,
        "torch.optim": PyTorchOptim,
    },
    "matrix solvers": {
        "scipy.linalg.lstsq": ScipyLstSq,
        "cofi.simple_newton": CoFISimpleNewton,
    },
    "sampling": {"emcee": Emcee},
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
