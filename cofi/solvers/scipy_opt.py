import inspect
from scipy.optimize import minimize

from . import BaseSolver


class ScipyOptMinSolver(BaseSolver):
        # get a list of arguments and defaults for scipy.optimize.minimize
    _scipy_minimize_args = dict(inspect.signature(minimize).parameters)
    _scipy_minimize_args["gradient"] = _scipy_minimize_args.pop("jac")
    _scipy_minimize_args["hessian"] = _scipy_minimize_args.pop("hess")
    _scipy_minimize_args["hessian_times_vector"] = _scipy_minimize_args.pop("hessp")
    required_in_problem = {"objective", "initial_model"}           # `fun`, `x0`
    optional_in_problem = {k:v.default for k,v in _scipy_minimize_args.items() if k in {"gradient", "hessian", "hessian_times_vector", "bounds", "constraints", "args"}}
    required_in_options = {}
    optional_in_options = {k:v.default for k,v in _scipy_minimize_args.items() if k in {"method", "tol", "callback", "options"}}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        raise NotImplementedError

    def __call__(self) -> dict:
        # TODO
        raise NotImplementedError

    def _validate_inv_options(self):
        raise NotImplementedError

    def _validate_inv_problem(self):
        raise NotImplementedError
