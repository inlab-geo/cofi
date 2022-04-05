import inspect
from scipy.optimize import minimize

from . import BaseSolver


class ScipyOptMinSolver(BaseSolver):
    required_in_problem: set = {"hessian", "jacobian", "dataset"}
    optional_in_problem: dict  = {}
    required_in_options: set = {}
    optional_in_options: dict = {"rcond": None}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        # get a list of argumnets and defaults for scipy.optimize.minimize
        scipy_minimize_args = dict(inspect.signature(minimize).parameters)
        self.required_in_problem = {"objective", "initial_model"}        # `fun`, `x0`
        optional_args = {k:v.default for k,v in scipy_minimize_args.items() if k not in ['fun', 'x0']}
        raise NotImplementedError

    def __call__(self) -> dict:
        # TODO
        raise NotImplementedError

    def _validate_inv_options(self):
        raise NotImplementedError

    def _validate_inv_problem(self):
        raise NotImplementedError
