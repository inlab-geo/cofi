import inspect
from pyexpat import model
from scipy.optimize import minimize

from . import BaseSolver


# methods available in scipy (can be string or callable):
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

# 'jac' is only for:
# CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact
# and trust-constr

# 'hess' is only for:
# Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

# 'hessp' is only for:
# Newton-CG, trust-ncg, trust-krylov, trust-constr

# 'bounds' is only for:
# Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr

# 'constraints' is only for:
# COBYLA, SLSQP and trust-constr

# other arguments include: tol, options, callback


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
        params = inv_options.get_params()
        self._func = inv_problem.objective
        self._x0 = inv_problem.initial_model
        self._method = params["method"] if "method" in params else self.optional_in_options["method"]
        self._jac = inv_problem.gradient if inv_problem.gradient_defined else self.optional_in_problem["gradient"]
        self._hess = inv_problem.hessian if inv_problem.hessian_defined else self.optional_in_problem["hessian"]
        self._hessp = inv_problem.hessian_times_vector if inv_problem.hessian_times_vector_defined else self.optional_in_problem["hessian_times_vector"]
        self._bounds = inv_problem.bounds if inv_problem.bounds_defined else self.optional_in_problem["bounds"]
        self._constraints = inv_problem.constraints if inv_problem.constraints_defined else self.optional_in_problem["constraints"]
        self._tol = params["tol"] if "tol" in params else self.optional_in_options["tol"]
        self._callback = params["callback"] if "callback" in params else self.optional_in_options["callback"]
        self._options = params["options"] if "options" in params else self.optional_in_options["options"]

    def __call__(self) -> dict:
        opt_result = minimize(self._func, self._x0, self._method, self._jac, self._hess, self._hessp, self._bounds, self._constraints, self._tol, self._callback, self._options)
        result = dict(opt_result.items())
        result["model"] = result.pop("x")
        return result
