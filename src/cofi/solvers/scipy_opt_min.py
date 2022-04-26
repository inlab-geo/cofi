import inspect
from scipy.optimize import minimize

from . import BaseSolver


# Official documentation for scipy.optimize.minimize
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
    documentation_links = [
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html"
    ]
    short_description = (
        "SciPy's optimisers that minimises a scalar function with respect to "
        "one or more variables, check SciPy's documentation page for a list of methods"
    )

    # get a list of arguments and defaults for scipy.optimize.minimize
    # TODO arguments not supported by BaseProblem due to myself not sure how this can be
    #    handled for other backend solvers: `args`
    _scipy_minimize_args = dict(inspect.signature(minimize).parameters)
    _scipy_minimize_args["gradient"] = _scipy_minimize_args.pop("jac")
    _scipy_minimize_args["hessian"] = _scipy_minimize_args.pop("hess")
    _scipy_minimize_args["hessian_times_vector"] = _scipy_minimize_args.pop("hessp")
    components_used: list = []
    required_in_problem = {"objective", "initial_model"}  # `fun`, `x0`
    optional_in_problem = {
        k: v.default
        for k, v in _scipy_minimize_args.items()
        if k
        in {
            "gradient",
            "hessian",
            "hessian_times_vector",
            "bounds",
            "constraints",
            "args",
        }
    }
    required_in_options = {}
    optional_in_options = {
        k: v.default
        for k, v in _scipy_minimize_args.items()
        if k in {"method", "tol", "callback", "options"}
    }

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)
        self._assign_args()

    def _assign_args(self):
        inv_problem = self.inv_problem
        # required_in_problem
        self._fun = inv_problem.objective
        self._x0 = inv_problem.initial_model
        # optional_in_problem
        _optional_in_problem_map = {
            "args": "args",
            "gradient": "jac",
            "hessian": "hess",
            "hessian_times_vector": "hessp",
            "bounds": "bounds",
            "constraints": "constraints",
        }
        defined_in_problem = self.inv_problem.defined_components()
        for component in _optional_in_problem_map:
            if component in defined_in_problem:
                setattr(
                    self,
                    f"_{_optional_in_problem_map[component]}",
                    getattr(self.inv_problem, component),
                )
                self.components_used.append(component)
            else:  # default
                setattr(
                    self,
                    f"_{_optional_in_problem_map[component]}",
                    self.optional_in_problem[component],
                )
        self._args = (
            inv_problem.args
            if hasattr(inv_problem, "args")
            else self.optional_in_problem["args"]
        )
        self._jac = (
            inv_problem.gradient
            if inv_problem.gradient_defined
            else self.optional_in_problem["gradient"]
        )
        self._hess = (
            inv_problem.hessian
            if inv_problem.hessian_defined
            else self.optional_in_problem["hessian"]
        )
        self._hessp = (
            inv_problem.hessian_times_vector
            if inv_problem.hessian_times_vector_defined
            else self.optional_in_problem["hessian_times_vector"]
        )
        self._bounds = (
            inv_problem.bounds
            if inv_problem.bounds_defined
            else self.optional_in_problem["bounds"]
        )
        self._constraints = (
            inv_problem.constraints
            if inv_problem.constraints_defined
            else self.optional_in_problem["constraints"]
        )
        # required_in_options, optional_in_options
        self._assign_options()

    def __call__(self) -> dict:
        opt_result = minimize(
            fun=self._fun,
            x0=self._x0,
            args=self._args,
            method=self._method,
            jac=self._jac,
            hess=self._hess,
            hessp=self._hessp,
            bounds=self._bounds,
            constraints=self._constraints,
            tol=self._tol,
            callback=self._callback,
            options=self._options,
        )
        result = dict(opt_result.items())
        result["model"] = result.pop("x")
        return result
