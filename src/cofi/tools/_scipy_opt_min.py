import functools

from . import BaseInferenceTool, error_handler


# Official documentation for scipy.optimize.minimize
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

# 'jac' will be used when choosing the following methods:
# CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact
# and trust-constr

# 'hess' will be used when choosing the following methods:
# Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

# 'hessp' will be used when choosing the following methods:
# Newton-CG, trust-ncg, trust-krylov, trust-constr

# 'bounds' will be used when choosing the following methods:
# Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr

# 'constraints' will be used when choosing the following methods:
# COBYLA, SLSQP and trust-constr

# other arguments include: tol, options, callback


class ScipyOptMin(BaseInferenceTool):
    documentation_links = [
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html"
    ]
    short_description = (
        "SciPy's optimizers that minimizes a scalar function with respect to "
        "one or more variables, check SciPy's documentation page for a list of methods"
    )

    @classmethod
    def required_in_problem(cls) -> set:
        return _init_class_methods()[0]

    @classmethod
    def optional_in_problem(cls) -> dict:
        return _init_class_methods()[1]

    @classmethod
    def required_in_options(cls) -> set:
        return _init_class_methods()[2]

    @classmethod
    def optional_in_options(cls) -> dict:
        return _init_class_methods()[3]

    @classmethod
    @functools.lru_cache(maxsize=None)
    def available_algorithms(cls) -> set:
        import scipy

        return set(scipy.optimize._minimize.MINIMIZE_METHODS)

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        self._assign_args()

    def _assign_args(self):
        inv_problem = self.inv_problem
        # required_in_problem
        self._fun = inv_problem.objective
        self._x0 = inv_problem.initial_model
        # optional_in_problem
        _optional_in_problem_map = {
            "gradient": "jac",
            "hessian": "hess",
            "hessian_times_vector": "hessp",
            "bounds": "bounds",
            "constraints": "constraints",
        }
        defined_in_problem = self.inv_problem.defined_components()
        for component in _optional_in_problem_map:
            if component in defined_in_problem:
                self._params[_optional_in_problem_map[component]] = getattr(
                    self.inv_problem, component
                )
                self._components_used.append(component)
            else:  # default
                self._params[_optional_in_problem_map[component]] = (
                    self.optional_in_problem()[component]
                )

    def __call__(self) -> dict:
        opt_result = self._call_np_minimize()
        result = dict(opt_result.items())
        result["model"] = result.pop("x")
        return result

    @error_handler(
        when="when solving the optimization problem",
        context="calling `scipy.optimize.minimize`",
    )
    def _call_np_minimize(self):
        from scipy.optimize import minimize

        return minimize(
            fun=self._fun,
            x0=self._x0,
            args=(),  # handled by cofi.BaseProblem
            method=self._params["method"],
            jac=self._params["jac"],
            hess=self._params["hess"],
            hessp=self._params["hessp"],
            bounds=self._params["bounds"],
            constraints=self._params["constraints"],
            tol=self._params["tol"],
            callback=self._params["callback"],
            options=self._params["options"],
        )


@functools.lru_cache(maxsize=None)
def _init_class_methods():
    """get a list of arguments and defaults for scipy.minimize.minimize"""
    import inspect
    from scipy.optimize import minimize

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
        }
    }
    required_in_options = set()
    optional_in_options = {
        k: v.default
        for k, v in _scipy_minimize_args.items()
        if k in {"method", "tol", "callback", "options"}
    }
    return (
        required_in_problem,
        optional_in_problem,
        required_in_options,
        optional_in_options,
    )


# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> Nelder-Mead
# description: Uses the Simplex algorithm.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> Powell
# description: A modification of Powell’s method which is a conjugate direction method.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html#optimize-minimize-powell

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> CG
# description: Uses a nonlinear conjugate gradient algorithm by Polak and Ribiere, a variant of the Fletcher-Reeves method
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html#optimize-minimize-cg

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> BFGS
# description: Uses the quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS).
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> Newton-CG
# description: Uses a Newton-CG algorithm (also known as the truncated Newton method).
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html#optimize-minimize-newtoncg

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> L-BFGS-B
# description: Uses the L-BFGS-B algorithm for bound constrained minimization.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> TNC
# description: Uses a truncated Newton algorithm to minimize a function with variables subject to bounds.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html#optimize-minimize-tnc

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> COBYLA
# description: Uses the Constrained Optimization BY Linear Approximation (COBYLA) method.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html#optimize-minimize-cobyla

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> SLSQP
# description: Uses Sequential Least SQuares Programming to minimize a function of several variables with any combination of bounds, equality and inequality constraints.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> trust-constr
# description: A trust-region algorithm for constrained optimization.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html#optimize-minimize-trustconstr

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> dogleg
# description: Uses the dog-leg trust-region algorithm for unconstrained minimization.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-dogleg.html#optimize-minimize-dogleg

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> trust-ncg
# description: Uses the Newton conjugate gradient trust-region algorithm for unconstrained minimization.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustncg.html#optimize-minimize-trustncg

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> trust-exact
# description: A trust-region method for unconstrained minimization in which quadratic subproblems are solved almost exactly.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustexact.html#optimize-minimize-trustexact

# CoFI -> Parameter estimation -> Optimization -> Non linear -> scipy.optimize.minimize -> trust-krylov
# description: Uses the Newton GLTR trust-region algorithm for unconstrained minimization.
# documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustkrylov.html#optimize-minimize-trustkrylov
