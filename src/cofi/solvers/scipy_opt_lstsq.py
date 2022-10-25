import inspect
from scipy.optimize import least_squares

from . import BaseSolver


# Official documentation for scipy.optimize.least_squares
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html


class ScipyOptLstSqSolver(BaseSolver):
    documentation_links = [
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html"
    ]
    short_description = (
        "SciPy's non-linear least squares solver with bounds on variables, "
        "algorithms include 'trf' (default), 'dogbox', and 'lm'"
    )

    # get a list of arguments and defaults for scipy.minimize.least_squares
    # TODO arguments not supported by BaseProblem due to myself not sure how this can be
    #    handled for other backend solvers: `args`, `kwargs`, `x_scale`, `loss`, `f_scale`
    _scipy_ls_args = dict(inspect.signature(least_squares).parameters)
    _scipy_ls_args["jacobian"] = _scipy_ls_args.pop("jac")
    required_in_problem = {"residual", "initial_model"}
    optional_in_problem = {
        k: v.default for k, v in _scipy_ls_args.items() if k in {"jacobian", "bounds"}
    }
    required_in_options = {}
    optional_in_options = {
        k: v.default
        for k, v in _scipy_ls_args.items()
        if k
        in {
            "method",
            "ftol",
            "xtol",
            "gtol",
            "x_scale",
            "f_scale",
            "loss",
            "diff_step",
            "tr_solver",
            "tr_options",
            "jac_sparsity",
            "max_nfev",
            "verbose",
        }
    }

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._assign_args()

    def _assign_args(self):
        inv_problem = self.inv_problem
        self.components_used = list(self.required_in_problem)
        # required_in_problem
        self._fun = inv_problem.residual
        self._x0 = inv_problem.initial_model
        # optional_in_problem
        defined_in_problem = self.inv_problem.defined_components()
        for component in self.optional_in_problem:
            if component in defined_in_problem:
                setattr(
                    self,
                    f"_{component}" if component != "jacobian" else "_jac",
                    getattr(self.inv_problem, component),
                )
                self.components_used.append(component)
            else:  # default
                setattr(
                    self,
                    f"_{component}" if component != "jacobian" else "_jac",
                    self.optional_in_problem[component],
                )

    def __call__(self) -> dict:
        opt_result = self._wrap_error_handler(
            least_squares,
            args=[],
            kwargs={
                "fun": self._fun,
                "x0": self._x0,
                "jac": self._jac,
                "bounds": self._bounds,
                "method": self._params["method"],
                "ftol": self._params["ftol"],
                "xtol": self._params["xtol"],
                "gtol": self._params["gtol"],
                "x_scale": self._params["x_scale"],
                "loss": self._params["loss"],
                "f_scale": self._params["f_scale"],
                "diff_step": self._params["diff_step"],
                "tr_solver": self._params["tr_solver"],
                "tr_options": self._params["tr_options"],
                "jac_sparsity": self._params["jac_sparsity"],
                "max_nfev": self._params["max_nfev"],
                "verbose": self._params["verbose"],
                "args": (),  # handled by cofi.BaseProblem
                "kwargs": {},  # handled by cofi.BaseProblem
            },
            when="when solving the least_squares optimization problem",
            context="calling `scipy.optimize.least_squares`",
        )
        result = dict(opt_result.items())
        result["model"] = result.pop("x")
        return result
