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
        k: v.default
        for k, v in _scipy_ls_args.items()
        if k in {"jacobian", "bounds", "x_scale", "loss", "f_scale", "args", "kwargs"}
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
        # required_in_options, optional_in_options
        self._assign_options()

    def __call__(self) -> dict:
        opt_result = least_squares(
            fun=self._fun,
            x0=self._x0,
            jac=self._jac,
            bounds=self._bounds,
            method=self._method,
            ftol=self._ftol,
            xtol=self._xtol,
            gtol=self._gtol,
            x_scale=self._x_scale,
            loss=self._loss,
            f_scale=self._f_scale,
            diff_step=self._diff_step,
            tr_solver=self._tr_solver,
            tr_options=self._tr_options,
            jac_sparsity=self._jac_sparsity,
            max_nfev=self._max_nfev,
            verbose=self._verbose,
            args=self._args,
            kwargs=self._kwargs,
        )
        result = dict(opt_result.items())
        result["model"] = result.pop("x")
        return result
