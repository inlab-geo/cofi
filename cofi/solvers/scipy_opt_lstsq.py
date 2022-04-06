import inspect
from scipy.optimize import least_squares

from . import BaseSolver


# Official documentation for scipy.optimize.least_squares
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html


class ScipyOptLstSqSolver(BaseSolver):
    # get a list of arguments and defaults for scipy.minimize.least_squares
    # TODO arguments not supported by BaseProblem due to myself not sure how this can be
    #    handled for other backend solvers: `args`, `kwargs`, `x_scale`, `loss`, `f_scale`
    _scipy_ls_args = dict(inspect.signature(least_squares).parameters)
    _scipy_ls_args["jacobian"] = _scipy_ls_args.pop("jac")
    required_in_problem = {"residual", "initial_model"}
    optional_in_problem = {k:v.default for k,v in _scipy_ls_args.items() if k in {"jacobian", "bounds", "x_scale", "loss", "f_scale", "args", "kwargs"}}
    required_in_options = {}
    optional_in_options = {k:v.default for k,v in _scipy_ls_args.items() if k in {"method", "ftol", "xtol", "gtol", "diff_step", "tr_solver", "tr_options", "jac_sparsity", "max_nfev", "verbose"}}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._assign_args()

    def _assign_args(self):
        params = self.inv_options.get_params()
        inv_problem = self.inv_problem
        self._fun = inv_problem.residual
        self._x0 = inv_problem.initial_model
        self._args = inv_problem.args if hasattr(inv_problem, "args") else self.optional_in_problem["args"]
        self._kwargs = inv_problem.kwargs if hasattr(inv_problem, "kwargs") else self.optional_in_problem["kwargs"]
        self._jac = inv_problem.jacobian if inv_problem.jacobian_defined else self.optional_in_problem["jacobian"]
        self._bounds = inv_problem.bounds if inv_problem.bounds_defined else self.optional_in_problem["bounds"]
        self._method = params["method"] if "method" in params else self.optional_in_options["method"]
        self._ftol = params["ftol"] if "ftol" in params else self.optional_in_options["ftol"]
        self._xtol = params["xtol"] if "xtol" in params else self.optional_in_options["xtol"]
        self._gtol = params["gtol"] if "gtol" in params else self.optional_in_options["gtol"]
        self._x_scale = inv_problem.x_scale if hasattr(inv_problem, "x_scale") else self.optional_in_problem["x_scale"]
        self._loss = inv_problem.loss if hasattr(inv_problem, "loss") else self.optional_in_problem["loss"]
        self._f_scale = inv_problem.f_scale if hasattr(inv_problem, "f_scale") else self.optional_in_problem["f_scale"]
        self._diff_step = params["diff_step"] if "diff_step" in params else self.optional_in_options["diff_step"]
        self._tr_solver = params["tr_solver"] if "tr_solver" in params else self.optional_in_options["tr_solver"]
        self._tr_options = params["tr_options"] if "tr_optiosn" in params else self.optional_in_options["tr_options"]
        self._jac_sparsity = params["jac_sparsity"] if "jac_sparsity" in params else self.optional_in_options["jac_sparsity"]
        self._max_nfev = params["max_nfev"] if "max_nfev" in params else self.optional_in_options["max_nfev"]
        self._verbose = params["verbose"] if "verbose" in params else self.optional_in_options["verbose"]

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
