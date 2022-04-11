import inspect
import numpy as np
from scipy.linalg import lstsq

from . import BaseSolver


class ScipyLstSqSolver(BaseSolver):
    documentation_link = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html"
    short_description = "SciPy's wrapper function over LAPACK's linear least-squares solver, " \
                        "using 'gelsd', 'gelsy' (default), or 'gelss' as backend driver"

    _scipy_lstsq_args = dict(inspect.signature(lstsq).parameters)
    required_in_problem: set = {"jacobian", "dataset"}
    optional_in_problem: dict  = {}
    required_in_options: set = {}
    optional_in_options: dict = {k:v.default for k,v in _scipy_lstsq_args.items() if k not in {'a','b'}}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._assign_args()

    def _assign_args(self):
        params = self.inv_options.get_params()
        inv_problem = self.inv_problem
        try:
            if inv_problem.initial_model_defined:
                jac_arg = inv_problem.initial_model
            elif inv_problem.model_shape_defined:
                jac_arg = np.ones(inv_problem.model_shape)
            else:
                jac_arg = np.ndarray([])
            self._a = inv_problem.jacobian(jac_arg)
        except:
            raise ValueError(
                "jacobian function isn't set properly for least squares solver, "
                "this should return a matrix unrelated to model vector"
            )
        self._b = inv_problem.data_y
        for opt in self.optional_in_options:
            setattr(self, f"_{opt}", params[opt] if opt in params else self.optional_in_options[opt])

    def __call__(self) -> dict:
        p, res, rnk, s = lstsq(
            a=self._a,
            b=self._b,
            cond=self._cond,
            overwrite_a=self._overwrite_a,
            overwrite_b=self._overwrite_b,
            check_finite=self._check_finite,
            lapack_driver=self._lapack_driver,
        )
        return {
            "success": True,
            "model": p,
            "sum of squared residuals": res,
            "effective rank": rnk,
            "singular values": s
        }
