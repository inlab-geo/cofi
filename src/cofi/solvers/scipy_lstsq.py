import inspect
import numpy as np
from scipy.linalg import lstsq

from . import BaseSolver


class ScipyLstSqSolver(BaseSolver):
    documentation_links = [
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html",
        "https://www.netlib.org/lapack/lug/node27.html",
    ]
    short_description = (
        "SciPy's wrapper function over LAPACK's linear least-squares solver, "
        "using 'gelsd', 'gelsy' (default), or 'gelss' as backend driver"
    )

    _scipy_lstsq_args = dict(inspect.signature(lstsq).parameters)
    components_used: list = []
    required_in_problem: set = {"jacobian", "data"}
    optional_in_problem: dict = {"data_covariance_inv": None, "data_covariance": None}
    required_in_options: set = {}
    optional_in_options: dict = {
        k: v.default for k, v in _scipy_lstsq_args.items() if k not in {"a", "b"}
    }
    optional_in_options["with_uncertainty_if_possible"] = True

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)
        self._assign_args()

    def _assign_args(self):
        self._assign_options()
        
        inv_problem = self.inv_problem
        try:  # to get jacobian matrix (presumably jacobian is a constant)
            if inv_problem.initial_model_defined:
                jac_arg = inv_problem.initial_model
            elif inv_problem.model_shape_defined:
                jac_arg = np.ones(inv_problem.model_shape)
            else:
                jac_arg = np.ndarray([])
            self._a = inv_problem.jacobian(jac_arg)
        except Exception as exception:
            raise ValueError(
                "jacobian function isn't set properly for least squares solver, "
                "this should return a matrix unrelated to model vector"
            ) from exception
        self._b = inv_problem.data

        # taking uncertainty into account, if possible
        self._with_uncertainty = self._with_uncertainty_if_possible and \
            (inv_problem.data_covariance_defined or inv_problem.data_covariance_inv_defined)
        if self._with_uncertainty:
            if not inv_problem.data_covariance_inv_defined:
                data_cov_inv = np.linalg.inv(inv_problem.data_covariance)
            else:
                data_cov_inv = inv_problem.data_covariance_inv
            jac = self._a
            d_obs = self._b
            gt_cdinv = jac.T @ data_cov_inv
            self._a = gt_cdinv @ jac
            self._b = gt_cdinv @ d_obs


    def __call__(self) -> dict:
        res_p, residual, rank, singular_vals = lstsq(
            a=self._a,
            b=self._b,
            cond=self._cond,
            overwrite_a=self._overwrite_a,
            overwrite_b=self._overwrite_b,
            check_finite=self._check_finite,
            lapack_driver=self._lapack_driver,
        )
        res = {
                "success": True,
                "model": res_p,
                "sum of squared residuals": residual,
                "effective rank": rank,
                "singular values": singular_vals,
            }
        if self._with_uncertainty:
            res["model covariance"] = np.linalg.inv(self._a)
        return res
