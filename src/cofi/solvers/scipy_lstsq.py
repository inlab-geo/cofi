import inspect
import numpy as np
from scipy.linalg import lstsq

from . import BaseSolver


class ScipyLstSqSolver(BaseSolver):
    r"""Wrapper for generalised linear system solver :func:`scipy.linalg.lstsq`

    There are four cases:

    1. basic case, where neither data noise nor regularisation is taken into account,
       :math:`m=(G^TG)^{-1}G^Td`
    2. with uncertainty, :math:`m=(G^TC_d^{-1}G)^{-1}G^TC_d^{-1}d`
    3. with Tikhonov regularisation, :math:`m=(G^TG+\lambda L^TL)^{-1}G^Td`
    4. with both uncertainty and regularisation, 
       :math:`m=(G^TC_d^{-1}G+\lambda L^TL)^{-1}G^TC_d^{-1}d`

    where:

    - :math:`m` refers to inferred model (solution)
    - :math:`G` refers to the Jacobian of the forward operator
    - :math:`d` refers to the data vector
    - :math:`\lambda` refers to the regularisation factor that adjusts the ratio of
      data misfit and regularisation term
    - :math:`L` refers to the regularisation matrix, and is usually chosen to be 
      damping (zeroth order Tikhonov), roughening (first order Tikhonov), or smoothing
      (second order Tikhonov).
    """
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
    optional_in_options["with_tikhonov_if_possible"] = True

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)
        self._assign_args()

    def _assign_args(self):
        self._assign_options()
        inv_problem = self.inv_problem

        # get jacobian matrix (presumably jacobian is a constant)
        if inv_problem.initial_model_defined:
            dummy_model = inv_problem.initial_model
        elif inv_problem.model_shape_defined:
            dummy_model = np.ones(inv_problem.model_shape)
        else:
            dummy_model = np.array([])
        try:
            self._G = inv_problem.jacobian(dummy_model)
        except Exception as exception:
            raise ValueError(
                "jacobian function isn't set properly for least squares solver, "
                "this should return a matrix unrelated to model vector"
            ) from exception

        # get data vector
        self._d = inv_problem.data

        # check whether to take uncertainty into account
        self._with_uncertainty = self._with_uncertainty_if_possible and (
            inv_problem.data_covariance_defined
            or inv_problem.data_covariance_inv_defined
        )
        # get Cd_inv if needed
        if self._with_uncertainty:
            if not inv_problem.data_covariance_inv_defined:
                self._Cd_inv = np.linalg.inv(inv_problem.data_covariance)
            else:
                self._Cd_inv = inv_problem.data_covariance_inv
            _gt_cdinv = self._G.T @ self._Cd_inv
            self._a = _gt_cdinv @ self._G
            self._b = _gt_cdinv @ self._d
        else:
            self._a = self._G.T @ self._G
            self._b = self._G.T @ self._d

        # check whether to take regularisation into account
        self._with_tikhonov = self._with_tikhonov_if_possible and \
            inv_problem.regularisation_defined
        # get lamda and L matrix if needed
        if self._with_tikhonov:
            self._lamda = inv_problem.regularisation_factor
            if inv_problem.regularisation_matrix_defined:
                try:
                    self._L = inv_problem.regularisation_matrix(dummy_model)
                except Exception as exception:
                    raise ValueError(
                        "regularisation matrix function isn't set properly for least "
                        "squares solver, this should return a matrix unrelated to "
                        "model vector"
                    ) from exception
            else:
                self._L = np.identity(self._G.shape[1])
            self._a += self._lamda * self._L.T @ self._L

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
