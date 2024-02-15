import functools
import numpy as np

from . import BaseInferenceTool, error_handler


class ScipyLstSq(BaseInferenceTool):
    r"""Wrapper for generalised linear system solver :func:`scipy.linalg.lstsq`

    There are four cases:

    1. basic case, where neither data noise nor regularization is taken into account,
       :math:`m=(G^TG)^{-1}G^Td`
    2. with uncertainty, :math:`m=(G^TC_d^{-1}G)^{-1}G^TC_d^{-1}d`
    3. with Tikhonov regularization, :math:`m=(G^TG+\lambda L^TL)^{-1}G^Td`
    4. with both uncertainty and regularization,
       :math:`m=(G^TC_d^{-1}G+\lambda L^TL)^{-1}G^TC_d^{-1}d`

    where:

    - :math:`m` refers to inferred model (solution)
    - :math:`G` refers to the Jacobian of the forward operator
    - :math:`d` refers to the data vector
    - :math:`\lambda` refers to the regularization factor that adjusts the ratio of
      data misfit and regularization term
    - :math:`L` refers to the regularization matrix, and is usually chosen to be
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
    def available_algorithms(cls) -> set:
        return {"gelsd", "gelsy", "gelss"}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        self._assign_args()

    def _assign_args(self):
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
        self._params["with_uncertainty"] = self._params[
            "with_uncertainty_if_possible"
        ] and (
            inv_problem.data_covariance_defined
            or inv_problem.data_covariance_inv_defined
        )
        # get Cd_inv if needed
        if self._params["with_uncertainty"]:
            if not inv_problem.data_covariance_inv_defined:
                self._Cd_inv = np.linalg.inv(inv_problem.data_covariance)
                self._components_used.append("data_covariance")
            else:
                self._Cd_inv = inv_problem.data_covariance_inv
                self._components_used.append("data_covariance_inv")
            # check diagonal (for potential shortcut in computation)
            diag_elem = np.diag(self._Cd_inv).copy()
            np.fill_diagonal(self._Cd_inv, 0)
            is_diagonal = (self._Cd_inv == 0).all()
            np.fill_diagonal(self._Cd_inv, diag_elem)
            if is_diagonal:
                _gt_cdinv = self._G.T * diag_elem
            else:
                _gt_cdinv = self._G.T @ self._Cd_inv
            self._a = _gt_cdinv @ self._G
            self._b = _gt_cdinv @ self._d
        else:
            self._a = self._G.T @ self._G
            self._b = self._G.T @ self._d

        # check whether to take regularization into account
        self._params["with_tikhonov"] = (
            self._params["with_tikhonov_if_possible"]
            and inv_problem.regularization_defined
        )
        # get lamda and L matrix if needed
        if self._params["with_tikhonov"]:
            if inv_problem.regularization_matrix_defined:
                try:
                    _L = inv_problem.regularization_matrix(dummy_model)
                except Exception as exception:
                    raise ValueError(
                        "regularization matrix function isn't set properly for least "
                        "squares solver, this should return a matrix unrelated to "
                        "model vector"
                    ) from exception
                self._LtL = _L.T @ _L
                self._components_used.append("regularization_matrix")
            else:
                self._LtL = np.identity(self._G.shape[1])
            self._a += self._LtL

    def __call__(self) -> dict:
        res_p, residual, rank, singular_vals = self._call_lstsq()
        res = {
            "success": True,
            "model": res_p,
            "sum_of_squared_residuals": residual,
            "effective_rank": rank,
            "singular_values": singular_vals,
        }
        if self._params["with_uncertainty"]:
            res["model_covariance"] = self._calculate_model_covariance()
        return res

    @error_handler(
        when="when solving the linear system equation",
        context="in the process of solving",
    )
    def _call_lstsq(self):
        from scipy.linalg import lstsq

        return lstsq(
            a=self._a,
            b=self._b,
            cond=self._params["cond"],
            overwrite_a=self._params["overwrite_a"],
            overwrite_b=self._params["overwrite_b"],
            check_finite=self._params["check_finite"],
            lapack_driver=self._params["lapack_driver"],
        )

    @error_handler(
        when="when calculating the model covariance estimation",
        context="after the solving process",
    )
    def _calculate_model_covariance(self):
        return np.linalg.inv(self._a)


@functools.lru_cache(maxsize=None)
def _init_class_methods():
    import inspect
    from scipy.linalg import lstsq

    _scipy_lstsq_args = dict(inspect.signature(lstsq).parameters)
    required_in_problem: set = {"jacobian", "data"}
    optional_in_problem: dict = {
        "data_covariance_inv": None,
        "data_covariance": None,
        "regularization_matrix": None,
    }
    required_in_options: set = set()
    optional_in_options: dict = {
        k: v.default for k, v in _scipy_lstsq_args.items() if k not in {"a", "b"}
    }
    optional_in_options["with_uncertainty_if_possible"] = True
    optional_in_options["with_tikhonov_if_possible"] = True
    return (
        required_in_problem,
        optional_in_problem,
        required_in_options,
        optional_in_options,
    )


# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.linalg.lstsq -> gelsd
# description: Solve linear least squares using divide-and-conquer SVD.

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.linalg.lstsq -> gelsy
# description: Solve linear least squares using complete orthogonal factorization.

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.linalg.lstsq -> gelss
# description: Solve linear least squares using SVD.
