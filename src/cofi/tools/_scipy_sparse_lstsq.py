import numpy as np

from ._base_inference_tool import error_handler
from ._scipy_lstsq import ScipyLstSq


class ScipySparseLstSq(ScipyLstSq):
    r"""Wrapper for Scipy's sparse linear system solvers as laid out in:
    https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems.

    These solvers include direct and iterative solvers:
    - :func:`scipy.sparse.linalg.spsolve`
    - :func:`scipy.sparse.linalg.spsolve_triangular`
    - :func:`scipy.sparse.linalg.bicg`
    - :func:`scipy.sparse.linalg.bicgstab`
    - :func:`scipy.sparse.linalg.cg`
    - :func:`scipy.sparse.linalg.cgs`
    - :func:`scipy.sparse.linalg.gmres`
    - :func:`scipy.sparse.linalg.lgmres`
    - :func:`scipy.sparse.linalg.minres`
    - :func:`scipy.sparse.linalg.qmr`
    - :func:`scipy.sparse.linalg.gcrotmk`
    - :func:`scipy.sparse.linalg.tfqmr`
    - :func:`scipy.sparse.linalg.lsqr`
    - :func:`scipy.sparse.linalg.lsmr`

    The usage of this tool is similar to ``scipy.linalg.lstsq`` and users can include
    data noise and regularization in the inversion process. More specifically, there
    are four cases:

    1. basic case, where neither data noise nor regularization is taken into account,
       we solve :math:`Gm=d`
    2. with uncertainty, we solve :math:`m=(G^TC_d^{-1}G)^{-1}G^TC_d^{-1}d`
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
        "https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems"
    ]
    short_description = (
        "SciPy's sparse linear system solvers for solving linear problems"
    )

    @classmethod
    def required_in_problem(cls) -> set:
        return {"jacobian", "data"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {
            "data_covariance_inv": None,
            "data_covariance": None,
            "regularization_matrix": None,
        }

    @classmethod
    def required_in_options(cls) -> set:
        return set()

    @classmethod
    def optional_in_options(cls) -> dict:
        return {
            "algorithm": "spsolve",
            "atol": 0.0,
            "algorithm_params": dict(),
            "with_uncertainty_if_possible": True,
            "with_tikhonov_if_possible": True,
        }

    @classmethod
    def available_algorithms(cls) -> set:
        return {
            "spsolve",
            "spsolve_triangular",
            "bicg",
            "bicgstab",
            "cg",
            "cgs",
            "gmres",
            "lgmres",
            "minres",
            "qmr",
            "gcrotmk",
            "tfqmr",
            "lsqr",
            "lsmr",
        }

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        self._assign_args()  # defined in super class (_scipy_lstsq.ScipyLstSq)
        self._assign_args_without_reg_or_noise()
        self._assign_atol()
        self._validate_algorithm()

    def __call__(self) -> dict:
        raw_result = self._call_backend_tool()
        if self._params["algorithm"] in ["spsolve", "spsolve_triangular"]:
            res = {"success": True, "model": raw_result}
        elif self._params["algorithm"] == "lsqr":
            res = {
                "success": True,
                "model": raw_result[0],
                "istop": raw_result[1],
                "itn": raw_result[2],
                "r1norm": raw_result[3],
                "r2norm": raw_result[4],
                "anorm": raw_result[5],
                "acond": raw_result[6],
                "arnorm": raw_result[7],
                "xnorm": raw_result[8],
                "var": raw_result[9],
            }
        elif self._params["algorithm"] == "lsmr":
            res = {
                "success": True,
                "model": raw_result[0],
                "istop": raw_result[1],
                "itn": raw_result[2],
                "normr": raw_result[3],
                "normar": raw_result[4],
                "norma": raw_result[5],
                "conda": raw_result[6],
                "normx": raw_result[7],
            }
        else:
            res = {
                "success": raw_result[1] == 0,
                "model": raw_result[0],
                "info": raw_result[1],
            }
        return res

    def _assign_args_without_reg_or_noise(self):
        if (
            (not self._params["with_uncertainty"])
            and (not self._params["with_tikhonov"])
            and self._params["algorithm"] in ["spsolve", "spsolve_triangular"]
        ):
            self._a = self._G
            self._b = self._d

    def _assign_atol(self):
        if self._params["algorithm"] not in ["spsolve", "spsolve_triangular", "minres"]:
            self._params["algorithm_params"]["atol"] = self._params["atol"]

    def _validate_algorithm(self):
        if self._params["algorithm"] not in self.available_algorithms():
            raise ValueError(
                f"the algorithm you've chosen ({self._params['algorithm']}) "
                "is invalid. Please choose from the following: "
                f"{self.available_algorithms()}"
            )
        if not isinstance(self._params["algorithm_params"], dict):
            raise ValueError(
                "algorithm_params should be a dictionary, e.g. {'rtol': 1e-6}"
            )
        import scipy

        self.func = getattr(scipy.sparse.linalg, self._params["algorithm"])

    @error_handler(
        when="when solving the linear system equation",
        context="in the process of solving",
    )
    def _call_backend_tool(self):
        return self.func(A=self._a, b=self._b, **self._params["algorithm_params"])


# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> spsolve
# description: Direct method that solves the sparse linear system Ax=b, where b may be a vector or a matrix
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> spsolve_triangular
# description: Direct method that solves the sparse linear system Ax=b, assuming A is a triangular matrix.
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve_triangular.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> bicg
# description: Iterative method that solves the sparse linear system Ax=b using the BIConjugate Gradient method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicg.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> bicgstab
# description: Iterative method that solves the sparse linear system Ax=b using the BIConjugate Gradient STABilized method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> cg
# description: Iterative method that solves the sparse linear system Ax=b using the Conjugate Gradient method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> cgs
# description: Iterative method that solves the sparse linear system Ax=b using the Conjugate Gradient Squared method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cgs.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> gmres
# description: Iterative method that solves the sparse linear system Ax=b using the Generalized Minimal RESidual method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> lgmres
# description: Iterative method that solves the sparse linear system Ax=b using the LGMRES algorithm
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> minres
# description: Iterative method that solves the sparse linear system Ax=b using the MINimum RESidual method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> qmr
# description: Iterative method that solves the sparse linear system Ax=b using the Quasi-Minimal Residual method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.qmr.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> qcrotmk
# description: Iterative method that solves the sparse linear system Ax=b using the flexible GCROT(m,k) algorithm
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.qcrotmk.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> tfqmr
# description: Iterative method that solves the sparse linear system Ax=b using the Transpose-Free Quasi-Minimal Residual method
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.tfqmr.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> lsqr
# description: Iterative method that finds the least-squares solution to a large, sparse, linear system of equations
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html

# CoFI -> Parameter estimation -> Matrix based solvers -> Linear system solvers -> scipy.sparse.linalg -> lsmr
# description: Iterative method that solves sparse least-squares problems
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html
