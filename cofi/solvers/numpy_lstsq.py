import numpy as np

from . import BaseSolver


REQUIRED_IN_PROBLEM = {"hessian", "jacobian", "dataset"}
OPTIONAL_IN_PROBLEM = {}
REQUIRED_IN_OPTIONS = {}
OPTIONAL_IN_OPTIONS = {"rcond": None}


class NumpyLstSqSolver(BaseSolver):
    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        
    def __call__(self) -> dict:
        x, residuals, rank, s = np.linalg.lstsq(self._G, self._y, self._rcond)
        return {"ok":True, "model":x, "residuals":residuals, "rank":rank, "singular_values":s}

    def _validate_inv_options(self):
        _hyperparams = self.inv_options.get_params()
        self._rcond = _hyperparams["rcond"] if "rcond" in _hyperparams else OPTIONAL_IN_OPTIONS["rcond"]

    def _validate_inv_problem(self):
        try:
            self._G = self.inv_problem.jacobian(np.ndarray([]))
        except:
            raise ValueError(
                "jacobian function isn't set properly for least squares solver, "
                "this should return a matrix unrelated to model vector"
            )
        self._y = self.inv_problem.data_y

    @staticmethod
    def _required_in_problem() -> set:
        return REQUIRED_IN_PROBLEM

    @staticmethod
    def _optional_in_problem() -> dict:
        return OPTIONAL_IN_PROBLEM

    @staticmethod
    def _required_in_options() -> set:
        return REQUIRED_IN_OPTIONS

    @staticmethod
    def _optional_in_options() -> dict:
        return OPTIONAL_IN_OPTIONS
