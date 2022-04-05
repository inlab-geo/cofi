import numpy as np

from . import BaseSolver


class NumpyLstSqSolver(BaseSolver):
    required_in_problem: set = {"hessian", "jacobian", "dataset"}
    optional_in_problem: dict  = {}
    required_in_options: set = {}
    optional_in_options: dict = {"rcond": None}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        
    def __call__(self) -> dict:
        x, residuals, rank, s = np.linalg.lstsq(self._G, self._y, self._rcond)
        return {"success":True, "model":x, "residuals":residuals, "rank":rank, "singular_values":s}

    def _validate_inv_problem(self):
        super()._validate_inv_problem()
        try:
            if self.inv_problem.initial_model_defined:
                jac_arg = self.inv_problem.initial_model
            elif self.inv_problem.model_shape_defined:
                jac_arg = np.ones(self.inv_problem.model_shape)
            else:
                jac_arg = np.ndarray([])
            self._G = self.inv_problem.jacobian(jac_arg)
        except:
            raise ValueError(
                "jacobian function isn't set properly for least squares solver, "
                "this should return a matrix unrelated to model vector"
            )
        self._y = self.inv_problem.data_y

    def _validate_inv_options(self):
        super()._validate_inv_options()
        _hyperparams = self.inv_options.get_params()
        self._rcond = _hyperparams["rcond"] if "rcond" in _hyperparams else self.optional_in_options["rcond"]
