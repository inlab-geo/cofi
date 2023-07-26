import numpy as np
import scipy

from . import BaseInferenceTool


class CoFISimpleNewton(BaseInferenceTool):
    documentation_links = [
        "https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization",
    ]
    short_description = (
        "CoFI's own solver - simple Newton's approach (for testing mainly)"
    )

    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective", "gradient", "hessian", "initial_model"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return dict()

    @classmethod
    def required_in_options(cls) -> set:
        return {"num_iterations"}

    @classmethod
    def optional_in_options(cls) -> dict:
        return {
            "step_length": 1,
            "verbose": True,
            "hessian_is_symmetric": False,
            "obj_tol": 1e-6,  # tolerance for change in objective function
            "param_tol": 1e-6,  # tolerance for change in model parameters
            "max_iterations": None,  # alias for num_iterations
        }

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._params["num_iterations"] = (
            self._params["max_iterations"] or self._params["num_iterations"]
        )

    def __call__(self) -> dict:
        m = self.inv_problem.initial_model
        m_prev = np.copy(m)
        obj_val = self.inv_problem.objective(m)
        obj_val_prev = obj_val
        self._n_obj_evaluations = 1
        self._n_grad_evaluations = 0
        self._n_hess_evaluations = 0

        for i in range(self._params["num_iterations"]):
            m = m + self._calculate_step(m)
            obj_val = self._verbose_objective_value(i, m)
            stop = self._stopping_criteria(obj_val, obj_val_prev, m, m_prev)
            if stop:
                break
            m_prev = m.copy()
            obj_val_prev = obj_val

        return {
            "model": m,
            "num_iterations": i,
            "objective_val": obj_val,
            "success": True,
            "n_obj_evaluations": self._n_obj_evaluations,
            "n_grad_evaluations": self._n_grad_evaluations,
            "n_hess_evaluations": self._n_hess_evaluations,
        }

    def _calculate_step(self, m):
        grad = self.inv_problem.gradient(m)
        self._n_grad_evaluations += 1
        hess = np.atleast_2d((self.inv_problem.hessian(m)))
        self._n_hess_evaluations += 1
        if self._params["hessian_is_symmetric"]:
            hess = scipy.sparse.csr_matrix(hess)
            step = scipy.sparse.linalg.minres(hess, -grad)[0]
        else:
            step = scipy.linalg.solve(hess, -grad)
        step = np.squeeze(np.asarray(step))
        step *= self._params["step_length"]
        return step

    def _stopping_criteria(self, obj_val, obj_val_prev, m, m_prev):
        # checking change in objective function value
        if np.abs(obj_val - obj_val_prev) < self._params["obj_tol"]:
            if self._params["verbose"]:
                print("Change in objective function below tolerance, stopping.")
            return True
        # checking biggest change in model parameters
        if np.max(np.abs(m - m_prev)) < self._params["param_tol"]:
            if self._params["verbose"]:
                print("Change in model parameters below tolerance, stopping.")
            return True
        return False

    def _verbose_objective_value(self, i, m):
        obj_val = self.inv_problem.objective(m)
        self._n_obj_evaluations += 1
        if self._params["verbose"]:
            print(f"Iteration #{i}, updated objective function value: {obj_val}")
        return obj_val


# CoFI -> Parameter estimation -> Optimization -> Non linear -> cofi.simple_newton -> Newton's method in optimization
# description: CoFI's own implementation of the Newton's method in optimization with stopping criteria.
# documentation: https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
