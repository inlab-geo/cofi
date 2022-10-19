import numpy as np

from . import BaseSolver


class CoFISimpleNewtonSolver(BaseSolver):
    documentation_links = [
        "https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization",
        "https://en.wikipedia.org/wiki/Backtracking_line_search",
    ]
    short_description = (
        "CoFI's own solver - simple Newton's approach (for testing mainly)"
    )

    required_in_problem = {"objective", "gradient", "hessian", "initial_model"}
    optional_in_problem = dict()
    required_in_options = {"max_iterations"}
    optional_in_options = {
        "step_length": 1,
        "verbose": True,
    }

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)

    def __call__(self) -> dict:
        m = self.inv_problem.initial_model
        n_obj_evaluations = 0
        n_grad_evaluations = 0
        n_hess_evaluations = 0
        for i in range(self._params["max_iterations"]):
            if self._params["verbose"]:
                print(
                    f"Iteration #{i}, objective function value:"
                    f" {self.inv_problem.objective(m)}"
                )
                n_obj_evaluations += 1
            grad = self.inv_problem.gradient(m)
            n_grad_evaluations += 1
            hess = np.atleast_2d((self.inv_problem.hessian(m)))
            n_hess_evaluations += 1
            step = -np.linalg.inv(hess).dot(grad)
            step = np.squeeze(np.asarray(step))
            m = m + self._params["step_length"] * step
        return {
            "model": m,
            "num_iterations": i,
            "objective_val": self.inv_problem.objective(m),
            "success": True,
            "n_obj_evaluations": n_obj_evaluations,
            "n_grad_evaluations": n_grad_evaluations,
            "n_hess_evaluations": n_hess_evaluations,
        }
