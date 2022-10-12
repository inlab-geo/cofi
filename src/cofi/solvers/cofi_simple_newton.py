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
        # "enable_line_search": False,
        "verbose": True,
    }

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)
        self._assign_options()

    def __call__(self) -> dict:
        m = self.inv_problem.initial_model
        for i in range(self._max_iterations):
            if self._verbose:
                print(
                    f"Iteration #{i}, objective function value:"
                    f" {self.inv_problem.objective(m)}"
                )
            grad = self.inv_problem.gradient(m)
            hess = np.atleast_2d((self.inv_problem.hessian(m)))
            step = -np.linalg.inv(hess).dot(grad)
            m = m + self._step_length * step
        return {
            "model": m,
            "num_iterations": i,
            "objective_val": self.inv_problem.objective(m),
            "success": True,
        }
