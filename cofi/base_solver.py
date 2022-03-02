from cofi.cofi_objective import Model


class BaseSolver:
    """Base class for all inverse solvers in CoFI.
    All inverse solvers must be sub-classes of this class and implements the 'solve()' method.

    Could potentially be split into categories based on what they need:
    - value of the objective function
    - value of the objective function and gradient
    - residual vector
    - residual vector and jacobian

    """

    def __init__(self):
        pass

    def solve(self, method=None) -> Model:
        raise NotImplementedError("inversion 'solve' method not implemented")

    def setMethod(self, method: str):
        self.method = method


class OptimiserMixin:
    """Mixin class for all optimisation based solvers in cofi."""

    def setOptions(self, options: dict):
        self.options = options
