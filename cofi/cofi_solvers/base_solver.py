from cofi.cofi_objective.model_params import Model


class BaseSolver:
    """Base class for all inverse solvers in CoFI.
    All inverse solvers must be sub-classes of this class and implements the 'solve()' method.
    """

    def __init__(self):
        pass

    def solve(self) -> Model:
        raise NotImplementedError("inversion 'solve' method not implemented")
