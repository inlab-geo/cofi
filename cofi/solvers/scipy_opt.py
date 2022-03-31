# from .. import BaseProblem, InversionOptions
from . import BaseSolver


class ScipyOptMinSolver(BaseSolver):
    def __init__(self, inv_problem, inv_options):
        # TODO
        super().__init__(inv_problem, inv_options)
        raise NotImplementedError

    def __call__(self) -> dict:
        # TODO
        raise NotImplementedError

    def _validate_inv_options(self):
        raise NotImplementedError

    def _validate_inv_problem(self):
        raise NotImplementedError

    @property
    def _required_in_problem(self) -> list:
        raise NotImplementedError
        return []

    @property
    def _optional_in_problem(self) -> list:
        raise NotImplementedError
        return []

    @property
    def _required_options(self) -> list:
        raise NotImplementedError
        return []

    @property
    def _optional_options(self) -> list:
        raise NotImplementedError
        return []
