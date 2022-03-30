from abc import abstractmethod, abstractproperty
# from .. import BaseProblem, InversionOptions


class BaseSolver:
    # def __init__(self, inv_problem: BaseProblem, inv_options: InversionOptions) -> None:
    def __init__(self, inv_problem, inv_options):
        self.inv_problem = inv_problem
        self.inv_options = inv_options
        self._validate_inv_options()
        self._validate_inv_problem()
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _validate_inv_options(self):
        # check whether inv_options matches current solver (correctness of dispatch) from callee
        #      (don't use the dispatch table in runner.py, avoid circular import)
        # check whether required options are provided (algorithm-specific)
        raise NotImplementedError

    @abstractmethod
    def _validate_inv_problem(self):
        # check whether enough information from inv_problem is provided
        raise NotImplementedError

    @abstractproperty
    def _required_in_problem(self) -> list:
        raise NotImplementedError

    @abstractproperty
    def _optional_in_problem(self) -> list:
        raise NotImplementedError

    @abstractproperty
    def _required_options(self) -> list:
        raise NotImplementedError

    @abstractproperty
    def _optional_options(self) -> list:
        raise NotImplementedError

    def __repr__(self) -> str:
        # TODO - refine this (more info?)
        return self.__class__
