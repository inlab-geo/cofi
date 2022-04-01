from abc import abstractmethod, abstractproperty, ABCMeta
# from .. import BaseProblem, InversionOptions


class BaseSolver(metaclass=ABCMeta):
    # def __init__(self, inv_problem: BaseProblem, inv_options: InversionOptions) -> None:
    def __init__(self, inv_problem, inv_options):
        self.inv_problem = inv_problem
        self.inv_options = inv_options
        self._validate_inv_problem()
        self._validate_inv_options()

    @abstractmethod
    def __call__(self) -> dict:
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

    @staticmethod
    def _required_in_problem() -> set:
        raise NotImplementedError

    @staticmethod
    def _optional_in_problem() -> dict:
        raise NotImplementedError

    @staticmethod
    def _required_in_options() -> set:
        raise NotImplementedError

    @staticmethod
    def _optional_in_options() -> dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        # TODO - refine this (more info?)
        return self.__class__
