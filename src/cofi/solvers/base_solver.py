from abc import abstractmethod, ABCMeta


class BaseSolver(metaclass=ABCMeta):
    documentation_links = list()
    short_description = str()
    components_used = set()
    required_in_problem = set()
    optional_in_problem = dict()
    required_in_options = set()
    optional_in_options = dict()

    def __init__(self, inv_problem, inv_options):
        self.inv_problem = inv_problem
        self.inv_options = inv_options
        self._validate_inv_problem()
        self._validate_inv_options()

    @abstractmethod
    def __call__(self) -> dict:
        raise NotImplementedError

    def _validate_inv_problem(self):
        # check whether enough information from inv_problem is provided
        defined = self.inv_problem.defined_components()
        required = self.required_in_problem
        if all({component in defined for component in required}):
            return True
        else:
            raise ValueError(
                f"you've chosen {self.__class__.__name__} to be your solving tool, but "
                f"not enough information is provided in the BaseProblem object - "
                f"required: {required}; provided: {defined}"
            )

    def _validate_inv_options(self):
        # check whether inv_options matches current solver (correctness of dispatch) from callee
        #      (don't use the dispatch table in runner.py, avoid circular import)
        # check whether required options are provided (algorithm-specific)
        defined = self.inv_options.get_params()
        required = self.required_in_options
        if all({option in defined for option in required}):
            return True
        else:
            raise ValueError(
                f"you've chosen {self.__class__.__name__} to be your solving tool, but "
                f"not enough information is provided in the InversionOptions object - "
                f"required: {required}; provided: {defined}"
            )

    def _assign_options(self):
        params = self.inv_options.get_params()
        for opt in self.required_in_options:
            setattr(self, f"_{opt}", params[opt])
        for opt in self.optional_in_options:
            setattr(
                self,
                f"_{opt}",
                params[opt] if opt in params else self.optional_in_options[opt],
            )

    def __repr__(self) -> str:
        return self.__class__.__name__