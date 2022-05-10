from abc import abstractmethod, ABCMeta


class BaseSolver(metaclass=ABCMeta):
    r"""Base class for backend solver wrappers

    This is the point where we connect ``cofi`` to other inversion libraries or
    code. We expose this as a part of ``cofi``'s public interface, to facilitate minimal
    effort to link ``cofi`` to your own inversion code or external libraries that aren't
    connected by us yet.

    To create your own inversion solver, simply subclass :class:`BaseSolver` and
    define the following methods & fields.

    .. admonition:: Example definition of a custom solver
        :class: dropdown, attention

        .. code-block:: pycon
            :emphasize-lines: 1-12, 16, 23-24

            >>> from cofi.solvers import BaseSolver
            >>> class MyDummySolver(BaseSolver):
            ...   short_description = "My dummy solver that always return (1,2) as result"
            ...   documentation_links = ["https://cofi.readthedocs.io/en/latest/api/generated/cofi.solvers.BaseSolver.html"]
            ...   required_in_problem = ["objective", "gradient"]
            ...   optional_in_problem = {"initial_model": [0,0]}
            ...   required_in_options = []
            ...   optional_in_options = {"method": "dummy"}
            ...   def __init__(self, inv_problem, inv_options):
            ...     super().__init__(inv_problem, inv_options)
            ...   def __call__(self):
            ...     return {"model": np.array([1,2]), "success": True}
            ...
            >>> 
            >>> from cofi import InversionOptions
            >>> inv_options = InversionOptions()
            >>> inv_options.set_tool(MyDummySolver)
            >>> inv_options.summary()
            Summary for inversion options
            =============================
            Solving method: None set
            Use `suggest_solving_methods()` to check available solving methods.
            -----------------------------
            Backend tool: `<class '__main__.MyDummySolver'>` - My dummy solver that always return (1,2) as result
            References: ['https://cofi.readthedocs.io/en/latest/api/generated/cofi.solvers.BaseSolver.html']
            Use `suggest_tools()` to check available backend tools.
            -----------------------------
            Solver-specific parameters: None set
            Use `suggest_solver_params()` to check required/optional solver-specific parameters.

    Define the following minimally:

    .. autosummary::
        BaseSolver.__init__
        BaseSolver.__call__
        
    In addition (to let us to validate and display properly):

    .. autosummary::
        BaseSolver.required_in_problem
        BaseSolver.optional_in_problem
        BaseSolver.required_in_options
        BaseSolver.optional_in_options
        BaseSolver.short_description
        BaseSolver.documentation_links

    """
    documentation_links = list()
    short_description = str()
    components_used = set()
    required_in_problem = set()
    optional_in_problem = dict()
    required_in_options = set()
    optional_in_options = dict()
    inv_problem = None
    inv_options = None

    def __init__(self, inv_problem, inv_options):
        """initialisation routine for the solver instance

        You will need to implement this in the subclass, and it's recommended to have
        the following line included::

            super().__init__(inv_problem, inv_options)

        What it does is (a) to attach the :class:`BaseProblem` and :class:`InversionOptions`
        objects to ``self``, and (b) to validate them based on the information in 
        :func:`BaseSolver.required_in_problem`, :func:`BaseSolver.optional_in_problem`,
        :func:`BaseSolver.required_in_options`, and :func:`BaseSolver.optional_in_options`
        (which is why it's recommended to define the above four fields).

        Alternatively (if you want), you can also define your own validation routines,
        then you don't have to call the ``__init__`` method defined in this super class,
        and don't have to add things to the fields.

        Solver-specific configurations are to be declared as a part of 
        :class:`InversionOptions` object using :func:`InversionOptions.set_params`.
        You can then extract these in your own ``__init__`` method as needed. It's
        recommended to document solver-specific settings clearly in your docstrings.

        Parameters
        ----------
        inv_problem : BaseProblem
            an inversion problem setup
        inv_options : InversionOptions
            an object that defines how to run the inversion

        """
        self.inv_problem = inv_problem
        self.inv_options = inv_options
        self._validate_inv_problem()
        self._validate_inv_options()

    @abstractmethod
    def __call__(self) -> dict:
        """the method that calls your own inversion routines

        This is an abstract method, meaning that you have to implement this on your
        own in the subclass, otherwise the definition of the subclass will cause an
        error directly.

        Returns
        -------
        dict
            a Python dictionary that has at least ``model``/``models`` and ``success`` as
            keys

        """
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
