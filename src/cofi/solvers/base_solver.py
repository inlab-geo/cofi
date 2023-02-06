from abc import abstractmethod, ABCMeta
import warnings

from ..exceptions import CofiError


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
            :emphasize-lines: 1-12, 17, 24-25

            >>> from cofi.solvers import BaseSolver
            >>> class MyDummySolver(BaseSolver):
            ...   short_description = "My dummy solver that always return (1,2) as result"
            ...   documentation_links = ["https://cofi.readthedocs.io/en/latest/api/generated/cofi.solvers.BaseSolver.html"]
            ...   @classmethod
            ...   def required_in_problem(cls): return ["objective", "gradient"]
            ...   def optional_in_problem(cls): return {"initial_model": [0,0]}
            ...   def required_in_options(cls): return []
            ...   def optional_in_options(cls): return {"method": "dummy"}
            ...   def __init__(self, inv_problem, inv_options):
            ...     super().__init__(inv_problem, inv_options)
            ...   def __call__(self):
            ...     return {"model": np.array([1,2]), "success": True}
            ...
            >>> from cofi import InversionOptions
            >>> inv_options = InversionOptions()
            >>> inv_options.set_tool(MyDummySolver)
            >>> inv_options.summary()
            =============================
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

    .. rubric:: Minimal implementation

    Define the following minimally. Input validation will be performed automatically
    when a new instance is created, based on what you've returned for the first four
    class methods below.

    .. autosummary::
        BaseSolver.required_in_problem
        BaseSolver.optional_in_problem
        BaseSolver.required_in_options
        BaseSolver.optional_in_options
        BaseSolver.__init__
        BaseSolver.__call__

    .. rubric:: Displaying

    In addition, the following class variables help us display properly.
    Failure to include them won't cause an error, but may result in some information
    mismatch in displaying methods like :func:`cofi.Inversion.summary`.

    .. autosummary::
        BaseSolver.short_description
        BaseSolver.documentation_links

    .. rubric:: Make it more complete

    All backend solvers in ``cofi`` also update the following field, and this will be
    displayed via :func:`cofi.Inversion.summary`. It's not required but good to keep track
    of this:

    .. autosummary::
        BaseSolver.components_used

    :ref:`back to top <top_BaseSolver>`

    """

    #: list: references about the backend solver. It helps ensure the audience
    #: understands what the backend solver does. The list can include the official
    #: documentation if the solver wraps an external library, or links to papers
    #: and other online resources explaining the approach
    #:
    #: For example, we use the following as the ``documentation_links`` for
    #: solver wrapping :func:`scipy.linalg.lstsq`, so that users can see the details
    #: of what's in the backend::
    #:
    #:     documentation_links = [
    #:         "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html",
    #:         "https://www.netlib.org/lapack/lug/node27.html",
    #:     ]
    documentation_links = list()
    #: str: a short introduction about the solver. This is for display purpose only
    short_description = str()

    def __init__(self, inv_problem, inv_options):
        """initialisation routine for the solver instance

        You will need to implement this in the subclass, and it's recommended to have
        the following line included::

            super().__init__(inv_problem, inv_options)


        What it does are:

        - Attaching the :class:`BaseProblem` and :class:`InversionOptions` objects to
          ``self``;
        - Validating both input objects based on the information in
          :meth:`required_in_problem`, :meth:`optional_in_problem`,
          :meth:`required_in_options`, and :meth:`optional_in_options`;
        - Assigning inversion tool parameters to ``self._params`` based on what are
          returned by class methods :meth:`required_in_options`,
          :meth:`optional_in_options`, and what are set by users in the
          :class:`InversionOptions` object;
        - Initializing the ``self._components_used`` dictionary based on what are
          returned by class methods :meth:`required_in_problem`,
          :meth:`optional_in_problem`, and what are defined by users in the
          :class:`BaseProblem` object. You can overwrite this if your solver is more
          specific in deciding what components get used.

        Alternatively (if you want), you can also define your own validation routines,
        then you don't have to call the ``__init__`` method defined in this super
        class, and don't have to add things to the fields.

        Parameters
        ----------
        inv_problem : BaseProblem
            an inversion problem setup
        inv_options : InversionOptions
            an object that defines how to run the inversion

        """
        self._inv_problem = inv_problem
        self._inv_options = inv_options
        self._validate_inv_problem()
        self._validate_inv_options()
        self._assign_options()  # assigns options to self._params
        self._update_components_used()  # update components to self._components_used

    @classmethod
    @abstractmethod
    def required_in_problem(cls) -> set:
        r"""a set of components required in :class:`BaseProblem` instance

        This is a standard part for a subclass of :class:`BaseSolver` and helps
        validate input :class:`BaseProblem` instance
        """
        return set()

    @classmethod
    @abstractmethod
    def optional_in_problem(cls) -> dict:
        r"""a dictionary of components that are optional in :class:`BaseProblem`
        instance

        The keys stand for name of the components in ``BaseProblem``, and the values
        input :class:`BaseProblem` instance
        """
        return dict()

    @classmethod
    @abstractmethod
    def required_in_options(cls) -> set:
        """a set of solver-specific options required in :class:`InversionOptions`
        instance

        This is a standard part for a subclass of :class:`BaseSolver` and helps
        validate input :class:`InversionOptions` instance
        """
        return set()

    @classmethod
    @abstractmethod
    def optional_in_options(cls) -> dict:
        """dict: a dictioanry of solver-specific options that are optional in
        :class:`InversionOptions` instance

        This is a standard part for a subclass of :class:`BaseSolver` and helps
        validate input :class:`InversionOptions` instance
        """
        return dict()

    @abstractmethod
    def __call__(self) -> dict:
        """the method that calls the backend inversion routines

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

    @property
    def components_used(self) -> set:
        r"""a set of strings describing what components defined in :class:`BaseProblem`
        are used in this solving process. This is typically the intersection of three
        sets: :func:`cofi.BaseProblem.all_components`,
        :func:`BaseSolver.required_in_problem` and keys of
        :func:`BaseSolver.optional_in_problem`
        """
        return self._components_used

    @property
    def inv_problem(self):
        r"""the inversion problem to be solved

        This is the first argument in the constructor of :class:`BaseSolver`
        """
        return self._inv_problem

    @property
    def inv_options(self):
        r"""the inveersion settings

        This is the second argument in the constructor of :class:`BaseSolver`
        """
        return self._inv_options

    def _validate_inv_problem(self):
        # check whether enough information from inv_problem is provided
        defined = self.inv_problem.defined_components()
        required = self.required_in_problem()
        if all({component in defined for component in required}):
            return True
        raise ValueError(
            f"you've chosen {self.__class__.__name__} to be your solving tool, but "
            "not enough information is provided in the BaseProblem object - "
            f"required: {required}; provided: {defined}"
        )

    def _validate_inv_options(self):
        # check whether inv_options matches current solver (correctness of dispatch) from callee
        # check whether required options are provided (algorithm-specific)
        defined = self.inv_options.get_params()
        required = self.required_in_options()
        optional = self.optional_in_options()
        all_required_are_defined = all({option in defined for option in required})
        if all_required_are_defined:
            defined_list = list(defined)
            defined_but_not_required_or_optional = [
                option not in optional and option not in required
                for option in defined_list
            ]
            if any(defined_but_not_required_or_optional):
                from itertools import compress

                items = list(
                    compress(defined_list, defined_but_not_required_or_optional)
                )
                warnings.warn(
                    "the following options are defined but not in parameter list for "
                    f"the chosen tool: {items}"
                )
            return True
        raise ValueError(
            f"you've chosen {self.__class__.__name__} to be your solving tool, but "
            "not enough information is provided in the InversionOptions object - "
            f"required: {required}; provided: {defined}"
        )

    def _assign_options(self):
        params = self.inv_options.get_params()
        self._params = dict()
        for opt in self.required_in_options():
            self._params[opt] = params[opt]
        for opt, val in self.optional_in_options().items():
            self._params[opt] = params.get(opt, val)

    def _update_components_used(self):
        self._components_used = list(self.required_in_problem())
        defined_in_problem = self.inv_problem.defined_components()
        optional_components_defined = set(defined_in_problem).union(
            self.optional_in_problem()
        )
        self._components_used.extend(list(optional_components_defined))

    def __repr__(self) -> str:
        return self.__class__.__name__


def error_handler(when, context):
    """Error handler for running solvers"""

    def wrap_error_handler(func):
        def wrapped_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise CofiError(
                    f"error ocurred {when} ({context}). Check exception details "
                    "from message above.",
                ) from e

        return wrapped_func

    return wrap_error_handler
