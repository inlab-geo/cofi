import warnings
import difflib
from typing import Union, Type
from collections.abc import Callable
import json

from .solvers import solver_suggest_table, solver_dispatch_table, solver_methods


class InversionOptions:
    r"""Class for specification on how an inversion will run, including which backend
    tool to use and solver-specific parameters.

    .. tip::

        A typical workflow of :code:`InversionOptions`:

        Step 1 (optional): let the :ref:`Guidance Methods <guide>` to walk you through
        available solving methods in a hierarchical way.

        Step 2: Use the :ref:`Set/Unset Backend Tools <set_unset_tools>` to fix your choice
        on which backend tool to use.

        Step 3: Set solver-specific parameters using the :ref:`Solver Params <set_params>`
        related methods.


    .. admonition:: Example usage of InversionOptions
        :class: dropdown, attention

        >>> from cofi import InversionOptions
        >>> inv_options = InversionOptions()
        >>> inv_options.get_default_tool()
        'scipy.optimize.minimize'
        >>> inv_options.suggest_tools()
        Here's a complete list of inversion solvers supported by CoFI (grouped by methods):
        {
            "optimisation": [
                "scipy.optimize.minimize",
                "scipy.optimize.least_squares"
            ],
            "linear least square": [
                "scipy.linalg.lstsq"
            ]
        }
        >>> inv_options.set_tool("scipy.linalg.lstsq")
        >>> inv_options.suggest_solver_params()
        Current backend tool scipy.linalg.lstsq has the following solver-specific parameters:
        Required parameters:
        -- nothing --
        Optional parameters & default settings:
        {'cond': None, 'overwrite_a': False, 'overwrite_b': False, 'check_finite': True, 'lapack_driver': None}
        >>> inv_options.summary()
        =============================
        Summary for inversion options
        =============================
        Solving method: None set
        Use `suggest_solving_methods()` to check available solving methods.
        -----------------------------
        Backend tool: `scipy.linalg.lstsq` - SciPy's wrapper function over LAPACK's
        linear least-squares solver, using 'gelsd', 'gelsy' (default), or 'gelss' as
        backend driver
        References: ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html',
        'https://www.netlib.org/lapack/lug/node27.html']
        Use `suggest_tools()` to check available backend tools.
        -----------------------------
        Solver-specific parameters: None set
        Use `suggest_solver_params()` to check required/optional solver-specific parameters.

    .. warning::
        Methods that guide users through available **solvers tree** is still under consideration -
        we are working on deciding how such APIs are named and used. Ideally, we have a
        tree in the backend, with the root level branching into ``sampling``, ``direct search``
        and ``optimisation`` and further categorisations that lead to lists of backend tools
        as the leaves.

    .. _guide:

    .. rubric:: Guidance Methods

    Here are how you can walk through our solvers tree as a guidance. Note that this
    step is optional, and you can always jump to :ref:`Set/Unset Backend Tools <set_unset_tools>`
    directly.

    .. autosummary::
        InversionOptions.set_solving_method
        InversionOptions.unset_solving_method

    :ref:`back to top <top_InversionOptions>`

    .. _set_unset_tools:

    .. rubric:: Set/Unset Backend Tools

    To select/unselect a backend tool, use the following methods

    .. autosummary::
        InversionOptions.set_tool
        InversionOptions.unset_tool
        InversionOptions.get_tool
        InversionOptions.get_default_tool
        InversionOptions.suggest_tools

    :ref:`back to top <top_InversionOptions>`

    .. _set_params:

    .. rubric:: Solver Params

    To set tool-specific parameters, use the following methods

    .. autosummary::
        InversionOptions.set_params
        InversionOptions.get_params
        InversionOptions.suggest_solver_params

    :ref:`back to top <top_InversionOptions>`

    """

    def __init__(self):
        self.hyper_params = {}
        self.tool = None
        self.method = None

    def set_params(self, **kwargs):
        r"""Sets solver-specific parameters

        Use :func:`InversionOptions.suggest_solver_params` to get a list of parameters
        required and optional.

        To set the parameters, use argument keyword to specify directly which parameter
        you refer to.

        Examples
        --------

        .. admonition:: code example
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 9

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.suggest_solver_params()
                Current backend tool scipy.optimize.minimize (default) has the following solver-specific parameters:
                Required parameters:
                -- nothing --
                Optional parameters & default settings:
                {'method': None, 'tol': None, 'callback': None, 'options': None}
                >>> inv_options.set_params(method="Nelder-Mead")

        """
        self.hyper_params.update(kwargs)

    def get_params(self) -> dict:
        r"""Get solver-specific parameters defined so far

        Returns
        -------
        dict
            a Python dictionary that maps solver-specific parameter name to the value
            you've set.

        Examples
        --------

        .. admonition:: code example
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 4

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.set_params(method="Nelder-Mead")
                >>> inv_options.get_params()
                {'method': 'Nelder-Mead'}

        """
        return self.hyper_params

    def set_solving_method(self, method: str):
        r"""Sets the solving method

        .. warning::
            The current version is a flattened version of our solvers tree, we are going
            to change this interface very soon.

        Use :func:`InversionOptions.suggest_solving_methods` to get a list of solving
        methods to choose from.

        Parameters
        ----------
        method : str
            the string that represents a solving approach

        Raises
        ------
        ValueError
            when the solving method you attempt to set is invalid

        Examples
        --------

        .. admonition:: code example
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 3

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.set_solving_method("linear least square")
                >>> inv_options.suggest_tools()
                Based on the solving method you've set, the following tools are suggested:
                ['scipy.linalg.lstsq']

                Use `InversionOptions.set_tool(tool_name)` to set a specific tool from above
                Use `InversionOptions.set_solving_method(method_name)` to change solving method
                Use `InversionOptions.unset_solving_method()` if you'd like to see more options
                Check CoFI documentation 'Advanced Usage' section for how to plug in your own solver

        """
        if method is None:
            self.unset_solving_method()
        elif method in solver_methods:
            self.method = method
        else:
            close_matches = difflib.get_close_matches(method, solver_methods)
            _error_msg_suffix = (
                f"\n\nDid you mean '{close_matches[0]}?'" if len(close_matches) else ""
            )
            raise ValueError(
                "the solver method is invalid, please choose from"
                f" {solver_methods}{_error_msg_suffix}"
            )

    def unset_solving_method(self):
        """Unsets the chosen solving approach"""
        self.method = None

    def set_tool(self, tool: Union[str, Type]):
        r"""Sets the tool that will be the backend solver for your inversion problem

        This can be:

        - a backend tool we support, use :func:`InversionOptions.suggest_tools` to get
          a list of tools you can choose from
        - or your own solver class, check `our tutorial - Advanced Usage <tutorial.html#advanced-usage>`_
          for details about how to define and use your custom solver

        Parameters
        ----------
        tool : Union[str, Type]
            either the name of a backend tool or your custom :class:`solver.BaseSolver`
            class

        Raises
        ------
        ValueError
            when the string you pass in isn't in our supported tools list, or when the solver
            class you pass in doesn't implement the ``__call__(self,)`` method.

        Examples
        --------

        .. admonition:: code example: set a supported tool
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 3

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.set_tool("scipy.linalg.lstsq")
                >>> inv_options.summary()
                =============================
                Summary for inversion options
                =============================
                Solving method: None set
                Use `suggest_solving_methods()` to check available solving methods.
                -----------------------------
                Backend tool: `scipy.linalg.lstsq` - SciPy's wrapper function over LAPACK's linear least-squares solver, using 'gelsd', 'gelsy' (default), or 'gelss' as backend driver
                References: ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html', 'https://www.netlib.org/lapack/lug/node27.html']
                Use `suggest_tools()` to check available backend tools.
                -----------------------------
                Solver-specific parameters: None set
                Use `suggest_solver_params()` to check required/optional solver-specific parameters.


        .. admonition:: code example: set your own solver
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 3-8, 11

                >>> from cofi.solvers import BaseSolver
                >>> from cofi import InversionOptions
                >>> class MyDummySolver(BaseSolver):
                ...   short_description = "My dummy solver that always return (1,2) as result"
                ...   def __init__(self, inv_problem, inv_options):
                ...     super().__init__(inv_problem, inv_options)
                ...   def __call__(self):
                ...     return {"model": np.array([1,2]), "success": True}
                ...
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
                References: []
                Use `suggest_tools()` to check available backend tools.
                -----------------------------
                Solver-specific parameters: None set
                Use `suggest_solver_params()` to check required/optional solver-specific parameters.

        """
        if tool is None:
            self.unset_tool()
        elif isinstance(tool, Type):
            if (
                issubclass(tool, Callable)
                and "__call__" not in tool.__abstractmethods__
            ):
                self.tool = tool
            else:
                raise ValueError(
                    "the custom solver class you've provided should implement"
                    " __call__(self,) methodread CoFI documentation 'tutorials -"
                    " Advanced Usage' section for how to plug inyour own solver"
                )
        else:
            if tool not in solver_dispatch_table:
                close_matches = difflib.get_close_matches(
                    tool, solver_dispatch_table.keys()
                )
                _error_msg_suffix = (
                    f"\n\nDid you mean '{close_matches[0]}?'"
                    if len(close_matches)
                    else ""
                )
                raise ValueError(
                    "the tool is invalid, please use"
                    " `InversionOptions.suggest_tools()` to see"
                    f" options{_error_msg_suffix}"
                )
            if self.method and tool not in solver_suggest_table[self.method]:
                warnings.warn(
                    f"the tool {tool} is valid but doesn't match the solving method"
                    f" you've selected: {self.method}"
                )
            self.tool = tool

    def unset_tool(self):
        """Unsets the chosen backend tool"""
        self.tool = None

    def get_tool(self) -> Union[str, Type]:
        """Get the backend tool chosen so far, or the default tool if not chosen

        Returns
        -------
        Union[str, Type]
            the name of the backend tool (if it's supported by us), or the class name
            of your own solver
        """
        return self.tool if self.tool else self.get_default_tool()

    def get_default_tool(self) -> str:
        """Get the default tool based on the chosen solving method

        Returns
        -------
        str
            the name of the default backend tool
        """
        if self.method:
            return solver_suggest_table[self.method][0]
        return solver_suggest_table["optimisation"][0]

    def suggest_solving_methods(self):
        """Prints a list of solving methods to choose from

        Examples
        --------

        .. admonition:: code example
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 3

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.suggest_solving_methods()
                The following solving methods are supported:
                {'optimisation', 'linear least square'}

                Use `suggest_tools()` to see a full list of backend tools for each method

        """
        print("The following solving methods are supported:")
        print(solver_methods)
        print(
            "\nUse `suggest_tools()` to see a full list of backend tools for each"
            " method"
        )

    def suggest_tools(self):
        """Prints a list of tools based on the solving method chosen

        Examples
        --------

        .. admonition:: code example
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 3, 15

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.suggest_tools()
                Here's a complete list of inversion solvers supported by CoFI (grouped by methods):
                {
                    "optimisation": [
                        "scipy.optimize.minimize",
                        "scipy.optimize.least_squares"
                    ],
                    "linear least square": [
                        "scipy.linalg.lstsq"
                    ]
                }
                >>> inv_options.set_solving_method("linear least square")
                >>> inv_options.suggest_tools()
                Based on the solving method you've set, the following tools are suggested:
                ['scipy.linalg.lstsq']

                Use `InversionOptions.set_tool(tool_name)` to set a specific tool from above
                Use `InversionOptions.set_solving_method(method_name)` to change solving method
                Use `InversionOptions.unset_solving_method()` if you'd like to see more options
                Check CoFI documentation 'Advanced Usage' section for how to plug in your own solver

        """
        if self.method:
            tools = solver_suggest_table[self.method]
            print(
                "Based on the solving method you've set, the following tools are"
                " suggested:"
            )
            print(tools)
            print(
                "\nUse `InversionOptions.set_tool(tool_name)` to set a specific tool"
                " from above"
            )
            print(
                "Use `InversionOptions.set_solving_method(method_name)` to change"
                " solving method"
            )
            print(
                "Use `InversionOptions.unset_solving_method()` if you'd like to see"
                " more options"
            )
            print(
                "Check CoFI documentation 'Advanced Usage' section for how to plug in"
                " your own solver"
            )
        else:
            print(
                "Here's a complete list of inversion solvers supported by CoFI (grouped"
                " by methods):"
            )
            print(json.dumps(solver_suggest_table, indent=4))

    def suggest_solver_params(self):
        """Prints required and optional solver-specific parameters

        Examples
        --------

        .. admonition:: code example
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 3

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.suggest_solver_params()
                Current backend tool scipy.optimize.minimize (default) has the following solver-specific parameters:
                Required parameters:
                -- nothing --
                Optional parameters & default settings:
                {'method': None, 'tol': None, 'callback': None, 'options': None}

        """
        tool, dft_suffix = (
            (self.tool, "")
            if self.tool
            else (f"{self.get_default_tool()}", " (default)")
        )
        solver = solver_dispatch_table[tool]
        print(
            f"Current backend tool {tool}{dft_suffix} has the following solver-specific"
            " parameters:"
        )
        print("Required parameters:")
        print(
            solver.required_in_options
            if solver.required_in_options
            else "-- nothing --"
        )
        print("Optional parameters & default settings:")
        print(
            solver.optional_in_options
            if solver.optional_in_options
            else "-- nothing --"
        )

    def summary(self):
        """Helper method that prints a summary of current ``InversionOptions`` object
        to console

        Examples
        --------

        .. admonition:: code example
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 3

                >>> from cofi import InversionOptions
                >>> inv_options = InversionOptions()
                >>> inv_options.summary()
                =============================
                Summary for inversion options
                =============================
                Solving method: None set
                Use `suggest_solving_methods()` to check available solving methods.
                -----------------------------
                Backend tool: `scipy.optimize.minimize (by default)` - SciPy's optimisers that minimises a scalar function with respect to one or more variables, check SciPy's documentation page for a list of methods
                References: ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html']
                Use `suggest_tools()` to check available backend tools.
                -----------------------------
                Solver-specific parameters: None set
                Use `suggest_solver_params()` to check required/optional solver-specific parameters.

        """
        self._summary()

    def _summary(self, display_lines=True):
        # inspiration from keras: https://keras.io/examples/vision/mnist_convnet/
        title = "Summary for inversion options"
        display_width = len(title)
        double_line = "=" * display_width
        single_line = "-" * display_width
        if display_lines:
            print(double_line)
        print(title)
        if display_lines:
            print(double_line)
        solving_method = self.method if self.method else "None set"
        tool, dft_suffix = (
            (self.tool, "")
            if self.tool
            else (f"{self.get_default_tool()}", " (by default)")
        )
        solver = solver_dispatch_table[tool] if isinstance(tool, str) else tool
        print(f"Solving method: {solving_method}")
        print("Use `suggest_solving_methods()` to check available solving methods.")
        if display_lines:
            print(single_line)
        print(f"Backend tool: `{tool}{dft_suffix}` - {solver.short_description}")
        # print(f"Backend tool description: {solver.short_description}")
        print(f"References: {solver.documentation_links}")
        print("Use `suggest_tools()` to check available backend tools.")
        if display_lines:
            print(single_line)
        params_suffix = "None set" if len(self.hyper_params) == 0 else ""
        print(f"Solver-specific parameters: {params_suffix}")
        for key, val in self.hyper_params.items():
            print(f"{key} = {val}")
        print(
            "Use `suggest_solver_params()` to check required/optional solver-specific"
            " parameters."
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        method = f"'{self.method}'" if self.method else "unknown"
        tool = (
            f"'{self.tool}'" if self.tool else f"(default)'{self.get_default_tool()}'"
        )
        return f"{class_name}(method={method},tool={tool})"
