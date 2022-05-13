from numbers import Number
from typing import Callable, Union, Tuple, Sequence
import difflib
import json

import numpy as np

from .solvers import solvers_table


class BaseProblem:
    r"""Base class for an inversion problem setup.

    An inversion problem can be defined from several tiers, depending on the
    level of flexibility or control in ways you'd like to evaluate a model, as
    well as the solving approaches you'd like to apply on the problem.

    To define an inversion problem that is intended to be solved by **optimisation**,
    you may consider the following tiers:

    .. figure:: ../../_static/BaseProblem_opt.svg
       :align: center

    To define an inversion problem that is intended to be solved by **sampling** (WIP),
    here is a rough structure of how you can define it:

    .. figure:: ../../_static/BaseProblem_spl.svg
       :align: center

    .. admonition:: One quick example of BaseProblem
       :class: dropdown, attention

        >>> from cofi import BaseProblem
        >>> import numpy as np
        >>> inv_problem = BaseProblem()
        >>> data_x = np.array([1, 2, 3, 4])
        >>> data_y = np.array([3.2, 3.9, 5.1, 6.2])
        >>> def my_forward(model):
        ...   assert len(model) == 2
        ...   return model[0] + model[1] * data_x
        ...
        >>> inv_problem.set_dataset(data_x, data_y)
        >>> inv_problem.set_forward(my_forward)
        >>> inv_problem.set_data_misfit("L2")
        >>> inv_problem.summary()
        Summary for inversion problem: BaseProblem
        =====================================================================
        Model shape: Unknown
        ---------------------------------------------------------------------
        List of functions/properties set by you:
        ['forward', 'dataset']
        ---------------------------------------------------------------------
        List of functions/properties created based on what you have provided:
        ['objective', 'residual', 'data_misfit']
        ( Note that you did not set regularisation )
        ---------------------------------------------------------------------
        List of functions/properties not set by you:
        ['objective', 'gradient', 'hessian', 'hessian_times_vector', 'residual', 'jacobian', 'jacobian_times_vector', 'data_misfit', 'regularisation', 'initial_model', 'model_shape', 'bounds', 'constraints']

    .. tip::

        Check :ref:`Set Methods <set_methods>` to see a full list of methods to attach
        information about the problem.

    Some blocks above may be deduced from other existing information. For instance,
    once you've defined your data, forward operator and how you'd like to calculate
    the data misfit, we are able to generate ``data_misfit`` and ``residual`` for you.
    The ``summary()`` method prints what blocks you've defined, what are not yet defined,
    and what are generated automatically for you.

    At any point of defining your inversion problem, the ``suggest_solvers()``
    method helps get a list of solvers that can be applied to your problem based on
    what have been supplied so far.

    .. tip::

        :ref:`Helper Methods <helper_methods>` are there to help you illustrate what's in your
        ``BaseProblem`` object.

        Additionally, :ref:`Properties/Functaions <prop_func>` set by you are accessible
        through the ``BaseProblem`` object directly.

    `back to top <#top>`_

    .. _set_methods:

    .. rubric:: Set Methods

    Here are a series of ``set`` methods:

    .. autosummary::
        BaseProblem.set_objective
        BaseProblem.set_gradient
        BaseProblem.set_hessian
        BaseProblem.set_hessian_times_vector
        BaseProblem.set_residual
        BaseProblem.set_jacobian
        BaseProblem.set_jacobian_times_vector
        BaseProblem.set_data_misfit
        BaseProblem.set_regularisation
        BaseProblem.set_forward
        BaseProblem.set_dataset
        BaseProblem.set_dataset_from_file
        BaseProblem.set_initial_model
        BaseProblem.set_model_shape
        .. BaseProblem.set_bounds
        .. BaseProblem.set_constraints

    `back to top <#top>`_

    .. _helper_methods:

    .. rubric:: Helper Methods

    Here are helper methods that check what you've defined to the ``BaseProblem``:

    .. autosummary::

        BaseProblem.summary
        BaseProblem.suggest_solvers
        BaseProblem.defined_components

    `back to top <#top>`_

    .. _prop_func:

    .. rubric:: Properties/Functions of the Problem

    In case you'd like to check, the properties/functions defined using the ``set``
    methods above are attached directly to ``BaseProblem`` and can be accessed:

    .. autosummary::

        BaseProblem.objective
        BaseProblem.gradient
        BaseProblem.hessian
        BaseProblem.hessian_times_vector
        BaseProblem.residual
        BaseProblem.jacobian
        BaseProblem.jacobian_times_vector
        BaseProblem.data_misfit
        BaseProblem.regularisation
        BaseProblem.forward
        BaseProblem.name
        BaseProblem.data_x
        BaseProblem.data_y
        BaseProblem.initial_model
        BaseProblem.model_shape
        BaseProblem.bounds
        BaseProblem.constraints

    `back to top <#top>`_

    """

    all_components = [
        "objective",
        "gradient",
        "hessian",
        "hessian_times_vector",
        "residual",
        "jacobian",
        "jacobian_times_vector",
        "data_misfit",
        "regularisation",
        "forward",
        "dataset",
        "initial_model",
        "model_shape",
        "bounds",
        "constraints",
    ]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def objective(self, model: np.ndarray) -> Number:
        """Method for computing the objective function given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Number
            The objective function value for the given model

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        if self.data_misfit_defined and self.regularisation_defined:
            return self.data_misfit(model) + self.regularisation(model)
        elif self.data_misfit_defined:
            return self.data_misfit(model)
        raise NotImplementedError(
            "`objective` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def gradient(self, model: np.ndarray) -> np.ndarray:
        """Method for computing the gradient of objective function with respect to model, given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        np.ndarray
            the gradient (first derivative) of objective function with repect to the model

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        raise NotImplementedError(
            "`gradient` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def hessian(self, model: np.ndarray) -> np.ndarray:
        """Method for computing the Hessian of objective function with respect to model, given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        np.ndarray
            the Hessian (second derivative) of objective function with respect to the model

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        raise NotImplementedError(
            "`hessian` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def hessian_times_vector(self, model: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Method for computing the dot product of the Hessian and an arbitrary vector, given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate
        vector : np.ndarray
            an arbitrary vector

        Returns
        -------
        np.ndarray
            Hessian times an arbitrary vector

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        if self.hessian_defined:
            return self.hessian(model) @ vector
        raise NotImplementedError(
            "`hessian_times_vector` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def residual(self, model: np.ndarray) -> np.ndarray:
        r"""Method for computing the residual vector given a model.

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        np.ndarray
            the residual vector, :math:`\text{forward}(\text{model})-\text{observations}`

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        if self.forward_defined and self.dataset_defined:
            return self.forward(model) - self.data_y
        raise NotImplementedError(
            "`residual` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def jacobian(self, model: np.ndarray) -> np.ndarray:
        r"""Method for computing the Jacobian of forward function with respect to model, given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        np.ndarray
            the Jacobian matrix, :math:`\frac{\partial{\text{forward}(\text{model})}}{\partial\text{model}}`

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        raise NotImplementedError(
            "`jacobian` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def jacobian_times_vector(
        self, model: np.ndarray, vector: np.ndarray
    ) -> np.ndarray:
        """Method for computing the dot product of the Jacobian and an arbitrary vector, given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate
        vector : np.ndarray
            an arbitrary vector

        Returns
        -------
        np.ndarray
            the Jacobian matrix times the given vector

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        if self.jacobian_defined:
            return self.jacobian(model) @ vector
        raise NotImplementedError(
            "`jacobian_times_vector` is required in the solving approach but you"
            " haven't implemented or added it to the problem setup"
        )

    def data_misfit(self, model: np.ndarray) -> Number:
        """Method for computing the data misfit value given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Number
            the data misfit evaluated based on how you've defined it

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        raise NotImplementedError(
            "`data_misfit` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def regularisation(self, model: np.ndarray) -> Number:
        """Method for computing the regularisation value given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Number
            the regularisation value evaluated based on how you've defined it

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        raise NotImplementedError(
            "`regularisation` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def forward(self, model: np.ndarray) -> Union[np.ndarray, Number]:
        """Method to perform the forward operation given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Union[np.ndarray, Number]
            the synthetics data

        Raises
        ------
        NotImplementedError
            when this method is not set and cannot be deduced
        """
        raise NotImplementedError(
            "`forward` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    # TO ADD a set method, remember to do the following:
    # - def set_something(self, something)
    # - def something(self), this is a property / function
    # - def something_defined(self) -> bool
    # - add checking to self.defined_components
    # - add tests in tests/test_base_problem.py ("test_non_set", etc.)

    def set_objective(self, obj_func: Callable[[np.ndarray], Number]):
        """Sets the function to compute the objective function to minimise

        Alternatively, objective function can be set implicitly (computed by us) if one of
        the following combinations is set:

        - :func:`BaseProblem.set_data_misfit` + :func:`BaseProblem.set_regularisation`
        - :func:`BaseProblem.set_data_misfit` (in this case, regularisation is default
          to 0)

        Parameters
        ----------
        obj_func : Callable[[np.ndarray], Number]
            the objective function that matches :func:`BaseProblem.objective` in
            signature
        """
        if obj_func:
            self.objective = obj_func
        else:
            del self.objective

    def set_gradient(self, grad_func: Callable[[np.ndarray], np.ndarray]):
        """Sets the function to compute the gradient of objective function w.r.t the
        model

        Parameters
        ----------
        obj_func : Callable[[np.ndarray], Number]
            the gradient function that matches :func:`BaseProblem.gradient` in
            signature
        """
        self.gradient = grad_func

    def set_hessian(
        self, hess_func: Union[Callable[[np.ndarray], np.ndarray], np.ndarray]
    ):
        """Sets the function to compute the Hessian of objective function w.r.t the
        model

        Parameters
        ----------
        hess_func : Union[Callable[[np.ndarray], np.ndarray], np.ndarray]
            the Hessian function that matches :func:`BaseProblem.hessian` in
            signature
        """
        if isinstance(hess_func, np.ndarray):
            self.hessian = lambda _: hess_func
        else:
            self.hessian = hess_func

    def set_hessian_times_vector(
        self, hess_vec_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ):
        """Sets the function to compute the Hessian (of objective function) times
        an arbitrary vector

        Alternatively, hessian_times_vector function can be set implicitly (computed by us)
        if :func:`set_hessian` is defined.

        Parameters
        ----------
        hess_vec_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
            the function that computes the product of Hessian and an arbitrary vector,
            in the same signature as :func:`BaseProblem.hessian_times_vector`
        """
        self.hessian_times_vector = hess_vec_func

    def set_residual(self, res_func: Callable[[np.ndarray], np.ndarray]):
        """Sets the function to compute the residual vector/matrix

        Alternatively, residual function can be set implicitly (computed by us)
        if both :func:`set_forward` and dataset (:func:`set_dataset` or
        :func:`set_dataset_from_file`) are defined.

        Parameters
        ----------
        res_func : Callable[[np.ndarray], np.ndarray]
            the residual function that matches :func:`BaseProblem.residual` in
            signature
        """
        self.residual = res_func

    def set_jacobian(
        self, jac_func: Union[Callable[[np.ndarray], np.ndarray], np.ndarray]
    ):
        """Sets the function to compute the Jacobian matrix, namely first
        derivative of forward function with respect to the model

        Parameters
        ----------
        jac_func : Union[Callable[[np.ndarray], np.ndarray], np.ndarray]
            the Jacobian function that matches :func:`BaseProblem.residual` in
            signature
        """
        if isinstance(jac_func, np.ndarray):
            self.jacobian = lambda _: jac_func
        else:
            self.jacobian = jac_func

    def set_jacobian_times_vector(
        self, jac_vec_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ):
        """Sets the function to compute the Jacobian (of forward function) times
        an arbitrary vector

        Alternatively, jacobian_times_vector function can be set implicitly (computed by us)
        if :func:`set_jacobian` is defined.

        Parameters
        ----------
        jac_vec_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
            the function that computes the product of Jacobian and an arbitrary vector,
            in the same signature as :func:`BaseProblem.jacobian_times_vector`
        """
        self.jacobian_times_vector = jac_vec_func

    def set_data_misfit(self, data_misfit: Union[str, Callable[[np.ndarray], Number]]):
        """Sets the function to compute the data misfit

        You can either pass in a custom function or a short string that describes the
        data misfit function. These are a list of pre-built data misfit functions we
        support:

        - "L2"

        If you choose one of the above, then you would also need to use
        :func:`BaseProblem.set_dataset` / :func:`BaseProblem.set_dataset_from_file`
        and :func:`BaseProblem.set_forward` so that we can generate the data misfit
        function for you.

        If the data misfit function you want isn't included above, then pass your own
        function as the input argument.

        Parameters
        ----------
        data_misfit : Union[str, Callable[[np.ndarray], Number]]
            either a string from ["L2"], or a data misfit function that matches
            :func:`BaseProblem.data_misfit` in signature.

        Raises
        ------
        ValueError
            when you've passed in a string not in our supported data misfit list
        """
        if isinstance(data_misfit, str):
            # TODO - define a dict for available data_misfit methods
            if data_misfit in [
                "L2",
                "l2",
                "euclidean",
                "L2 norm",
                "l2 norm",
                "mse",
                "MSE",
            ]:
                self.data_misfit = self._data_misfit_L2
            else:  # TODO - other options?
                raise ValueError(
                    "the data misfit method you've specified isn't supported yet,"
                    " please report an issue here:"
                    " https://github.com/inlab-geo/cofi/issues if you find it valuable"
                    " to support it from our side"
                )
        else:
            self.data_misfit = data_misfit

    def set_regularisation(
        self,
        regularisation: Union[str, Callable[[np.ndarray], Number]],
        factor: Number = 0.1,
    ):
        r"""Sets the function to compute the regularisation

        You can either pass in a custom function or a string/number that describes the
        order of the norm. We use :func:`numpy.linalg.norm` as our backend
        implementation, so the order can be chosen from:

        { ``None``, ``"fro"``, ``"nuc"``, ``numpy.inf``, ``-numpy.inf`` } :math:`\cup\;\mathbb{R}^*`

        Parameters
        ----------
        regularisation : Union[str, Callable[[np.ndarray], Number]]
            either a string from pre-built functions above, or a regularisation function that
            matches :func:`BaseProblem.regularisation` in signature.
        factor : Number, optional
            the regularisation factor that adjusts the ratio of the regularisation
            term over the data misfit, by default 0.1. If ``regularisation`` and ``data_misfit``
            are set but ``objective`` isn't, then we will generate ``objective`` function as
            following: :math:`\text{objective}(model)=\text{data_misfit}(model)+\text{factor}\times\text{regularisation}(model)`

        Raises
        ------
        ValueError
            when you've passed in a string not in our supported regularisation list

        Examples
        --------

        >>> from cofi import BaseProblem
        >>> inv_problem = BaseProblem()
        >>> inv_problem.set_regularisation(1)                      # example 1
        >>> inv_problem.regularisation([1,1])
        0.2
        >>> inv_problem.set_regularisation("inf")                  # example 2
        >>> inv_problem.regularisation([1,1])
        0.1
        >>> inv_problem.set_regularisation(lambda x: sum(x))       # example 3
        >>> inv_problem.regularisation([1,1])
        0.2
        >>> inv_problem.set_regularisation(2, 0.5)                 # example 4
        >>> inv_problem.regularisation([1,1])
        0.7071067811865476
        """
        if (
            isinstance(regularisation, str)
            or isinstance(regularisation, Number)
            or not regularisation
        ):
            ord = regularisation
            if isinstance(ord, str):
                if ord in ["inf", "-inf"]:
                    ord = float(ord)
                elif ord not in ["fro", "nuc"]:
                    raise ValueError(
                        "the regularisation order you've entered is invalid, please"
                        " choose from the following:\n{None, 'fro', 'nuc', numpy.inf,"
                        " -numpy.inf} or any positive number"
                    )
            elif isinstance(ord, Number):
                if ord < 0:
                    raise ValueError(
                        "the regularisation order you've entered is invalid, please"
                        " choose from the following:\n{None, 'fro', 'nuc', numpy.inf,"
                        " -numpy.inf} or any positive number"
                    )
            _reg = lambda x: np.linalg.norm(x, ord=ord)
        else:
            _reg = regularisation
        self.regularisation = lambda m: _reg(m) * factor

    def set_forward(self, forward: Callable[[np.ndarray], Union[np.ndarray, Number]]):
        """Sets the function to perform the forward operation

        Parameters
        ----------
        forward : Callable[[np.ndarray], Union[np.ndarray, Number]]
            the forward function that matches :func:`BaseProblem.forward` in signature
        """
        self.forward = forward

    def set_dataset(self, data_x: np.ndarray, data_y: np.ndarray):
        """Sets the dataset

        Parameters
        ----------
        data_x : np.ndarray
            the features of data points
        data_y : np.ndarray
            the observations
        """
        self._data_x = data_x
        self._data_y = data_y

    def set_dataset_from_file(self, file_path, obs_idx=-1):
        """Sets the dataset for this problem from a give file path

        This function uses :func:`numpy.loadtxt` or :func:`numpy.load` to read
        dataset file, depending on the file type.

        Parameters
        ----------
        file_path : str
            a relative/absolute file path for the dataset
        obs_idx : Union[int,list], optional
            the index/indices of observations within the dataset, by default -1
        """
        delimiter = None  # try to detect what delimiter is used
        if file_path.endswith(("npy", "npz")):
            data = np.load(file_path)
        elif file_path.endswith(("pickle", "pkl")):
            data = np.load(file_path, allow_pickle=True)
        else:
            with open(file_path) as f:
                first_line = f.readline()
                if "," in first_line:
                    delimiter = ","
            data = np.loadtxt(file_path, delimiter=delimiter)
        self.set_dataset(np.delete(data, obs_idx, 1), data[:, obs_idx])

    def set_initial_model(self, init_model: np.ndarray):
        """Sets the starting point for the model

        Once set, we will infer the property :func:`BaseProblem.model_shape` in
        case this is required for some inference solvers

        Parameters
        ----------
        init_model : np.ndarray
            the initial model
        """
        self._initial_model = init_model
        self._model_shape = init_model.shape if hasattr(init_model, "shape") else (1,)

    def set_model_shape(self, model_shape: Tuple):
        """Sets the model shape explicitly

        Parameters
        ----------
        model_shape : Tuple
            a tuple that describes model shape

        Raises
        ------
        ValueError
            when you've defined an initial_model through :func:`BaseProblem.set_initial_model`
            but their shapes don't match
        """
        if self.initial_model_defined and self._model_shape != model_shape:
            try:
                np.reshape(self.initial_model, model_shape)
            except ValueError as e:
                raise ValueError(
                    f"The model_shape you've provided {model_shape} doesn't match the"
                    " initial_model you set which has the shape:"
                    f" {self.initial_model.shape}"
                ) from e
        self._model_shape = model_shape

    def set_bounds(self, bounds: Sequence[Tuple[Number, Number]]):
        """TODO document me

        Parameters
        ----------
        bounds : Sequence[Tuple[Number, Number]]
            _description_
        """
        self._bounds = bounds

    def set_constraints(self, constraints):
        """TODO document me

        Parameters
        ----------
        constraints : _type_
            _description_
        """
        # TODO - what's the type of this? (ref: scipy has Constraint class)
        self._constraints = constraints

    def _defined_components(self, defined_only=True) -> Tuple[set, set]:
        _to_check = self.all_components
        defined = [
            func_name
            for func_name in _to_check
            if getattr(self, f"{func_name}_defined")
        ]
        if defined_only:
            return defined

        def _check_created(elem):
            if (
                elem == "dataset"
            ):  # dataset won't be derived, it's always provided if exists
                return False
            not_defined_by_set_methods = hasattr(getattr(self, elem), "__self__")
            in_base_class = self.__class__.__name__ == "BaseProblem"
            in_sub_class_not_overridden = getattr(self.__class__, elem) == getattr(
                BaseProblem, elem
            )
            not_overridden = in_base_class or in_sub_class_not_overridden
            return not_defined_by_set_methods and not_overridden

        created_by_us = [elem for elem in defined if _check_created(elem)]
        return [elem for elem in defined if elem not in created_by_us], created_by_us

    def defined_components(self) -> set:
        """Returns a set of components that are defined for the ``BaseProblem`` object

        These include both the ones you've set explicitly through the :ref:`Set Methods <set_methods>`
        and the ones that are deduced from existing information.

        Returns
        -------
        set
            a set of strings describing what are defined
        """
        return self._defined_components()

    def suggest_solvers(self, print_to_console=True) -> dict:
        r"""Prints / Returns the backend inversion tool that you can use, based on things
        defined for this ``BaseProblem`` instance, grouped by solving method

        Parameters
        ----------
        print_to_console : bool, optional
            if set to ``True``, this method will both print and return the dictionary
            of backend tools in a tree structure; if set to ``False``, then it will not
            print to console and will only return the dictionary; by default ``True``

        Returns
        -------
        dict
            a tree structure of solving methods we provide, with the leaf nodes being a
            list of backend inversion solver suggested based on what information you've
            provided to this ``BaseProblem`` object

        Examples
        --------

        .. admonition:: example usage for BaseProblem.suggest_solvers()
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 6

                >>> from cofi import BaseProblem
                >>> import numpy as np
                >>> inv_problem = BaseProblem()
                >>> inv_problem.set_initial_model(np.array([1,2,3]))
                >>> inv_problem.set_data_misfit("L2")
                >>> inv_problem.suggest_solvers()
                Based on what you've provided so far, here are possible solvers:
                {
                    "optimisation": [
                        "scipy.optimize.minimize"
                    ],
                    "linear least square": []
                }
                {'optimisation': ['scipy.optimize.minimize'], 'linear least square': []}

        """
        to_suggest = dict()
        all_components = self.defined_components()
        for solving_method in solvers_table:
            backend_tools = solvers_table[solving_method]
            to_suggest[solving_method] = []
            for tool in backend_tools:
                solver_class = backend_tools[tool]
                required = solver_class.required_in_problem
                if required.issubset(all_components):
                    to_suggest[solving_method].append(tool)
        if print_to_console:
            print("Based on what you've provided so far, here are possible solvers:")
            print(json.dumps(to_suggest, indent=4))
        return to_suggest

    @property
    def data_x(self) -> np.ndarray:
        """the features of data points, set by :func:`BaseProblem.set_dataset` or
        :func:`BaseProblem.set_dataset_from_file`

        Raises
        ------
        NameError
            when it's not defined by methods above
        """
        if hasattr(self, "_data_x"):
            return self._data_x
        raise NameError(
            "data has not been set, please use either `set_dataset()` or "
            "`set_dataset_from_file()` to add dataset to the problem setup"
        )

    @property
    def data_y(self) -> np.ndarray:
        """the observations, set by :func:`BaseProblem.set_dataset` or
        :func:`BaseProblem.set_dataset_from_file`

        Raises
        ------
        NameError
            when it's not defined by methods above
        """
        if hasattr(self, "_data_y"):
            return self._data_y
        raise NameError(
            "data has not been set, please use either `set_dataset()` or "
            "`set_dataset_from_file()` to add dataset to the problem setup"
        )

    @property
    def initial_model(self) -> np.ndarray:
        """the initial model, needed for some iterative optimisation tools that
        requires a starting point

        Raises
        ------
        NameError
            when it's not defined (by :func:`BaseProblem.set_initial_model`)
        """
        if hasattr(self, "_initial_model"):
            return self._initial_model
        raise NameError(
            "initial model has not been set, please use `set_initial_model()`"
            " to add to the problem setup"
        )

    @property
    def model_shape(self) -> Union[Tuple, np.ndarray]:
        """the model shape

        Raises
        ------
        NameError
            when it's not defined (by either :func:`BaseProblem.set_model_shape` or
            :func:`BaseProblem.set_model_shape`)
        """
        if hasattr(self, "_model_shape"):
            return self._model_shape
        raise NameError(
            "model shape has not been set, please use either `set_initial_model()`"
            " or `set_model_shape() to add to the problem setup"
        )

    @property
    def bounds(self):
        r"""TODO: document me!

        Raises
        ------
        NameError
            when it's not defined (by :func:`BaseProblem.set_bounds`)
        """
        if hasattr(self, "_bounds"):
            return self._bounds
        raise NameError(
            "bounds have not been set, please use `set_bounds()` to add to the "
            "problem setup"
        )

    @property
    def constraints(self):
        r"""TODO: document me!

        Raises
        ------
        NameError
            when it's not defined (by :func:`BaseProblem.set_constraints`)
        """
        if hasattr(self, "_constraints"):
            return self._constraints
        raise NameError(
            "constraints have not been set, please use `set_constraints()` to add "
            "to the problem setup"
        )

    @property
    def objective_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.objective` has been defined"""
        return self._check_defined(self.objective)

    @property
    def gradient_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.gradient` has been defined"""
        return self._check_defined(self.gradient)

    @property
    def hessian_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.hessian` has been defined"""
        return self._check_defined(self.hessian)

    @property
    def hessian_times_vector_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.hessian_times_vector` has been defined"""
        return self._check_defined(self.hessian_times_vector, 2)

    @property
    def residual_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.residual` has been defined"""
        return self._check_defined(self.residual)

    @property
    def jacobian_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.jacobian` has been defined"""
        return self._check_defined(self.jacobian)

    @property
    def jacobian_times_vector_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.jacobian_times_vector` has been defined"""
        return self._check_defined(self.jacobian_times_vector, 2)

    @property
    def data_misfit_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.data_misfit` has been defined"""
        return self._check_defined(self.data_misfit)

    @property
    def regularisation_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.regularisation` has been defined"""
        return self._check_defined(self.regularisation)

    @property
    def forward_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.forward` has been defined"""
        return self._check_defined(self.forward)

    @property
    def dataset_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.data_x` and :func:`BaseProblem.data_y`
        has been defined
        """
        try:
            self.data_x
            self.data_y
        except NameError:
            return False
        else:
            return True

    @property
    def initial_model_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.initial_model` has been defined"""
        try:
            self.initial_model
        except NameError:
            return False
        else:
            return True

    @property
    def model_shape_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.model_shape` has been defined"""
        try:
            self.model_shape
        except NameError:
            return False
        else:
            return True

    @property
    def bounds_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.bounds` has been defined"""
        try:
            self.bounds
        except NameError:
            return False
        else:
            return True

    @property
    def constraints_defined(self) -> bool:
        r"""indicates whether :func:`BaseProblem.constraints` has been defined"""
        try:
            self.constraints
        except NameError:
            return False
        else:
            return True

    @staticmethod
    def _check_defined(func, args_num=1):
        try:
            func(*[np.array([])] * args_num)
        except NotImplementedError:
            return False
        except:  # it's ok if there're errors caused by dummy input argument np.array([])
            return True
        else:
            return True

    @property
    def name(self) -> str:
        """Name of the current BaseProblem object, for display purposes, no actual
        meaning

        Returns
        -------
        str
            a name you've set
        """
        return self._name if hasattr(self, "_name") else self.__class__.__name__

    @name.setter
    def name(self, problem_name):
        self._name = problem_name

    def _data_misfit_L2(self, model: np.ndarray) -> Number:
        if self.residual_defined:
            res = self.residual(model)
            return np.linalg.norm(res) / res.shape[0]
        else:
            raise ValueError(
                "insufficient information provided to calculate mean squared error"
            )

    def summary(self):
        r"""Helper method that prints a summary of current ``BaseProblem`` object to
        console

        Examples
        --------

        .. admonition:: examples usage for BaseProblem.summary()
            :class: dropdown, attention

            .. code-block:: pycon
                :emphasize-lines: 6

                >>> from cofi import BaseProblem
                >>> import numpy as np
                >>> inv_problem = BaseProblem()
                >>> inv_problem.set_initial_model(np.array([1,2,3]))
                >>> inv_problem.set_data_misfit("L2")
                >>> inv_problem.summary()
                Summary for inversion problem: BaseProblem
                =====================================================================
                Model shape: (3,)
                ---------------------------------------------------------------------
                List of functions/properties set by you:
                ['initial_model', 'model_shape']
                ---------------------------------------------------------------------
                List of functions/properties created based on what you have provided:
                ['objective', 'data_misfit']
                ( Note that you did not set regularisation )
                ---------------------------------------------------------------------
                List of functions/properties not set by you:
                ['objective', 'gradient', 'hessian', 'hessian_times_vector', 'residual', 'jacobian', 'jacobian_times_vector', 'data_misfit', 'regularisation', 'forward', 'dataset', 'bounds', 'constraints']

        """
        self._summary()

    def _summary(self, display_lines=True):
        # inspiration from keras: https://keras.io/examples/vision/mnist_convnet/
        title = f"Summary for inversion problem: {self.name}"
        sub_title1 = "List of functions/properties set by you:"
        sub_title2 = (
            "List of functions/properties created based on what you have provided:"
        )
        sub_title3 = "List of functions/properties not set by you:"
        display_width = max(len(title), len(sub_title1), len(sub_title2))
        double_line = "=" * display_width
        single_line = "-" * display_width
        set_by_user, created_for_user = self._defined_components(False)
        not_set = [
            component
            for component in self.all_components
            if component not in set_by_user
        ]
        print(title)
        if display_lines:
            print(double_line)
        model_shape = self.model_shape if self.model_shape_defined else "Unknown"
        print(f"Model shape: {model_shape}")
        if display_lines:
            print(single_line)
        print(sub_title1)
        print(set_by_user if set_by_user else "-- none --")
        if display_lines:
            print(single_line)
        print(sub_title2)
        print(created_for_user if created_for_user else "-- none --")
        if (
            "objective" in created_for_user
            and self.data_misfit_defined
            and not self.regularisation_defined
        ):
            print("( Note that you did not set regularisation )")
        if display_lines:
            print(single_line)
        print(sub_title3)
        print(not_set if not_set else "-- none --")

    def __repr__(self) -> str:
        return f"{self.name}"
