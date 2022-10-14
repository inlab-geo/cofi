from numbers import Number
from typing import Callable, Union, Tuple, Sequence
import json

import numpy as np

from .exceptions import (
    DimensionMismatchError,
    InvalidOptionError,
    InvocationError,
    NotDefinedError,
)


class BaseProblem:
    r"""Base class for an inversion problem setup.

    An inversion problem can be defined in different ways, depending on the level of
    flexibility or control in ways you'd like to evaluate a model, as well as the
    solving approaches you'd like to apply on the problem.

    To define an inversion problem that is intended to be solved by **parameter estimation**,
    you may consider setting the following functions or properties:

    - ``objective`` function, or
    - ``data_misfit`` function plus ``regularization`` function
    - ``data_misfit="L2"``, ``data``, ``forward`` and ``regularization`` function
    - In addition, it can sometimes be helpful (e.g. increase the speed of inversion)
      to define more things in a ``BaseProblem`` object: ``gradient`` of objective
      function, ``residual`` vector, ``jacobian`` of forward function, etc.

    To define an inversion problem that is intended to be solved by **ensemble methods**
    (work in progress),
    you may consider setting the following functions or properties:

    - ``log_posterier`` function, or
    - ``log_likelihood`` and ``log_prior`` functions

    .. TBD: we will also add support for ``bounds`` and ``constraints`` as a part of
    .. ``BaseProblem`` definition.

    Here is a complete list of how we would deduce from existing information about the
    ``BaseProblem`` object you've defined:

    .. list-table:: Table: user defined -> we generate for you
        :widths: 35 35 30
        :header-rows: 1

        * - what you define
          - what we generate for you
          - examples
        * - ``data_misfit``
          - ``objective`` (assuming there's no regularization)
          - (work in progress)
        * - ``data_misfit``, ``regularization``
          - ``objective``
          - `linear regression (optimizer) <https://github.com/inlab-geo/cofi-examples/blob/main/notebooks/linear_regression/linear_regression_optimizer_minimize.py>`_
        * - ``forward``, ``data``
          - ``residual``
          - (work in progress)
        * - ``hessian``
          - ``hessian_times_vector``
          - (work in progress)
        * - ``jacobian``
          - ``jacobian_times_vector``
          - `linear regression (linear system solver) <https://github.com/inlab-geo/cofi-examples/blob/main/notebooks/linear_regression/linear_regression_linear_system_solver.py>`_
        * - ``log_prior``, ``log_likelihood``
          - ``log_posterior``
          - `linear regression (sampler) <https://github.com/inlab-geo/cofi-examples/blob/main/notebooks/linear_regression/linear_regression_emcee_sampler.py>`_


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
        >>> inv_problem.set_data(data_y)
        >>> inv_problem.set_forward(my_forward)
        >>> inv_problem.set_data_misfit("L2")
        >>> inv_problem.summary()
        Summary for inversion problem: BaseProblem
        =====================================================================
        Model shape: Unknown
        ---------------------------------------------------------------------
        List of functions/properties set by you:
        ['forward', 'data']
        ---------------------------------------------------------------------
        List of functions/properties created based on what you have provided:
        ['objective', 'residual', 'data_misfit']
        ( Note that you did not set regularization )
        ---------------------------------------------------------------------
        List of functions/properties not set by you:
        ['objective', 'gradient', 'hessian', 'hessian_times_vector', 'residual',
        'jacobian', 'jacobian_times_vector', 'data_misfit', 'regularization',
        'initial_model', 'model_shape', 'bounds', 'constraints']

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

    :ref:`back to top <top_BaseProblem>`

    .. _set_methods:

    .. rubric:: Set Methods

    Here are a series of ``set`` methods:

    .. autosummary::
        BaseProblem.set_objective
        BaseProblem.set_log_posterior
        BaseProblem.set_log_posterior_with_blobs
        BaseProblem.set_blobs_dtype
        BaseProblem.set_log_likelihood
        BaseProblem.set_log_prior
        BaseProblem.set_gradient
        BaseProblem.set_hessian
        BaseProblem.set_hessian_times_vector
        BaseProblem.set_residual
        BaseProblem.set_jacobian
        BaseProblem.set_jacobian_times_vector
        BaseProblem.set_data_misfit
        BaseProblem.set_regularization
        BaseProblem.set_forward
        BaseProblem.set_data
        BaseProblem.set_data_covariance
        BaseProblem.set_data_covariance_inv
        BaseProblem.set_data_from_file
        BaseProblem.set_initial_model
        BaseProblem.set_model_shape
        BaseProblem.set_walkers_starting_pos
        .. BaseProblem.set_bounds
        .. BaseProblem.set_constraints

    :ref:`back to top <top_BaseProblem>`

    .. _helper_methods:

    .. rubric:: Helper Methods

    Here are helper methods that check what you've defined to the ``BaseProblem``:

    .. autosummary::

        BaseProblem.summary
        BaseProblem.suggest_solvers
        BaseProblem.defined_components

    :ref:`back to top <top_BaseProblem>`

    .. _prop_func:

    .. rubric:: Properties/Functions of the Problem

    In case you'd like to check, the properties/functions defined using the ``set``
    methods above are attached directly to ``BaseProblem`` and can be accessed:

    .. autosummary::

        BaseProblem.objective
        BaseProblem.log_posterior
        BaseProblem.log_posterior_with_blobs
        BaseProblem.log_likelihood
        BaseProblem.log_prior
        BaseProblem.gradient
        BaseProblem.hessian
        BaseProblem.hessian_times_vector
        BaseProblem.residual
        BaseProblem.jacobian
        BaseProblem.jacobian_times_vector
        BaseProblem.data_misfit
        BaseProblem.regularization
        BaseProblem.regularization_matrix
        BaseProblem.regularization_factor
        BaseProblem.forward
        BaseProblem.name
        BaseProblem.data
        BaseProblem.data_covariance
        BaseProblem.data_covariance_inv
        BaseProblem.model_covariance
        BaseProblem.model_covariance_inv
        BaseProblem.initial_model
        BaseProblem.model_shape
        BaseProblem.walkers_starting_pos
        BaseProblem.blobs_dtype
        BaseProblem.bounds
        BaseProblem.constraints

    :ref:`back to top <top_BaseProblem>`

    """

    all_components = [
        "objective",
        "log_posterior",
        "log_posterior_with_blobs",
        "log_likelihood",
        "log_prior",
        "gradient",
        "hessian",
        "hessian_times_vector",
        "residual",
        "jacobian",
        "jacobian_times_vector",
        "data_misfit",
        "regularization",
        "regularization_matrix",
        "regularization_factor",
        "forward",
        "data",
        "data_covariance",
        "data_covariance_inv",
        "initial_model",
        "model_shape",
        "walkers_starting_pos",
        "blobs_dtype",
        "bounds",
        "constraints",
    ]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def objective(self, model: np.ndarray, *args, **kwargs) -> Number:
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
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="objective")

    def log_posterior(self, model: np.ndarray, *args, **kwargs) -> Number:
        """Method for computing the log of posterior probability density given a model

        This is typically the sum of log prior and log likelihood.

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Number
            the posterior probability density value

        Raises
        ------
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="log_posterior")

    def log_posterior_with_blobs(
        self, model: np.ndarray, *args, **kwargs
    ) -> Tuple[Number]:
        """Method for computing the log of posterior probability density and related
        information given a model

        The "related information" can be defined by you
        (via :meth:`set_log_posterior_with_blobs`), but they will only be
        stored properly when you perform sampling with ``emcee``.

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Tuple[Number]
            the posterior probability density value, and other information you've set to
            return together with the former

        Raises
        ------
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="log_posterior_with_blobs")

    def log_prior(self, model: np.ndarray, *args, **kwargs) -> Number:
        """Method for computing the log of prior probability density given a model

        This reflects your prior belief about the model distribution.

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Number
            the prior probability density value

        Raises
        ------
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="log_prior")

    def log_likelihood(self, model: np.ndarray, *args, **kwargs) -> Number:
        """Method for computing the log of likelihood probability density given a model

        This reflects the probability distribution of the observations given the model.

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Number
            the likelihood probability density value

        Raises
        ------
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="log_likelihood")

    def gradient(self, model: np.ndarray, *args, **kwargs) -> np.ndarray:
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
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="gradient")

    def hessian(self, model: np.ndarray, *args, **kwargs) -> np.ndarray:
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
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="hessian")

    def hessian_times_vector(
        self, model: np.ndarray, vector: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
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
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="hessian_times_vector")

    def residual(self, model: np.ndarray, *args, **kwargs) -> np.ndarray:
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
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="residual")

    def jacobian(self, model: np.ndarray, *args, **kwargs) -> np.ndarray:
        r"""method for computing the jacobian of forward function with respect to model, given a model

        parameters
        ----------
        model : np.ndarray
            a model to evaluate

        returns
        -------
        np.ndarray
            the jacobian matrix, :math:`\frac{\partial{\text{forward}(\text{model})}}{\partial\text{model}}`

        raises
        ------
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="jacobian")

    def jacobian_times_vector(
        self, model: np.ndarray, vector: np.ndarray, *args, **kwargs
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
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="jacobian_times_vector")

    def data_misfit(self, model: np.ndarray, *args, **kwargs) -> Number:
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
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="data_misfit")

    def regularization(self, model: np.ndarray, *args, **kwargs) -> Number:
        """Method for computing the regularization value given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        Number
            the regularization value evaluated based on how you've defined it

        Raises
        ------
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="regularization")

    def regularization_matrix(self, model: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Method for computing the regularization weighting matrix

        Parameters
        ----------
        model : np.ndarray
            a model that helps calculate regularization matrix. In most cases this is
            not needed, but you have the flexibility to set this as a function

        Returns
        -------
        np.ndarray
            the regularization matrix of dimension ``(model_size, model_size)``

        Raises
        ------
        NotDefinedError
            when this method is not set
        """
        raise NotDefinedError(needs="regularization_matrix")

    def forward(self, model: np.ndarray, *args, **kwargs) -> Union[np.ndarray, Number]:
        """Method to perform the forward operation given a model

        Parameters
        ----------
        model : np.ndarray
            a model to evaluate

        Returns
        -------
        np.ndarray or Number
            the synthetics data

        Raises
        ------
        NotDefinedError
            when this method is not set and cannot be generated from known information
        """
        raise NotDefinedError(needs="forward")

    # TO ADD a set method, remember to do the following:
    # - def set_something(self, something)
    # - def something(self), this is a property / function
    # - def something_defined(self) -> bool
    # - add checking to self.dall_components
    # - add `set_something` and `something` to documentation list on top of this file
    # - check if there's anything to add to autogen_table
    # - add tests in tests/test_base_problem.py ("test_non_set", etc.)

    def set_objective(
        self,
        obj_func: Callable[[np.ndarray], Number],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the objective function to minimize

        Alternatively, objective function can be set implicitly (computed by us) if one of
        the following combinations is set:

        - :meth:`set_data_misfit` + :meth:`set_regularization`
        - :meth:`set_data_misfit` (in this case, regularization is default
          to 0)

        Parameters
        ----------
        obj_func : Callable[[np.ndarray], Number]
            the objective function that matches :meth:`objective` in
            signature
        args : list, optional
            extra list of positional arguments for the objective function
        kwargs : dict, optional
            extra dict of keyword arguments for the objective function
        """
        self.objective = _FunctionWrapper("objective", obj_func, args, kwargs)
        self._update_autogen("objective")

    def set_log_posterior(
        self,
        log_posterior_func: Callable[[np.ndarray], Number],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the log of posterior probability density

        Alternatively, log_posterior function can be set implicitly (computed by us) if
        :func:`set_log_prior` and :func:`set_log_likelihood` are defined.

        Parameters
        ----------
        log_posterior_func : Callable[[np.ndarray], Number]
            the log_posterior function that matches :meth:`log_posterior`
            in signature
        args : list, optional
            extra list of positional arguments for log_posterior function
        kwargs : dict, optional
            extra dict of keyword arguments for log_posterior function

        """
        self.log_posterior = _FunctionWrapper(
            "log_posterior", log_posterior_func, args, kwargs
        )
        self._update_autogen("log_posterior")

    def set_log_posterior_with_blobs(
        self,
        log_posterior_blobs_func: Callable[[np.ndarray], Tuple[Number]],
        blobs_dtype=None,
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function that computes the log of posterior prabability density
        and returns extra information along with log posterior

        The extra blobs returned will only get used when you are using ``emcee`` to
        sample the posterior distribution. Check
        `this emcee documentation page <https://emcee.readthedocs.io/en/stable/user/blobs/>`_
        to understand what blobs are.

        If you use other backend samplers, you can still set ``log_posterior`` using
        this function, and we will generate :meth:`log_posterior` to return
        only the first output from :meth:`log_posterior_with_blobs`.

        This method is also generated automatically by us if you've defined both
        :meth:`log_prior` and :meth:`log_likelihood`. In that
        case, the ``blobs_dtype`` is set to be
        ``[("log_likelihood", float), ("log_prior", float)]``.

        Parameters
        ----------
        log_posterior_blobs_func : Callable[[np.ndarray], Tuple[Number]
            the log_posterior_with_blobs function that matches
            :meth:`log_posterior_blobs_func` in signature
        blobs_dtype : list, optional
            a list of tuples that specify the names and type of the blobs, e.g.
            ``[("log_likelihood", float), ("log_prior", float)]``. If not set, the
            blobs will still be recorded during sampling in the order they are
            returned from :meth:`log_posterior_blobs_func`
        args : list, optional
            extra list of positional arguments for log_posterior function
        kwargs : dict, optional
            extra dict of keyword arguments for log_posterior function
        """
        self.log_posterior_with_blobs = _FunctionWrapper(
            "log_posterior_with_blobs", log_posterior_blobs_func, args, kwargs
        )
        self._update_autogen("log_posterior_with_blobs")
        if blobs_dtype:
            self._blobs_dtype = blobs_dtype

    def set_blobs_dtype(self, blobs_dtype: list):
        r"""Sets the name and type for the extra information you'd like to calculate on
        each sampling step

        This only gets used when you are using ``emcee`` to sample the posterior
        distribution. Check `this emcee documentation page <https://emcee.readthedocs.io/en/stable/user/blobs/>`_
        to understand what blobs are.

        Parameters
        ----------
        blobs_dtype : list
            a list of tuples that specify the names and type of the blobs, e.g.
            ``[("log_likelihood", float), ("log_prior", float)]``
        """
        self._blobs_dtype = blobs_dtype
        self._update_autogen("blobs_dtype")

    def set_log_prior(
        self,
        log_prior_func: Callable[[np.ndarray], Number],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the log of prior probability density

        Parameters
        ----------
        log_prior_func : Callable[[np.ndarray], Number]
            the log_prior function that matches :meth:`log_prior`
            in signature
        args : list, optional
            extra list of positional arguments for log_prior function
        kwargs : dict, optional
            extra dict of keyword arguments for log_prior function
        """
        self.log_prior = _FunctionWrapper("log_prior", log_prior_func, args, kwargs)
        self._update_autogen("log_prior")

    def set_log_likelihood(
        self,
        log_likelihood_func: Callable[[np.ndarray], Number],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the log of likelihood probability density

        Parameters
        ----------
        log_likelihood_func : Callable[[np.ndarray], Number]
            the log_likelihood function that matches :meth:`log_likelihood`
            in signature
        args : list, optional
            extra list of positional arguments for log_likelihood function
        kwargs : dict, optional
            extra dict of keyword arguments for log_likelihood function
        """
        self.log_likelihood = _FunctionWrapper(
            "log_likelihood", log_likelihood_func, args, kwargs
        )
        self._update_autogen("log_likelihood")

    def set_gradient(
        self,
        grad_func: Callable[[np.ndarray], np.ndarray],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the gradient of objective function w.r.t the
        model

        Parameters
        ----------
        obj_func : Callable[[np.ndarray], Number]
            the gradient function that matches :meth:`gradient` in
            signature
        args : list, optional
            extra list of positional arguments for gradient function
        kwargs : dict, optional
            extra dict of keyword arguments for gradient function
        """
        self.gradient = _FunctionWrapper("gradient", grad_func, args, kwargs)
        self._update_autogen("gradient")

    def set_hessian(
        self,
        hess_func: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the Hessian of objective function w.r.t the
        model

        Parameters
        ----------
        hess_func : (function - np.ndarray -> np.ndarray) or np.ndarray
            the Hessian function that matches :meth:`hessian` in
            signature. Alternatively, provide a matrix if the Hessian is a constant.
        args : list, optional
            extra list of positional arguments for hessian function
        kwargs : dict, optional
            extra dict of keyword arguments for hessian function
        """
        if isinstance(hess_func, np.ndarray):
            self.hessian = _FunctionWrapper(
                "hessian", _matrix_to_func, args=[hess_func]
            )
        else:
            self.hessian = _FunctionWrapper("hessian", hess_func, args, kwargs)
        self._update_autogen("hessian")

    def set_hessian_times_vector(
        self,
        hess_vec_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the Hessian (of objective function) times
        an arbitrary vector

        Alternatively, hessian_times_vector function can be set implicitly (computed by us)
        if :func:`set_hessian` is defined.

        Parameters
        ----------
        hess_vec_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
            the function that computes the product of Hessian and an arbitrary vector,
            in the same signature as :meth:`hessian_times_vector`
        args : list, optional
            extra list of positional arguments for hessian_times_vector function
        kwargs : dict, optional
            extra dict of keyword arguments for hessian_times_vector function
        """
        self.hessian_times_vector = _FunctionWrapper(
            "hessian_times_vector", hess_vec_func, args, kwargs
        )
        self._update_autogen("hessian_times_vector")

    def set_residual(
        self,
        res_func: Callable[[np.ndarray], np.ndarray],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the residual vector/matrix

        Alternatively, residual function can be set implicitly (computed by us)
        if both :func:`set_forward` and data (:func:`set_data` or
        :func:`set_data_from_file`) are defined.

        Parameters
        ----------
        res_func : Callable[[np.ndarray], np.ndarray]
            the residual function that matches :meth:`residual` in
            signature
        args : list, optional
            extra list of positional arguments for residual function
        kwargs : dict, optional
            extra dict of keyword arguments for residual function
        """
        self.residual = _FunctionWrapper("residual", res_func, args, kwargs)
        self._update_autogen("residual")

    def set_jacobian(
        self,
        jac_func: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the Jacobian matrix, namely first
        derivative of forward function with respect to the model

        Parameters
        ----------
        jac_func : (function - np.ndarray -> np.ndarray) or np.ndarray
            the Jacobian function that matches :meth:`residual` in
            signature. Alternatively, provide a matrix if the Jacobian is a constant.
        args : list, optional
            extra list of positional arguments for jacobian function
        kwargs : dict, optional
            extra dict of keyword arguments for jacobian function
        """
        if isinstance(jac_func, np.ndarray):
            self.jacobian = _FunctionWrapper(
                "jacobian", _matrix_to_func, args=[jac_func]
            )
        else:
            self.jacobian = _FunctionWrapper("jacobian", jac_func, args, kwargs)
        self._update_autogen("jacobian")

    def set_jacobian_times_vector(
        self,
        jac_vec_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the Jacobian (of forward function) times
        an arbitrary vector

        Alternatively, jacobian_times_vector function can be set implicitly (computed by us)
        if :func:`set_jacobian` is defined.

        Parameters
        ----------
        jac_vec_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
            the function that computes the product of Jacobian and an arbitrary vector,
            in the same signature as :meth:`jacobian_times_vector`
        args : list, optional
            extra list of positional arguments for jacobian_times_vector function
        kwargs : dict, optional
            extra dict of keyword arguments for jacobian_times_vector function
        """
        self.jacobian_times_vector = _FunctionWrapper(
            "jacobian_times_vector", jac_vec_func, args, kwargs
        )
        self._update_autogen("jacobian_times_vector")

    def set_data_misfit(
        self,
        data_misfit: Union[str, Callable[[np.ndarray], Number]],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the data misfit

        You can either pass in a custom function or a short string that describes the
        data misfit function (e.g. ``"squared error"``)

        If you choose ``data_misfit="squared error"``, and:

        - If you have :meth:`residual` defined, or :meth:`data` and :meth:`forward`
          defined, then :math:`\text{data_misfit}=\text{residual}^T \text{residual}`

          - where :math:`\text{residual}=\text{forward}(\text{model})-\text{observations}`

        - If you **additionally** have :meth:`data_covariance_inv` defined, then
          :math:`\text{data_misfit}=\text{residual}^TC_d^{-1}\text{residual}`

          - where :math:`C_d^{-1}=\text{data_covariance_inv}`

        - Otherwise you might face an error when actually calling the
          :meth:`data_misfit` method.

        Alternatively, pass in your own data misfit function (or objective function
        directly through :meth:`set_objective`).

        Parameters
        ----------
        data_misfit : str or (function - np.ndarray -> Number)
            either ``"squared error"``, or a data misfit function that matches
            :meth:`data_misfit` in signature.
        args : list, optional
            extra list of positional arguments for data_misfit function
        kwargs : dict, optional
            extra dict of keyword arguments for data_misfit function

        Raises
        ------
        InvalidOptionError
            when you've passed in a string not in our supported data misfit list
        """
        if isinstance(data_misfit, str):
            # if we have more options later, handle in same way as set_regularization
            if data_misfit in [
                "least squares",
                "least square",
                "squared error",
            ]:
                self.data_misfit = _FunctionWrapper(
                    "data_misfit", self._data_misfit_squared_error, autogen=True
                )
            else:
                raise InvalidOptionError(
                    name="data misfit",
                    invalid_option=data_misfit,
                    valid_options=["least squares"],
                )
        else:
            self.data_misfit = _FunctionWrapper(
                "data_misfit", data_misfit, args, kwargs
            )
        self._update_autogen("data_misfit")

    def set_regularization(
        self,
        regularization: Union[str, Callable[[np.ndarray], Number]],
        regularization_factor: Number = 1,
        regularization_matrix: Union[
            np.ndarray, Callable[[np.ndarray], np.ndarray]
        ] = None,
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to compute the regularization

        You can either pass in a custom function or a string/number that describes the
        order of the norm. We use :func:`numpy.linalg.norm` as our backend
        implementation, so the order can be chosen from:

        { ``None``, ``"fro"``, ``"nuc"``, ``numpy.inf``, ``-numpy.inf`` } :math:`\cup\;\mathbb{R}^*`

        Parameters
        ----------
        regularization : str or (function - np.ndarray -> Number)
            either a string from pre-built functions above, or a regularization function that
            matches :meth:`regularization` in signature.
        regularization_factor : Number, optional
            the regularization factor (lamda) that adjusts the ratio of the regularization
            term over the data misfit, by default 1. If ``regularization`` and ``data_misfit``
            are set but ``objective`` isn't, then we will generate ``objective`` function as
            following: :math:`\text{objective}(model)=\text{data_misfit}(model)+\text{factor}\times\text{regularization}(model)`
        regularization_matrix : np.ndarray or (function - np.ndarray -> np.ndarray)
            a matrix of shape ``(model_size, model_size)``, or a function that takes in
            a model and calculates the (weighting) matrix.

            - If this is None,
              :math:`\text{regularization}(model)=\lambda\times\text{regularization}(model)`
            - If this is set to be a matrix (np.ndarray, or other array like types),
              :math:`\text{regularization}(model)=\lambda\times\text{regularization}(\text{regularization_matrix}\cdot model)`
            - If this is set to be a function that returns a matrix,
              :math:`\text{regularization}(model)=\lambda\times\text{regularization}(\text{regularization_matrix}(model)\cdot model)`

        args : list, optional
            extra list of positional arguments for regularization function
        kwargs : dict, optional
            extra dict of keyword arguments for regularization function

        Raises
        ------
        InvalidOptionError
            when you've passed in a string not in our supported regularization list

        Examples
        --------

        We demonstrate with a few examples on a ``BaseProblem`` instance.

        >>> from cofi import BaseProblem
        >>> inv_problem = BaseProblem()

        1. Example with an L1 norm

        >>> inv_problem.set_regularization(1)
        >>> inv_problem.regularization([1,1])
        2

        2. Example with an inf norm

        >>> inv_problem.set_regularization("inf")
        >>> inv_problem.regularization([1,1])
        1

        3. Example with a custom regularization function

        >>> inv_problem.set_regularization(lambda x: sum(x))
        >>> inv_problem.regularization([1,1])
        2

        4. Example with an L2 norm and regularization factor of 0.5 (by default 1)

        >>> inv_problem.set_regularization(2, 0.5)
        >>> inv_problem.regularization([1,1])
        0.7071067811865476

        5. Example with a regularization matrix

        >>> inv_problem.set_regularization(2, 0.5, np.array([[2,0], [0,1]]))
        >>> inv_problem.regularization([1,1])
        1.118033988749895
        """
        # preprocess regularization_matrix
        if np.ndim(regularization_matrix) != 0:
            self.regularization_matrix = _FunctionWrapper(
                "regularization_matrix", _matrix_to_func, args=[regularization_matrix]
            )
        elif callable(regularization_matrix):
            self.regularization_matrix = _FunctionWrapper(
                "regularization_matrix", regularization_matrix
            )
        else:
            self.regularization_matrix = None
        # preprocess regularization function without lambda
        if isinstance(regularization, (Number, str)) or not regularization:
            order = regularization
            if (
                isinstance(order, str)
                and order not in ["fro", "nuc", "inf", "-inf"]
                or isinstance(order, Number)
                and order < 0
            ):
                raise InvalidOptionError(
                    name="regularization order",
                    invalid_option=order,
                    valid_options=(
                        "[None, 'fro', 'nuc', numpy.inf, -numpy.inf] or any positive"
                        " number"
                    ),
                )
            elif isinstance(order, str) and order in ["inf", "-inf"]:
                order = float(order)
            _reg = _FunctionWrapper(
                "regularization_none_lamda", np.linalg.norm, args=[order]
            )
        else:
            _reg = _FunctionWrapper(
                "regularization_none_lamda", regularization, args, kwargs
            )
        # wrapper function that calculates: lambda * raw regularization value
        self._regularization_factor = regularization_factor
        if self.regularization_matrix is None:
            self.regularization = _FunctionWrapper(
                "regularization",
                _regularization_with_lamda,
                args=[_reg, regularization_factor],
            )
        else:
            self.regularization = _FunctionWrapper(
                "regularization",
                _regularization_with_lamda_n_matrix,
                args=[_reg, regularization_factor, self.regularization_matrix],
            )
        # update some autogenerated functions (as usual)
        self._update_autogen("regularization")

    def set_forward(
        self,
        forward: Callable[[np.ndarray], Union[np.ndarray, Number]],
        args: list = None,
        kwargs: dict = None,
    ):
        r"""Sets the function to perform the forward operation

        Parameters
        ----------
        forward : function - np.ndarray -> (np.ndarray or Number)
            the forward function that matches :meth:`forward` in signature
        args : list, optional
            extra list of positional arguments for forward function
        kwargs : dict, optional
            extra dict of keyword arguments for forward function
        """
        self.forward = _FunctionWrapper("forward", forward, args, kwargs)
        self._update_autogen("forward")

    def set_data(
        self,
        data_obs: np.ndarray,
        data_cov: np.ndarray = None,
        data_cov_inv: np.ndarray = None,
    ):
        """Sets the data observations and optionally data covariance matrix

        Parameters
        ----------
        data_obs : np.ndarray
            the observations
        data_cov : np.ndarray, optional
            the data covariance matrix that helps estimate uncertainty, with dimension
            (N,N) where N is the number of data points
        """
        self._data = data_obs
        self._update_autogen("data")
        if data_cov is not None:
            self.set_data_covariance(data_cov)
        if data_cov_inv is not None:
            self.set_data_covariance_inv(data_cov_inv)

    def set_data_covariance(self, data_cov: np.ndarray):
        """Sets the data covariance matrix to help estimate uncertainty

        Parameters
        ----------
        data_cov : np.ndarray
            the data covariance matrix, with dimension (N,N) where N is the number
            of data points
        """
        self._data_covariance = data_cov
        self._update_autogen("data_covariance")

    def set_data_covariance_inv(self, data_cov_inv: np.ndarray):
        """Sets the data covariance matrix to help estimate uncertainty

        Parameters
        ----------
        data_cov : np.ndarray
            the data covariance matrix, with dimension (N,N) where N is the number
            of data points
        """
        self._data_covariance_inv = data_cov_inv
        self._update_autogen("data_covariance_inv")

    def set_data_from_file(self, file_path, obs_idx=-1, data_cov: np.ndarray = None):
        r"""Sets the data for this problem from a give file path

        This function uses :func:`numpy.loadtxt` or :func:`numpy.load` to read
        data file, depending on the file type.

        Parameters
        ----------
        file_path : str
            a relative/absolute file path for the data
        obs_idx : int or list, optional
            the index/indices of observations within the data file, by default -1
        data_cov : np.ndarray, optional
            the data covariance matrix that helps estimate uncertainty, with dimension
            (N,N) where N is the number of data points
        """
        delimiter = None  # try to detect what delimiter is used
        if file_path.endswith(("npy", "npz")):
            data = np.load(file_path)
        elif file_path.endswith(("pickle", "pkl")):
            data = np.load(file_path, allow_pickle=True)
        else:
            with open(file_path) as file:
                first_line = file.readline()
                if "," in first_line:
                    delimiter = ","
            data = np.loadtxt(file_path, delimiter=delimiter)
        self.set_data(data[:, obs_idx], data_cov)

    def set_initial_model(self, init_model: np.ndarray):
        r"""Sets the starting point for the model

        Once set, we will infer the property :meth:`model_shape` in
        case this is required for some inference solvers

        Parameters
        ----------
        init_model : np.ndarray
            the initial model
        """
        self._initial_model = init_model
        self._model_shape = init_model.shape if hasattr(init_model, "shape") else (1,)

    def set_model_shape(self, model_shape: Tuple):
        r"""Sets the model shape explicitly

        Parameters
        ----------
        model_shape : Tuple
            a tuple that describes model shape

        Raises
        ------
        DimensionMismatchError
            when you've defined an initial_model through :meth:`set_initial_model`
            but their shapes don't match
        """
        if self.initial_model_defined and self._model_shape != model_shape:
            try:
                np.reshape(self.initial_model, model_shape)
            except ValueError as err:
                raise DimensionMismatchError(
                    entered_dimension=model_shape,
                    entered_name="model shape",
                    expected_dimension=self.initial_model.shape,
                    expected_source="initial model",
                ) from err
        self._model_shape = model_shape

    def set_walkers_starting_pos(self, starting_pos: np.ndarray):
        r"""Sets the starting positions for each walker in sampling methods

        This initialisation is optional. If not set, we rely on the default behaviour
        of the backend sampling tools.

        Parameters
        ----------
        starting_pos : np.ndarray
            starting positions, with the shape ``(nwalkers, ndims)``, where
            ``nwalkers`` is the number of walkers you plan to use for the sampler, and
            ``ndims`` is the dimension of your model parameters (for fixed dimension
            samplers)
        """
        self._walkers_starting_pos = starting_pos
        self._model_shape = (starting_pos.shape[1],)

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

        def _check_autogen(elem):
            _elem = getattr(self, elem)
            return isinstance(_elem, _FunctionWrapper) and _elem.autogen

        created_by_us = [elem for elem in defined if _check_autogen(elem)]
        return [elem for elem in defined if elem not in created_by_us], created_by_us

    def defined_components(self) -> set:
        r"""Returns a set of components that are defined for the ``BaseProblem`` object

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
                    "optimization": [
                        "scipy.optimize.minimize"
                    ],
                    "matrix solvers": []
                }
                {'optimization': ['scipy.optimize.minimize'], 'matrix solvers': []}

        """
        to_suggest = dict()
        all_components = self.defined_components()
        from .solvers import solvers_table

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
    def data(self) -> np.ndarray:
        r"""the observations, set by :meth:`set_data` or
        :meth:`set_data_from_file`

        Raises
        ------
        NotDefinedError
            when this property has not been defined by methods above
        """
        if hasattr(self, "_data") and self._data is not None:
            return self._data
        raise NotDefinedError(needs="data")

    @property
    def data_covariance(self) -> np.ndarray:
        """the data covariance matrix, set by :meth:`set_data_covariance`,
        :meth:`set_data` or :meth:`set_data_from_file`.

        Raises
        ------
        NotDefinedError
            when this property has not been defined by methods above
        """
        if hasattr(self, "_data_covariance") and self._data_covariance is not None:
            return self._data_covariance
        raise NotDefinedError(needs="data covariance matrix")

    @property
    def data_covariance_inv(self) -> np.ndarray:
        """the data covariance matrix, set by :meth:`set_data_covariance_inv`,
        :meth:`set_data` or :meth:`set_data_from_file`.

        Raises
        ------
        NotDefinedError
            when this property has not been defined by methods above
        """
        if (
            hasattr(self, "_data_covariance_inv")
            and self._data_covariance_inv is not None
        ):
            return self._data_covariance_inv
        raise NotDefinedError(needs="inverse data covariance matrix")

    @property
    def initial_model(self) -> np.ndarray:
        r"""the initial model, needed for some iterative optimization tools that
        requires a starting point

        Raises
        ------
        NotDefinedError
            when this property has not been defined (by
            :meth:`set_initial_model`)
        """
        if hasattr(self, "_initial_model") and self._initial_model is not None:
            return self._initial_model
        raise NotDefinedError(needs="initial_model")

    @property
    def model_shape(self) -> Union[Tuple, np.ndarray]:
        r"""the model shape

        Raises
        ------
        NotDefinedError
            when this property has not been defined (by either
            :meth:`set_model_shape`,
            :meth:`set_model_shape`, or
            :meth:`set_walkers_starting_pos`)
        """
        if hasattr(self, "_model_shape") and self._model_shape is not None:
            return self._model_shape
        raise NotDefinedError(needs="model_shape")

    @property
    def walkers_starting_pos(self) -> np.ndarray:
        r"""the starting positions for each walker

        Raises
        ------
        NotDefinedError
            when this property has not been defined (by
            :meth:`set_walkers_starting_pos`)
        """
        if (
            hasattr(self, "_walkers_starting_pos")
            and self._walkers_starting_pos is not None
        ):
            return self._walkers_starting_pos
        raise NotDefinedError(needs="walkers' starting positions")

    @property
    def blobs_dtype(self) -> list:
        r"""the name and type for the blobs that
        :meth:`log_posterior_with_blobs` will return

        Raises
        ------
        NotDefinedError
            when this property has not been defined (by either
            :meth:`set_blobs_dtype` or
            :meth:`set_log_posterior_with_blobs`)
        """
        if hasattr(self, "_blobs_dtype") and self._blobs_dtype is not None:
            return self._blobs_dtype
        raise NotDefinedError(needs="blobs name and type")

    @property
    def regularization_factor(self) -> Number:
        r"""regularization factor (lambda) that adjusts weights of the regularization
        term

        Raises
        ------
        NotDefinedError
            when this property has not been defined (by
            :meth:`set_regularization`
        """
        if (
            hasattr(self, "_regularization_factor")
            and self._regularization_factor is not None
        ):
            return self._regularization_factor
        raise NotDefinedError(needs="regularization_factor (lamda)")

    @property
    def bounds(self):
        r"""TODO: document me!

        Raises
        ------
        NotDefinedError
            when this property has not been defined (by
            :meth:`set_bounds`)
        """
        if hasattr(self, "_bounds") and self._bounds is not None:
            return self._bounds
        raise NotDefinedError(needs="bounds")

    @property
    def constraints(self):
        r"""TODO: document me!

        Raises
        ------
        NotDefinedError
            when this property has not been defined (by
            :meth:`set_constraints`)
        """
        if hasattr(self, "_constraints") and self._constraints is not None:
            return self._constraints
        raise NotDefinedError(needs="constraints")

    @property
    def objective_defined(self) -> bool:
        r"""indicates whether :meth:`objective` has been defined"""
        return self._check_defined(self.objective)

    @property
    def log_posterior_defined(self) -> bool:
        r"""indicates whether :meth:`log_posterior` has been defined"""
        return self._check_defined(self.log_posterior)

    @property
    def log_posterior_with_blobs_defined(self) -> bool:
        r"""indicates whether :meth:`log_posterior_with_blobs` has been
        defined
        """
        return self._check_defined(self.log_posterior_with_blobs)

    @property
    def log_prior_defined(self) -> bool:
        r"""indicates whether :meth:`log_prior` has been defined"""
        return self._check_defined(self.log_prior)

    @property
    def log_likelihood_defined(self) -> bool:
        r"""indicates whether :meth:`log_likelihood` has been defined"""
        return self._check_defined(self.log_likelihood)

    @property
    def gradient_defined(self) -> bool:
        r"""indicates whether :meth:`gradient` has been defined"""
        return self._check_defined(self.gradient)

    @property
    def hessian_defined(self) -> bool:
        r"""indicates whether :meth:`hessian` has been defined"""
        return self._check_defined(self.hessian)

    @property
    def hessian_times_vector_defined(self) -> bool:
        r"""indicates whether :meth:`hessian_times_vector` has been defined"""
        return self._check_defined(self.hessian_times_vector, 2)

    @property
    def residual_defined(self) -> bool:
        r"""indicates whether :meth:`residual` has been defined"""
        return self._check_defined(self.residual)

    @property
    def jacobian_defined(self) -> bool:
        r"""indicates whether :meth:`jacobian` has been defined"""
        return self._check_defined(self.jacobian)

    @property
    def jacobian_times_vector_defined(self) -> bool:
        r"""indicates whether :meth:`jacobian_times_vector` has been defined"""
        return self._check_defined(self.jacobian_times_vector, 2)

    @property
    def data_misfit_defined(self) -> bool:
        r"""indicates whether :meth:`data_misfit` has been defined"""
        return self._check_defined(self.data_misfit)

    @property
    def regularization_defined(self) -> bool:
        r"""indicates whether :meth:`regularization` has been defined"""
        return self._check_defined(self.regularization)

    @property
    def regularization_matrix_defined(self) -> bool:
        r"""indicates whether :meth:`regularization_matrix` has been defined"""
        return self._check_defined(self.regularization_matrix)

    @property
    def forward_defined(self) -> bool:
        r"""indicates whether :meth:`forward` has been defined"""
        return self._check_defined(self.forward)

    @property
    def data_defined(self) -> bool:
        r"""indicates whether :meth:`data` has been defined"""
        return self._check_property_defined("data")

    @property
    def data_covariance_defined(self) -> bool:
        r"""indicates whether :meth:`data_covariance` has been defined"""
        return self._check_property_defined("data_covariance")

    @property
    def data_covariance_inv_defined(self) -> bool:
        r"""indicates whether :meth:`data_covariance_inv` has been defined"""
        return self._check_property_defined("data_covariance_inv")

    @property
    def initial_model_defined(self) -> bool:
        r"""indicates whether :meth:`initial_model` has been defined"""
        return self._check_property_defined("initial_model")

    @property
    def model_shape_defined(self) -> bool:
        r"""indicates whether :meth:`model_shape` has been defined"""
        return self._check_property_defined("model_shape")

    @property
    def walkers_starting_pos_defined(self) -> bool:
        r"""indicates whether :meth:`walkers_starting_pos` has been defined"""
        return self._check_property_defined("walkers_starting_pos")

    @property
    def blobs_dtype_defined(self) -> bool:
        r"""indicates whether :meth:`blobs_dtype` has been defined"""
        return self._check_property_defined("blobs_dtype")

    @property
    def regularization_factor_defined(self) -> bool:
        r"""indicates whether :meth:`regularization_factor` has been defined"""
        return self._check_property_defined("regularization_factor")

    @property
    def bounds_defined(self) -> bool:
        r"""indicates whether :meth:`bounds` has been defined"""
        return self._check_property_defined("bounds")

    @property
    def constraints_defined(self) -> bool:
        r"""indicates whether :meth:`constraints` has been defined"""
        return self._check_property_defined("constraints")

    @staticmethod
    def _check_defined(func, args_num=1):
        if func is None:
            return False
        if isinstance(func, _FunctionWrapper):
            return True
        try:
            func(*[np.array([])] * args_num)
        except NotDefinedError:
            return False
        except Exception:  # ok if there're errors caused by dummy input
            return True

    def _check_property_defined(self, prop):
        try:
            getattr(self, prop)
        except NotDefinedError:
            return False
        else:
            return True

    # autogen_table: (tuple of defined things) ->
    #       (name of deduced item, reference to new function to generate)
    @property
    def autogen_table(self):
        return {
            ("data_misfit",): ("objective", _objective_from_dm),
            (
                "data_misfit",
                "regularization",
            ): ("objective", _objective_from_dm_reg),
            (
                "log_likelihood",
                "log_prior",
            ): ("log_posterior_with_blobs", _log_posterior_with_blobs_from_ll_lp),
            ("log_posterior_with_blobs",): (
                "log_posterior",
                _log_posterior_from_lp_with_blobs,
            ),
            ("hessian",): ("hessian_times_vector", _hessian_times_vector_from_hess),
            (
                "forward",
                "data",
            ): ("residual", _residual_from_fwd_dt),
            ("jacobian",): ("jacobian_times_vector", _jacobian_times_vector_from_jcb),
        }

    def _update_autogen(self, updated_item):
        update_dict = {k: v for k, v in self.autogen_table.items() if updated_item in k}
        for need_defined, (to_update, new_func) in update_dict.items():
            if getattr(self, f"{to_update}_defined"):
                to_update_existing = getattr(self, to_update)
                if (
                    isinstance(to_update_existing, _FunctionWrapper)
                    and not to_update_existing.autogen
                ):
                    continue  # already defined by user, don't overwrite
            if all(
                (getattr(self, f"{nm}_defined") for nm in need_defined)
            ):  # can update
                defined_items = list((getattr(self, nm) for nm in need_defined))
                new_func = _FunctionWrapper(
                    to_update, new_func, args=defined_items, autogen=True
                )
                setattr(self, to_update, new_func)
                if to_update == "log_posterior_with_blobs":
                    self.set_blobs_dtype(
                        [("log_likelihood", float), ("log_prior", float)]
                    )
                self._update_autogen(to_update)

    # ---------- Extra inforamtion inferred, not generally used by solvers -----------
    def model_covariance(self, model: np.ndarray):
        C_minv = self.model_covariance_inv(model)
        return np.linalg.inv(C_minv)

    def model_covariance_inv(self, model: np.ndarray):
        G = self.jacobian(model)
        C_dinv = self.data_covariance_inv
        return G.T @ C_dinv @ G

    # ---------- Display related ------------------------------------------------------
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

    def _data_misfit_squared_error(self, model: np.ndarray) -> Number:
        try:
            res = self.residual(model)
            if self.data_covariance_inv_defined:
                if _is_diag(self.data_covariance_inv):
                    weighted_res = np.diag(self.data_covariance_inv) * res
                    return np.sum(np.square(weighted_res))
                else:
                    return res.T @ self.data_covariance_inv @ res
            elif self.data_covariance_defined and _is_diag(self.data_covariance):
                weighted_res = res / np.diag(self.data_covariance)
                return np.sum(np.square(weighted_res))
            else:
                return np.sum(np.square(res))
        except Exception as exception:
            raise InvocationError(func_name="data misfit", autogen=True) from exception

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
                =====================================================================
                Summary for inversion problem: BaseProblem
                =====================================================================
                Model shape: (3,)
                ---------------------------------------------------------------------
                List of functions/properties set by you:
                ['initial_model', 'model_shape']
                ---------------------------------------------------------------------
                List of functions/properties created based on what you have provided:
                ['objective', 'data_misfit']
                ( Note that you did not set regularization )
                ---------------------------------------------------------------------
                List of functions/properties not set by you: 
                (not all of these may be relevant to your inversion workflow)
                ['objective', 'gradient', 'hessian', 'hessian_times_vector', 'residual', 'jacobian', 'jacobian_times_vector', 'data_misfit', 'regularization', 'forward', 'data', 'bounds', 'constraints']

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
        if display_lines:
            print(double_line)
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
            and not self.regularization_defined
        ):
            print("( Note that you did not set regularization )")
        if display_lines:
            print(single_line)
        print(sub_title3)
        print("( not all of these may be relevant to your inversion workflow )")
        print(not_set if not_set else "-- none --")

    def __repr__(self) -> str:
        return f"{self.name}"


# ---------- End of BaseProblem class -------------------------------------------------


# ---------- Auto generated functions -------------------------------------------------
def _objective_from_dm_reg(model, data_misfit, regularization):
    try:
        return data_misfit(model) + regularization(model)
    except Exception as exception:
        raise InvocationError(
            func_name="objective function from data misfit and regularization",
            autogen=True,
        ) from exception


def _objective_from_dm(model, data_misfit):
    try:
        return data_misfit(model)
    except Exception as exception:
        raise InvocationError(
            func_name="objective function from data misfit", autogen=True
        ) from exception


def _log_posterior_with_blobs_from_ll_lp(model, log_likelihood, log_prior):
    try:
        ll = log_likelihood(model)
        lp = log_prior(model)
        return ll + lp, ll, lp
    except Exception as exception:
        raise InvocationError(
            func_name="log posterior function from log likelihood and log prior",
            autogen=True,
        ) from exception


def _log_posterior_from_lp_with_blobs(model, log_posterior_with_blobs):
    try:
        return log_posterior_with_blobs(model)[0]
    except Exception as exception:
        raise InvocationError(
            func_name="log posterior function from log likelihood and log prior",
            autogen=True,
        ) from exception


def _hessian_times_vector_from_hess(model, vector, hessian):
    try:
        return np.squeeze(np.asarray(hessian(model) @ vector))
    except Exception as exception:
        raise InvocationError(
            func_name="hessian_times_vector function from given hessian function",
            autogen=True,
        ) from exception


def _residual_from_fwd_dt(model, forward, data):
    try:
        return forward(model) - data
    except Exception as exception:
        raise InvocationError(
            func_name="residual function from forward and data provided", autogen=True
        ) from exception


def _jacobian_times_vector_from_jcb(model, vector, jacobian):
    try:
        return np.squeeze(np.asarray(jacobian(model) @ vector))
    except Exception as exception:
        raise InvocationError(
            func_name="jacobian_times_vector from given jacobian function", autogen=True
        ) from exception


def _regularization_with_lamda(model, reg_func, lamda):
    return lamda * reg_func(model)


def _regularization_with_lamda_n_matrix(model, reg_func, lamda, reg_matrix_func):
    return lamda * reg_func(reg_matrix_func(model) @ model)


def _matrix_to_func(_, matrix):
    return matrix


def _is_diag(matrix):
    diag_elem = np.diag(matrix).copy()
    np.fill_diagonal(matrix, 0)
    out = (matrix == 0).all()
    np.fill_diagonal(matrix, diag_elem)
    return out


# ---------- function wrapper to help make things pickleable --------------------------
class _FunctionWrapper:
    def __init__(
        self, name, func, args: list = None, kwargs: dict = None, autogen=False
    ):
        if not callable(func):
            raise InvalidOptionError(
                name=f"{name} function",
                invalid_option="not-callable input",
                valid_options="functions that are callable",
            )
        self.name = name
        self.func = func
        self.args = list() if args is None else args
        self.kwargs = dict() if kwargs is None else kwargs
        self.autogen = autogen

    def __call__(self, model, *extra_args):
        try:
            return self.func(model, *extra_args, *self.args, **self.kwargs)
        except Exception as exception:
            if self.autogen:
                raise exception
            else:
                raise InvocationError(
                    func_name=self.name, autogen=self.autogen
                ) from exception

            # import traceback
            # print(f"cofi: Exception while calling your {self.name} function:")
            # print("  params:", model, *extra_args)
            # print("  args:", self.args, len(self.args))
            # print("  kwargs:", self.kwargs)
            # print("  exception:")
            # traceback.print_exc()
            # raise exception
