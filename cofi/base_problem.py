from numbers import Number
from typing import Callable, Union, Tuple, Sequence

import numpy as np

from .inv_problems import forward_dispatch_table


class BaseProblem:
    """Base class for a inversion problem setup.

    An inversion problem can be defined from several tiers, depending on the
    level of flexibility or control in ways you'd like to evaluate a model, as
    well as the solving approaches you'd like to apply on the problem.

    To define an inversion problem that is intended to be solved by optimisation,
    the following combinations are to be supplied:
    - objective + gradient (optional) + hessian (optional)
    - data_misfit + regularisation
    - data + forward + specified in-built data_misfit + regularisation
    - etc.

    To define an inversion problem that is intended to be solved by sampling,
    one of the following combinations are to be supplied:
    - prior + likelihood + proposal_dist (optional)
    - posterior + proposal_dist (optional)
    - etc. (WIP)

    At any point of defining your inversion problem, the `BaseProblem().suggest_solvers()`
    method can be used to get a list of solvers that can be applied on your problem
    based on what have been supplied so far.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def objective(self, model: np.ndarray) -> Number:
        if self.data_misfit_defined and self.regularisation_defined:
            return self.data_misfit(model) + self.regularisation(model)
        raise NotImplementedError(
            "`objective` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def gradient(self, model: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "`gradient` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def hessian(self, model: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "`hessian` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def hessian_times_vector(self, model: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if self.hessian_defined:
            return self.hessian(model) @ vector
        raise NotImplementedError(
            "`hessian_times_vector` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def residual(self, model: np.ndarray) -> np.ndarray:
        if self.forward_defined and self.dataset_defined:
            return self.forward(model) - self.data_y
        raise NotImplementedError(
            "`residual` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def jacobian(self, model: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "`jacobian` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def jacobian_times_vector(self, model: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if self.jacobian_defined:
            return self.jacobian(model) @ vector
        raise NotImplementedError(
            "`jacobian_times_vector` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def data_misfit(self, model: np.ndarray) -> Number:
        raise NotImplementedError(
            "`data_misfit` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def regularisation(self, model: np.ndarray) -> Number:
        raise NotImplementedError(
            "`regularisation` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def forward(self, model: np.ndarray) -> Union[np.ndarray, Number]:
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
        self.objective = obj_func

    def set_gradient(self, grad_func: Callable[[np.ndarray], np.ndarray]):
        self.gradient = grad_func

    def set_hessian(self, hess_func: Union[Callable[[np.ndarray], np.ndarray], np.ndarray]):
        if isinstance(hess_func, np.ndarray):
            self.hessian = lambda _: hess_func
        else:
            self.hessian = hess_func

    def set_hessian_times_vector(self, hess_vec_func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.hessian_times_vector = hess_vec_func

    def set_residual(self, res_func: Callable[[np.ndarray], np.ndarray]):
        self.residual = res_func

    def set_jacobian(self, jac_func: Union[Callable[[np.ndarray], np.ndarray], np.ndarray]):
        if isinstance(jac_func, np.ndarray):
            self.jacobian = lambda _: jac_func
        else:
            self.jacobian = jac_func

    def set_jacobian_times_vector(self, jac_vec_func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.jacobian_times_vector = jac_vec_func

    def set_data_misfit(self, data_misfit: Union[str, Callable[[np.ndarray], Number]]):
        if isinstance(data_misfit, str):
            # TODO - define a dict on top of this file for available data_misfit methods
            if data_misfit in ["L2", "l2", "euclidean", "L2 norm", "l2 norm", "mse", "MSE"]:
                self.data_misfit = self._data_misfit_L2
            else:   # TODO - other options?
                raise NotImplementedError(
                    "the data misfit method you've specified isn't supported yet, please "
                    "report an issue here: https://github.com/inlab-geo/cofi/issues if you "
                    "find it valuable to support it from our side"
                )
        else:
            self.data_misfit = data_misfit

    def set_regularisation(self, regularisation: Union[str, Callable[[np.ndarray], Number]], factor:Number=0.1):
        if isinstance(regularisation, str):
            # TODO - define a dict on top of this file for available reg methods
            if regularisation in ["L0", "l0", "L0 norm", "l0 norm"]:
                _reg = lambda x: np.linalg.norm(x, ord=0)
            elif regularisation in ["L1", "l1", "manhattan", "taxicab", "L1 norm", "l1 norm"]:
                _reg = lambda x: np.linalg.norm(x, ord=1)
            elif regularisation in ["L2", "l2", "euclidean", "L2 norm", "l2 norm"]:
                _reg = lambda x: np.linalg.norm(x, ord=2)
            else:
                raise NotImplementedError(
                    "the regularisation method you've specified isn't supported yet, please "
                    "report an issue here: https://github.com/inlab-geo/cofi/issues if you "
                    "find it valuable to support it from our side"
                )
        else:
            _reg = regularisation
        self.regularisation = lambda m: _reg(m) * factor

    def set_forward(self, forward: Union[str, Callable[[np.ndarray], Union[np.ndarray,Number]]]):
        if isinstance(forward, str):
            # TODO - add available forward operator here, maybe a dict defined on top of this file is nice
            if forward not in forward_dispatch_table:
                raise NotImplementedError(
                    "the forward operator you've specified is not implemented by cofi, please "
                    "supply a full function or check our documentation for available forwrad problems"
                )
            elif not self.dataset_defined:
                raise NotImplementedError("dataset is not provided before setting your forward function")
            else:
                self.forward = forward_dispatch_table[forward](self.data_x)
        else:
            self.forward = forward

    def set_dataset(self, data_x:np.ndarray, data_y:np.ndarray):
        self._data_x = data_x
        self._data_y = data_y

    def set_dataset_from_file(self, file_path, obs_idx=-1):
        """Set the dataset for this problem from a give file path. This function
        uses :func:`numpy.loadtxt` to load dataset file.

        :param file_path: a relative/absolute file path for the dataset
        :type file_path: str
        :param obs_idx: _description_, defaults to -1
        :type obs_idx: int, optional
        """
        delimiter = None    # try to detect what delimiter is used
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
        self.set_dataset(np.delete(data,obs_idx,1), data[:,obs_idx])

    def set_initial_model(self, init_model: np.ndarray):
        self._initial_model = init_model
        self._model_shape = init_model.shape if hasattr(init_model, "shape") else (1,)

    def set_model_shape(self, model_shape: Tuple):
        if self.initial_model_defined and self._model_shape != model_shape:
            try:
                np.reshape(self.initial_model, model_shape)
            except ValueError as e:
                raise ValueError(
                    f"The model_shape you've provided {model_shape} doesn't match the "
                    f"initial_model you set which has the shape: {self.initial_model.shape}"
                ) from e
        self._model_shape = model_shape

    def set_bounds(self, bounds: Sequence[Tuple[Number,Number]]):
        self._bounds = bounds

    def set_constraints(self, constraints):
        # TODO - what's the type of this? (ref: scipy has Constraint class)
        self._constraints = constraints

    def defined_components(self) -> set:
        _to_check = [
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
        return {func_name for func_name in _to_check if getattr(self, f"{func_name}_defined")}

    def suggest_solvers(self) -> list:
        # TODO - use self.defined_components() to suggest solvers
        raise NotImplementedError

    @property
    def data_x(self):
        if hasattr(self, "_data_x"): return self._data_x
        raise NameError(
            "data has not been set, please use either `set_dataset()` or "
            "`set_dataset_from_file()` to add dataset to the problem setup"
        )

    @property
    def data_y(self):
        if hasattr(self, "_data_y"): return self._data_y
        raise NameError(
            "data has not been set, please use either `set_dataset()` or "
            "`set_dataset_from_file()` to add dataset to the problem setup"
        )

    @property
    def initial_model(self):
        if hasattr(self, "_initial_model"): return self._initial_model
        raise NameError(
            "initial model has not been set, please use `set_initial_model()`"
            " to add to the problem setup"
        )

    @property
    def model_shape(self):
        if hasattr(self, "_model_shape"): return self._model_shape
        raise NameError(
            "model shape has not been set, please use either `set_initial_model()`"
            " or `set_model_shape() to add to the problem setup"
        )

    @property
    def bounds(self):
        if hasattr(self, "_boundsj"): return self._bounds
        raise NameError(
            "bounds have not been set, please use `set_bounds()` to add to the "
            "problem setup"
        )

    @property
    def constraints(self):
        if hasattr(self, "_constraints"): return self._constraints
        raise NameError(
            "constraints have not been set, please use `set_constraints()` to add "
            "to the problem setup"
        )

    @property
    def objective_defined(self):
        return self.check_defined(self.objective)

    @property
    def gradient_defined(self):
        return self.check_defined(self.gradient)

    @property
    def hessian_defined(self):
        return self.check_defined(self.hessian)
    
    @property
    def hessian_times_vector_defined(self):
        return self.check_defined(self.hessian_times_vector, 2)

    @property
    def residual_defined(self):
        return self.check_defined(self.residual)

    @property
    def jacobian_defined(self):
        return self.check_defined(self.jacobian)

    @property
    def jacobian_times_vector_defined(self):
        return self.check_defined(self.jacobian_times_vector, 2)

    @property
    def data_misfit_defined(self):
        return self.check_defined(self.data_misfit)

    @property
    def regularisation_defined(self):
        return self.check_defined(self.regularisation)

    @property
    def forward_defined(self):
        return self.check_defined(self.forward)

    @property
    def dataset_defined(self):
        try:
            self.data_x
            self.data_y
        except NameError:
            return False
        else:
            return True

    @property
    def initial_model_defined(self):
        try:
            self.initial_model
        except NameError:
            return False
        else:
            return True

    @property
    def model_shape_defined(self):
        try:
            self.model_shape
        except NameError:
            return False
        else:
            return True

    @property
    def bounds_defined(self):
        try:
            self.bounds
        except NameError:
            return False
        else:
            return True

    @property
    def constraints_defined(self):
        try:
            self.constraints
        except NameError:
            return False
        else:
            return True

    @staticmethod
    def check_defined(func, args_num=1):
        try:
            func(*[np.array([])]*args_num)
        except NotImplementedError:
            return False
        except:  # it's ok if there're errors caused by dummy input argument np.array([])
            return True
        else:
            return True

    @property
    def name(self):
        return self._name if hasattr(self, "_name") else self.__class__.__name__

    @name.setter
    def name(self, problem_name):
        self._name = problem_name

    def _data_misfit_L2(self, model: np.ndarray) -> Number:
        if self.residual_defined:
            return np.linalg.norm(self.residual(model)) / self.data_x.shape[0]
        else:
            raise NotImplementedError("insufficient information provided to calculate mean squared error")

    def summary(self):
        self._summary()

    def _summary(self, display_lines=True):
        # inspiration from keras: https://keras.io/examples/vision/mnist_convnet/
        title = f"Summary for inversion problem: {self.name}"
        sub_title = "List of functions / properties defined:"
        display_width = max(len(title), len(sub_title))
        double_line = "=" * display_width
        single_line = "-" * display_width
        print(title)
        if display_lines: print(double_line)
        model_shape = self.model_shape if self.model_shape_defined else "Unknown"
        print(f"Model shape: {model_shape}")
        if display_lines: print(single_line)
        print(sub_title)
        print(self.defined_components())

    def __repr__(self) -> str:
        return f"{self.name}"
