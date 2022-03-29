from numbers import Number
from typing import Callable, Union

import numpy as np


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
        # TODO - this can be calculated once data misfit + regularisation are defined
        raise NotImplementedError(
            "`objective` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def gradient(self, model: np.ndarray) -> Number:
        raise NotImplementedError(
            "`gradient` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def hessian(self, model: np.ndarray) -> Number:
        raise NotImplementedError(
            "`hessian` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def residual(self, model: np.ndarray) -> Number:
        # TODO - this can be calculated once forward + data are supplied
        raise NotImplementedError(
            "`residual` is required in the solving approach but you haven't"
            " implemented or added it to the problem setup"
        )

    def jacobian(self, model: np.ndarray) -> Number:
        raise NotImplementedError(
            "`jacobian` is required in the solving approach but you haven't"
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
    # - add checking to self.defined_list
    # - add tests in tests/test_base_problem.py ("test_non_set", etc.)
    def set_objective(self, obj_func: Callable[[np.ndarray], Number]):
        self.objective = obj_func

    def set_gradient(self, grad_func: Callable[[np.ndarray], Number]):
        self.gradient = grad_func

    def set_hessian(self, hess_func: Callable[[np.ndarray], Number]):
        self.hessian = hess_func

    def set_residual(self, res_func: Callable[[np.ndarray], Number]):
        self.residual = res_func

    def set_jacobian(self, jac_func: Callable[[np.ndarray], Number]):
        self.jacobian = jac_func

    def set_data_misfit(self, data_misfit: Union[str, Callable[[np.ndarray], Number]]):
        if isinstance(data_misfit, str):
            # TODO - define a dict on top of this file for available data_misfit methods
            if data_misfit in ["L2", "l2", "euclidean", "L2 norm", "l2 norm"]:
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
                _reg = self._regularisation_L0
            elif regularisation in ["L1", "l1", "manhattan", "taxicab", "L1 norm", "l1 norm"]:
                _reg = self._regularisation_L1
            elif regularisation in ["L2", "l2", "euclidean", "L2 norm", "l2 norm"]:
                _reg = self._regularisation_L2
            else:   # TODO - other options?
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
            raise NotImplementedError(
                "the forward operator you've specified is not implemented by cofi, please "
                "supply a full function or check our documentation for available forwrad problems"
            )
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
        with open(file_path) as f:
            first_line = f.readline()
            if "," in first_line:
                delimiter = ","
        data = np.loadtxt(file_path, delimiter=delimiter)
        self.set_dataset(np.delete(data,obs_idx,1), data[:,obs_idx])

    def defined_components(self) -> list:
        # TODO - return a list of functions that are defined
        raise NotImplementedError

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
    def objective_defined(self):
        return self.check_defined(self.objective)

    @property
    def gradient_defined(self):
        return self.check_defined(self.gradient)

    @property
    def hessian_defined(self):
        return self.check_defined(self.hessian)

    @property
    def residual_defined(self):
        return self.check_defined(self.residual)

    @property
    def jacobian_defined(self):
        return self.check_defined(self.jacobian)

    @property
    def data_misfit_defined(self):
        return self.check_defined(self.data_misfit)

    @property
    def regularisation_defined(self):
        return self.check_defined(self.regularisation)

    @property
    def dataset_defined(self):
        try:
            self.data_x
            self.data_y
        except NameError:
            return False
        else:
            return True

    @staticmethod
    def check_defined(func):
        try:
            func(np.array([]))
        except NotImplementedError:
            return False
        except:  # it's ok if there're errors caused by dummy input argument np.array([])
            return True
        else:
            return True

    def defined_list(self) -> list:
        _to_check = [
            "objective",
            "gradient",
            "hessian",
            "residual",
            "jacobian",
            "data_misfit",
            "regularisation",
            "dataset",
        ]
        return [func_name for func_name in _to_check if getattr(self, f"{func_name}_defined")]

    def _data_misfit_L2(self,  model: np.ndarray) -> Number:
        # TODO
        raise NotImplementedError

    def _regularisation_L0(self, model: np.ndarray) -> Number:
        # TODO
        raise NotImplementedError

    def _regularisation_L1(self, model: np.ndarray) -> Number:
        # TODO
        raise NotImplementedError

    def _regularisation_L2(self, model: np.ndarray) -> Number:
        # TODO
        raise NotImplementedError

    def summary(self):
        # TODO - print detailed information, including what have been defined and data shape
        # inspiration from keras: https://keras.io/examples/vision/mnist_convnet/
        raise NotImplementedError

    def __repr__(self) -> str:
        # TODO - make a list of what functions are defined so far
        return f"{self.__class__.__name__} with the following information defined: []"
