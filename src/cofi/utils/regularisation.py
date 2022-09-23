from numbers import Number
from abc import abstractmethod, ABCMeta
from typing import Union, Any
import numpy as np
import findiff

from ..exceptions import DimensionMismatchError


REG_TYPES = {
    "damping": 0,
    "flattening": 1,
    "roughening": 1,
    "smoothing": 2,
}


class BaseRegularisation(metaclass=ABCMeta):
    def __init__(self,):
        pass

    @property
    @abstractmethod
    def model_size(self) -> Number:
        raise NotImplementedError

    def __call__(self, model: np.ndarray, *args: Any, **kwds: Any) -> Any:
        r"""a class instance itself can also be called as a function
        
        It works exactly the same as :meth:`reg`.
        """
        return self.reg(model, *args, **kwds)

    @abstractmethod
    def reg(self, model: np.ndarray, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def gradient(self, model: np.ndarray, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def hessian(self, model: np.ndarray, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def __add__(self, other_reg):
        if not isinstance(other_reg, BaseRegularisation):
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__.__name__}' "
                f"and '{other_reg.__class__.__name__}"
            )
        if self.model_size != other_reg.model_size:
            raise DimensionMismatchError(
                entered_name="the second regularisation term",
                entered_dimension=other_reg.model_size,
                expected_source="the first regularisation term",
                expected_dimension=self.model_size,
            )
        tmp_model_size = self.model_size
        tmp_reg = self.reg
        tmp_grad = self.gradient
        tmp_hess = self.hessian
        class NewRegularisation(BaseRegularisation):
            @property
            def model_size(self):
                return tmp_model_size
            def reg(self, model):
                return tmp_reg(model) + other_reg(model)
            def gradient(self, model):
                return tmp_grad(model) + other_reg.gradient(model)
            def hessian(self, model):
                return tmp_hess(model) + other_reg.hessian(model)
        return NewRegularisation()


class QuadraticReg(BaseRegularisation):
    r"""CoFI's utility class to calculate damping, flattening (roughening), and
    smoothing regularisation

    They correspond to the zeroth order, first order and second order Tikhonov
    regularisation approaches respectively.
    
    - If ``reg_type == "damping"``, then

      - :math:`\text{reg}=\text{factor}\times||m-m_0||_2^2`
      - :math:`\frac{\partial\text{reg}}{\partial m}=\text{factor}\times(m-m_0)`
      - :math:`\frac{\partial^2\text{reg}}{\partial m}=\text{factor}\times I`
      - where :math:`m_0` is a reference model that you can specify in ``ref_model`` argument 
        (default to zero)

    - If ``reg_type == "roughening"`` (or equivalently ``"flattening"``),
      then 

      - :math:`\text{reg}=\text{factor}\times||Dm||_2^2`
      - :math:`\frac{\partial\text{reg}}{\partial m}=\text{factor}\times D^TDm`
      - :math:`\frac{\partial^2\text{reg}}{\partial m}=\text{factor}\times D^TD`
      - where :math:`D` matrix helps calculate the first order derivative of :math:`m` and looks like
      - :math:`\begin{pmatrix}-1&1&&&&\\&-1&1&&&\\&&...&...&&\\&&&-1&1&\\&&&&&-1&1\end{pmatrix}`

    - If ``reg_type == "smoothing"``, then

      - :math:`\text{reg}=\text{factor}\times||Dm||_2^2`
      - :math:`\frac{\partial\text{reg}}{\partial m}=\text{factor}\times D^TDm`
      - :math:`\frac{\partial^2\text{reg}}{\partial m}=\text{factor}\times D^TD`
      - where :math:`D` matrix helps calculate the second order derivatives of :math:`m` and looks like
      - :math:`\begin{pmatrix}1&-2&1&&&&\\&1&-2&1&&&\\&&...&...&...&&\\&&&1&-2&1&\\&&&&1&-2&1\end{pmatrix}`

    - If ``reg_type == None``, then we assume you want to use the argument 
      ``byo_matrix``,
      
      - :math:`\text{reg}=\text{factor}\times||Dm||_2^2`
      - :math:`\frac{\partial\text{reg}}{\partial m}=\text{factor}\times D^TDm`
      - :math:`\frac{\partial^2\text{reg}}{\partial m}=\text{factor}\times D^TD`
      - where :math:`D` matrix is ``byo_matrix`` from the arguments (or identity matrix if ``None``)

    Parameters
    ----------
    factor : Number
        the scale for the regularisation term
    model_size : Number
        the number of elements in a inference model
    reg_type : str
        specify what kind of regularisation is to be calculated, by default
        ``"damping"``
    ref_model : np.ndarray
        reference model used only when ``reg_type == "damping"``,
        by default None (if this is None, then reference model is assumed to be zero)
    byo_matrix : np.ndarray
        bring-your-own matrix, activated only when ``reg_type == None``
    
    Raises
    ------
    ValueError
        when ...

    Examples
    --------

    Generate a quadratic damping regularisation matrix for model of size 3:

    >>> from cofi.utils import QuadraticReg
    >>> reg = QuadraticReg(factor=1, model_size=3)
    >>> reg(np.array([1,2,3]))
    3.0

    To use together with :class:`cofi.BaseProblem`:

    >>> from cofi import BaseProblem
    >>> from cofi.utils import QuadraticReg
    >>> reg = QuadraticReg(factor=1, model_size=3)
    >>> my_problem = BaseProblem()
    >>> my_problem.set_regularisation(reg)

    You may also combine two regularisation terms:

    >>> from cofi import BaseProblem
    >>> from cofi.utils import QuadraticReg
    >>> reg1 = QuadraticReg(factor=1, model_size=3, reg_type="damping")
    >>> reg2 = QuadraticReg(factor=2, model_size=3, reg_type="smoothing")
    >>> my_problem = BaseProblem()
    >>> my_problem.set_regularisation(reg1 + reg2)
    """
    def __init__(
        self, 
        factor: Number, 
        model_size: Union[Number, np.ndarray], 
        reg_type: str="damping",
        ref_model: np.ndarray=None,
        byo_matrix: np.ndarray=None,
    ):
        super().__init__()
        self._factor = self._validate_factor(factor)
        self._model_size = model_size
        self._reg_type = self._validate_reg_type(reg_type)
        self._ref_model = ref_model
        self._byo_matrix = byo_matrix
        self._generate_matrix()
    
    @property
    def model_size(self) -> Number:
        if self._reg_type == "damping":
            return self._model_size
        else:
            return self._model_size[0] * self._model_size[1]

    @property
    def matrix(self) -> np.ndarray:
        return self._D

    def reg(self, model: np.ndarray) -> Number:
        flat_m = self._validate_model(model)
        if self._reg_type == "damping":
            if self._ref_model is None:
                return self._factor * flat_m.T @ flat_m
            diff_ref = flat_m - self._ref_model
            return self._factor * diff_ref.T @ diff_ref
        else:
            flat_m = self._validate_model(model)
            weighted_m = self.matrix @ flat_m
            return self._factor * weighted_m.T @ weighted_m

    def gradient(self, model: np.ndarray) -> np.ndarray:
        flat_m = self._validate_model(model)
        if self._reg_type == "damping":
            if self._ref_model is None:
                return self._factor * flat_m
            return self._factor * (flat_m - self._ref_model)
        else:
            return self._factor * self.matrix.T @ self.matrix @ flat_m
    
    def hessian(self, model: np.ndarray) -> np.ndarray:
        if self._reg_type == "damping":
            return self._factor * np.eye(self._model_size)
        else:
            return self._factor * self.matrix.T @ self.matrix

    @staticmethod
    def _validate_factor(factor):
        if not isinstance(factor, Number):
            raise ValueError("the regularisation factor must be a number")
        elif factor < 0:
            raise ValueError("the regularisation factor must be non-negative")
        return factor

    @staticmethod
    def _validate_reg_type(reg_type):
        if reg_type is not None and (not isinstance(reg_type, str) or \
            reg_type not in REG_TYPES):
            raise ValueError(
                "Please choose a valid regularisation type. `damping`, "
                    "`flattening` and `smoothing` are supported."
            )
        return reg_type

    def _generate_matrix(self):
        if self._reg_type == "damping":
            if not isinstance(self._model_size, Number):
                raise ValueError(
                    "model_size must be a number when damping is selected"
                )
            self._D = np.identity(self._model_size)
        elif self._reg_type in REG_TYPES:       # 1st/2nd order Tikhonov
            if np.size(self._model_size) == 2 and np.ndim(self._model_size) == 1:
                nx = self._model_size[0]
                ny = self._model_size[1]
                order = REG_TYPES[self._reg_type]
                if nx < order+2 or ny < order+2:
                    raise ValueError(
                        f"the model_size needs to be at least (>={order+2}, >={order+2}) "
                        f"for regularisation type '{self._reg_type}'"
                    )
                d_dx2 = findiff.FinDiff(0, 1, order)            # x direction
                d_dy2 = findiff.FinDiff(1, 1, order)            # y direction
                matx = d_dx2.matrix((nx, ny))                   # scipy sparse matrix
                maty = d_dy2.matrix((nx, ny))                   # scipy sparse matrix
                self._D = np.vstack((matx.toarray(), maty.toarray()))   # combine above
            else:
                raise NotImplementedError(
                    "only 2D derivative operators implemented"
                )
        elif self._reg_type is None:
            if not isinstance(self._model_size, Number):
                raise ValueError(
                    "please provide a number for 'model_size' when bringing your "
                    "own weighting matrix"
                )
            if self._byo_matrix is None:
                self._D = np.identity(self._model_size)
            else:
                self._D = self._byo_matrix
            if len(self._D.shape) != 2:
                raise ValueError(
                    "the bring-your-own regularisation matrix must be 2-dimensional"
                )
            elif self._D.shape[1] != self._model_size:
                raise ValueError(
                    "the bring-your-own regularisation matrix must be in shape (_, M) "
                    "where M is the model size"
                )

    def _validate_model(self, model):
        flat_m = np.ravel(model)
        if self._reg_type == "damping":
            if flat_m.size != self._model_size:
                raise DimensionMismatchError(
                    entered_name="model",
                    entered_dimension=model.shape,
                    expected_source="model",
                    expected_dimension=self._model_size
                )
        else:
            if flat_m.shape[0] != self._model_size[0] * self._model_size[1]:
                raise DimensionMismatchError(
                    entered_name="model",
                    entered_dimension=model.shape,
                    expected_source="model",
                    expected_dimension=self._model_size
                )
        return flat_m
