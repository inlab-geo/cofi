from numbers import Number
from abc import abstractmethod, ABCMeta
from typing import Union, Any
import numpy as np
import findiff


REG_TYPES = {
    "damping": 0,
    "flattening": 1,
    "roughening": 1,
    "smoothing": 2,
}


class BaseRegularisation(metaclass=ABCMeta):
    def __init__(self,):
        pass

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
    
    @staticmethod
    def _validate_factor(factor):
        if not isinstance(factor, Number):
            raise ValueError("the regularisation factor must be a number")
        elif factor < 0:
            raise ValueError("the regularisation factor must be non-negative")
        return factor

    @staticmethod
    def _validate_reg_type(reg_type):
        if reg_type is not None and reg_type not in REG_TYPES:
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

    @property
    def matrix(self):
        return self._D

    def reg(self, model: np.ndarray) -> Number:
        if self._reg_type == "damping":
            if self._ref_model is None:
                return self._factor * np.linalg.norm(model)
            diff_ref = model - self._ref_model
            return self._factor * diff_ref.T @ diff_ref
        else:
            flat_m = np.ravel(model)
            weighted_m = self.matrix @ flat_m
            return self._factor * weighted_m.T @ weighted_m

    def gradient(self, model: np.ndarray):
        if self._reg_type == "damping":
            if self._ref_model is None:
                return self._factor * model
            return self._factor * (model - self._ref_model)
        else:
            flat_m = np.ravel(model)
            return self._factor * self.matrix.T @ self.matrix @ flat_m
    
    def hessian(self, model: np.ndarray):
        if self._reg_type == "damping":
            return self._factor * np.eye(self._model_size)
        else:
            return self._factor * self.matrix.T @ self.matrix

    def __add__(self, other):
        # TODO implement me
        # REMEMBER to check whether model_size match
        raise NotImplementedError
