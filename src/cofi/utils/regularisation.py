from numbers import Number
from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import Union, Any
import numpy as np
import findiff


class RegularisationType(Enum):
    r"""Enum type of regularisation term

    This is used as an input for :class:`Regularisation`
    """
    damping = 0
    flattening = 1
    roughening = 1
    smoothing = 2


class BaseRegularisation(metaclass=ABCMeta):
    def __init__(self,):
        pass

    def __call__(self, model: np.ndarray, *args: Any, **kwds: Any) -> Any:
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
    
    If ``reg_type == RegularisationType.damping``, then
    :math:`\text{reg}=\text{factor}\times||m-m_0||_2^2`, where :math:`m_0` is a
    reference model that you can specify in ``ref_model`` argument (default to zero)

    If ``reg_type == RegularisationType.roughening`` (or equivalently ``flattening``),
    then :math:`\text{reg}=\text{factor}\times||Dm||_2^2`, where
    :math:`D` gives the first order difference of :math:`m` and looks like
    :math:`\begin{pmatrix}-1&1&&&&\\&-1&1&&&\\&&...&...&&\\&&&-1&1&\\&&&&&-1&1\end{pmatrix}`

    If ``reg_type == RegularisationType.smoothing``, then
    :math:`\text{reg}=\text{factor}\times||Dm||_2^2`, where
    :math:`D` gives the second order difference of :math:`m` and looks like
    :math:`\begin{pmatrix}1&-2&1&&&&\\&1&-2&1&&&\\&&...&...&...&&\\&&&1&-2&1&\\&&&&1&-2&1\end{pmatrix}`

    If ``reg_type == None``, then we assume you have ``byo_matrix`` defined (otherwise 
    a ``ValueError`` is raised),
    :math:`\text{reg}=\text{factor}\times||\text{byo_matrix}m||_2^2`

    Parameters
    ----------
    factor : Number
        the scale for the regularisation term
    model_size : Number
        the number of elements in a inference model
    reg_type : RegularisationType | str | None
        specify what kind of regularisation is to be calculated, by default
        ``RegularisationType.damping``
    ref_model : np.ndarray
        reference model used only when ``reg_type == RegularisationType.damping``,
        by default None (if this is None, then reference model is assumed to be zero)
    byo_matrix : np.ndarray
        bring-your-own matrix, used only when ``reg_type == None``
    
    Raises
    ------
    ValueError
        when ...

    Examples
    --------

    >>> from cofi.utils import QuadraticReg
    >>> reg1 = QuadraticReg(1, 10, )
    """
    def __init__(
        self, 
        factor: Number, 
        model_size: Union[Number, np.ndarray], 
        reg_type: Union[RegularisationType, str, None]="damping",
        ref_model: np.ndarray=None,
        byo_matrix: np.ndarray=None,
    ):
        super().__init__()
        self._factor = factor
        self._model_size = model_size
        self._reg_type = self._validate_reg_type(reg_type)
        self._ref_model = ref_model
        self._byo_matrix = byo_matrix
        self._generate_matrix()
    
    @staticmethod
    def _validate_reg_type(reg_type):
        if reg_type is None or isinstance(reg_type, RegularisationType):
            pass
        elif isinstance(reg_type, str) and hasattr(RegularisationType, reg_type):
            reg_type = getattr(RegularisationType, reg_type)
        else:
            raise ValueError(
                "Please choose a valid regularisation type. `damping`, "
                    "`flattening` and `smoothing` are supported."
            )
        return reg_type

    def _generate_matrix(self):
        if self._reg_type == RegularisationType.damping:
            self._D = np.identity(self._model_size)
        elif self._reg_type == RegularisationType.flattening or \
            self._reg_type == RegularisationType.smoothing:
            raise NotImplementedError
        elif self._reg_type is None:
            self._D = self._byo_matrix or np.identity(self._model_size)

    def reg(self, model: np.ndarray) -> Number:
        if self._reg_type == RegularisationType.damping:
            if self._ref_model is None:
                return self._factor * np.linalg.norm(model)
            return self._factor * np.linalg.norm(model - self._ref_model)
        else:
            raise NotImplementedError

    def gradient(self, model: np.ndarray):
        raise NotImplementedError
    
    def hessian(self, model: np.ndarray):
        raise NotImplementedError
