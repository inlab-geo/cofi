from typing import Union
from numbers import Number
import numpy as np
from scipy import sparse

from ._reg_base import BaseRegularization
from .._exceptions import DimensionMismatchError


REG_TYPES = {
    "damping": 0,
    "flattening": 1,
    "roughening": 1,
    "smoothing": 2,
}


class LpNormRegularization(BaseRegularization):
    r"""CoFI's utility class to calculate Lp-norm regularization, given the p value
    (default to 2), an optional weighting matrix and an optional reference value

    :math:`L(p, W, m_0) = ||W(m-m_0)||_p^p = \sum_i |W(m-m_0)_i|^p`

    With element-wise gradient of:

    :math:`\sum_i p|(W(m-m_0))_i|^{p-1}sign((W(m-m_0))_i)W_{ij}`

    And element-wise hessian of:

    :math:`\sum_i p(p-1)|(W(m-m_0))_i|^{p-2}W_{ij}W_{ik}`

    Where :math:`W` is a weighting matrix either generated given a specified type
    (e.g. :code:`weighting_matrix="smoothing"`), or a bring-your-own matrix
    (e.g. :code:`weighting_matrix=my_matrix`). This weighting matrix is by default
    in sparse type :class:`scipy.sparse.csr_matrix`.

    The weighting matrix (if not bring-your-own) can be generated provided with an
    option from {:code:`"damping"`, :code:`"flattening"`, :code:`"smoothing"`}.

    - If ``weighting_matrix == "damping"``, then

      .. toggle::

        - :attr:`matrix` is the identity matrix of size :math:`(M,M)`, where
          :math:`M` is the number of model parameters

    - If ``weighting_matrix == "roughening"`` (or equivalently ``"flattening"``),
      then

      .. toggle::

        - :attr:`matrix` is :math:`W` that we generate based on the model shape you've
          provided. We use the Python package :mod:`findiff` to generate it.
          "By default, findiff uses finite difference schemes with second
          order accuracy in the grid spacing." (See `findiff documentation on
          Derivatives <https://findiff.readthedocs.io/en/latest/source/getstarted.html#accuracy-control>`_
          for more details). For 1D problems, it looks like

          :math:`\begin{pmatrix}-1.5&2&-0.5&&&\\-0.5&&0.5&&&&&\\&-0.5&&0.5&&&&\\&&...&&...&&&\\&&&-0.5&&0.5&\\&&&&-0.5&&0.5\\&&&&0.5&-2&1.5\end{pmatrix}`

          While for higher dimension problems, by default it's a flattened version of
          an N-D array. The actual ordering of model parameters in higher dimensions
          is controlled by :class:`findiff.operators.FinDiff`.

    - If ``reg_type == "smoothing"``, then

      .. toggle::

        - :attr:`matrix` is :math:`W` that we generate based on the model shape you've
          provided. We use the Python package :mod:`findiff` to generate it.
          "By default, findiff uses finite difference schemes with second
          order accuracy in the grid spacing." (See `findiff documentation on
          Derivatives <https://findiff.readthedocs.io/en/latest/source/getstarted.html#accuracy-control>`_
          for more details). For 1D problems, it looks like

          :math:`\begin{pmatrix}2&-5&4&-1&&&\\1&-2&1&&&&\\&1&-2&1&&&\\&&...&...&...&&\\&&&1&-2&1&\\&&&&1&-2&1\\&&&-1&4&-5&2\end{pmatrix}`

          .. :math:`\begin{pmatrix}1&-2&1&&&&\\&1&-2&1&&&\\&&...&...&...&&\\&&&1&-2&1&\\&&&&1&-2&1\end{pmatrix}`

          While for higher dimension problems, by default it's a flattened version of
          an N-D array. The actual ordering of model parameters in higher dimensions
          is controlled by :class:`findiff.operators.FinDiff`.

    Parameters
    ----------
    p : Number
        order value (p in the formula above), default to 2
    weighting_matrix: str or np.ndarray
        regularization type (one of {:code:`"damping"`, :code:`"flattening"`
        :code:`"smoothing"`}), or a bring-your-own weighting matrix, default to
        "damping" (identity matrix for weighting)
    model_shape: tuple
        shape of the model, must be supplied if the :code:`reference_model` is not
        given
    reference_model: np.ndarray
        :math:`m_0` in the formula above

    Raises
    ------
    ValueError
        if neither :code:`model_size` nor :code:`reference_model` is given
    DimensionMismatchError
        if both :code:`model_size` and :code:`reference_model` are given but they don't
        match in dimension

    Examples
    --------

    Generate an L1 norm damping regularization for models of size 3:

    >>> from cofi.utils import LpNormRegularization
    >>> my_reg = LpNormRegularization(p=1, model_shape=(3,))
    >>> my_reg(np.array([0,2,1]))
    3.0

    To use togethter with :class:`cofi.BaseProblem`:

    >>> from cofi import BaseProblem
    >>> my_problem.set_regularization(my_reg)
    """

    def __init__(
        self,
        p: Number = 2,
        weighting_matrix: Union[str, np.ndarray] = "damping",
        model_shape: tuple = None,
        reference_model: np.ndarray = None,
    ):
        self._order = self._validate_p(p)
        self._weighting_matrix = weighting_matrix
        self._model_shape = self._validate_shape(model_shape, reference_model)
        self._reference_model = reference_model
        self._generate_weighting_matrix()

    def reg(self, model: np.ndarray) -> Number:
        flat_m = self._validate_model(model)
        diff_m = self._model_diff_to_ref(flat_m)
        weighted_diff_m = self._weighting_matrix @ diff_m
        return self._lp_norm(weighted_diff_m)

    def gradient(self, model: np.ndarray) -> np.ndarray:
        flat_m = self._validate_model(model)
        diff_m = self._model_diff_to_ref(flat_m)
        weighted_diff_m = self._weighting_matrix @ diff_m
        grad_lp_norm = self._lp_norm_gradient(weighted_diff_m)
        return self.matrix.T @ grad_lp_norm

    def hessian(self, model: np.ndarray) -> np.ndarray:
        W = self._weighting_matrix
        flat_m = self._validate_model(model)
        diff_m = self._model_diff_to_ref(flat_m)
        weighted_diff_m = W @ diff_m
        hess_lp_norm = self._lp_norm_hessian(weighted_diff_m)
        return W.T @ np.diag(hess_lp_norm) @ W

    @property
    def model_shape(self) -> tuple:
        return self._model_shape

    @property
    def matrix(self) -> sparse.csr_matrix:
        """the regularization matrix

        This is either an identity matrix, or first/second order difference matrix
        (generated by Python package ``findiff``), or a custom matrix brought on your
        own.
        """
        return self._weighting_matrix

    def _generate_weighting_matrix(self):
        import findiff

        if (
            isinstance(self._weighting_matrix, str)
            and self._weighting_matrix in REG_TYPES
        ) or self._weighting_matrix is None:
            _reg_type = self._weighting_matrix
            if _reg_type == "damping" or _reg_type is None:  # 0th order difference
                self._weighting_matrix = sparse.identity(self.model_size, format="csr")
            elif _reg_type in REG_TYPES:  # 1st/2nd order difference
                if np.size(self.model_shape) == 1:  # 1D model
                    order = REG_TYPES[_reg_type]
                    if self.model_size < order + 2:
                        raise ValueError(
                            f"the model_size needs to be at least >={order+2} "
                            f"for regularization type '{_reg_type}'"
                        )
                    d_dx = findiff.FinDiff(0, 1, order)
                    self._weighting_matrix = d_dx.matrix((self.model_size,))
                elif (
                    np.size(self.model_shape) == 2 and np.ndim(self.model_shape) == 1
                ):  # 2D model
                    nx = self.model_shape[0]
                    ny = self.model_shape[1]
                    order = REG_TYPES[_reg_type]
                    if nx < order + 2 or ny < order + 2:
                        raise ValueError(
                            f"the model_size needs to be at least (>={order+2},"
                            f" >={order+2}) for regularization type '{_reg_type}'"
                        )
                    d_dx = findiff.FinDiff(0, 1, order)  # x direction
                    d_dy = findiff.FinDiff(1, 1, order)  # y direction
                    matx = d_dx.matrix((nx, ny))  # scipy sparse matrix
                    maty = d_dy.matrix((nx, ny))  # scipy sparse matrix
                    self._weighting_matrix = np.vstack(
                        (matx.toarray(), maty.toarray())
                    )  # combine above
                else:
                    raise NotImplementedError(
                        "only 1D and 2D derivative operators implemented"
                    )
        elif is_matrix_like(self._weighting_matrix):  # byo matrix
            if len(self._weighting_matrix.shape) != 2:
                raise ValueError(
                    "the bring-your-own regularization matrix must be 2-dimensional"
                )
            elif self._weighting_matrix.shape[1] != self.model_size:
                raise ValueError(
                    "the bring-your-own regularization matrix must be in shape (_, M) "
                    "where M is the model size"
                )
            self._weighting_matrix = sparse.csr_matrix(self._weighting_matrix)
        else:
            raise ValueError(
                "please specify the weighting matrix either via a string among "
                "\{`damping`, `flattening`, `smoothing`\}, or bringing your own matrix"
            )

    @staticmethod
    def _validate_p(p):
        if not isinstance(p, Number):
            raise ValueError(
                f"number expected for argument `p` but got {p} of type {type(p)}"
            )
        elif p <= 0:
            raise ValueError(f"positive number expected for argument `p` but got {p}")
        return p

    @staticmethod
    def _validate_shape(model_shape, reference_model):
        if model_shape is None and reference_model is None:
            raise ValueError("please provide the model shape")
        elif model_shape is None and reference_model is not None:
            return reference_model.shape
        elif model_shape is not None and reference_model is None:
            if not isinstance(model_shape, tuple):
                raise TypeError(
                    "expected model shape in tuple (e.g. `(100,)`) but got "
                    f"{model_shape} instead"
                )
            return model_shape
        else:
            if reference_model.shape != model_shape:
                try:
                    np.reshape(reference_model, model_shape)
                except:
                    raise DimensionMismatchError(
                        entered_dimension=reference_model.shape,
                        entered_name="reference_model",
                        expected_dimension=model_shape,
                        expected_source="model_shape",
                    )
            return model_shape

    def _validate_model(self, model):
        flat_m = np.ravel(model)
        if flat_m.size != self.model_size:
            raise DimensionMismatchError(
                entered_name="model",
                entered_dimension=model.shape,
                expected_source="model_size",
                expected_dimension=self.model_size,
            )
        return flat_m

    def _model_diff_to_ref(self, model):
        if self._reference_model is None:
            return model
        else:
            return model - np.ravel(self._reference_model)

    def _lp_norm(self, mat):
        return np.sum(np.abs(mat) ** self._order)

    def _lp_norm_gradient(self, mat):
        return self._order * np.abs(mat) ** (self._order - 1) * np.sign(mat)

    def _lp_norm_hessian(self, mat):
        p = self._order
        return p * (p - 1) * np.abs(mat) ** (p - 2)


class QuadraticReg(LpNormRegularization):
    r"""CoFI's utility class to calculate weighted L2 norm regularization with an
    optional reference model

    :math:`L(W, m_0) = ||D(m-m_0)||_2^2`

    With gradient of:

    :math:`2\times W(m-m_0)`

    And hessian of:

    :math:`2\times W.T W`

    Where :math:`W` is a weighting matrix either generated given a specified type
    (e.g. :code:`weighting_matrix="smoothing"`), or a bring-your-own matrix
    (e.g. :code:`weighting_matrix=my_matrix`). This weighting matrix is by default
    in sparse type :class:`scipy.sparse.csr_matrix`.

    This class is a special case of :class:`LpNormRegularization` with :code:`p=2`.

    Parameters
    ----------
    weighting_matrix: str or np.ndarray
        regularization type (one of {:code:`"damping"`, :code:`"flattening"`
        :code:`"smoothing"`}), or a bring-your-own weighting matrix, default to
        "damping" (identity matrix for weighting)
    model_shape : tuple
        shape of the model, must be supplied if the :code:`reference_model` is not
        given
    reference_model: np.ndarray
        :math:`m_0` in the formula above

    Raises
    ------
    ValueError
        if neither :code:`model_size` nor :code:`reference_model` is given
    DimensionMismatchError
        if both :code:`model_size` and :code:`reference_model` are given but they don't
        match in dimension

    Examples
    --------

    Generate an quadratic smoothing regularization for models of size 3:

    >>> from cofi.utils import QuadraticReg
    >>> my_reg = QuadraticReg(weighting_matrix="smoothing", model_shape=(4,))
    >>> my_reg(np.array([0,2,1,0]))
    53.99999999999999

    To use togethter with :class:`cofi.BaseProblem`:

    >>> from cofi import BaseProblem
    >>> my_problem = BaseProblem()
    >>> my_problem.set_regularization(my_reg)
    """

    def __init__(
        self,
        weighting_matrix: Union[str, np.ndarray] = "damping",
        model_shape: tuple = None,
        reference_model: np.ndarray = None,
    ):
        super().__init__(
            p=2,
            weighting_matrix=weighting_matrix,
            model_shape=model_shape,
            reference_model=reference_model,
        )


matrix_like_classes = [np.ndarray] + [
    getattr(sparse, name) for name in sparse.__all__ if name.endswith("_matrix")
]


def is_matrix_like(obj):
    return any(isinstance(obj, cls) for cls in matrix_like_classes)
