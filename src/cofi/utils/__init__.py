r"""Utility classes and functions (e.g. to generate regularization terms and more)

The class inheritance of regularization classes:

.. mermaid::

    graph TD;
    BaseRegularization --> LpNormRegularization;
    LpNormRegularization --> QuadraticReg;
    BaseRegularization --> ModelCovariance;
    ModelCovariance --> GaussianPrior;

"""

from ._reg_base import BaseRegularization
from ._reg_lp_norm import LpNormRegularization, QuadraticReg
from ._reg_model_cov import ModelCovariance, GaussianPrior

from ._multiple_runs import InversionPool


__all__ = [
    "BaseRegularization",
    "LpNormRegularization",
    "QuadraticReg",
    "ModelCovariance",
    "GaussianPrior",
    "InversionPool",
]
