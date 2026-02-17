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
from ._lik_base import BaseLikelihood
from ._kernel import SquaredExponentialKernel
from ._reduced_likelihood import ReducedLikelihood
from ._reduced_likelihood_manager import ReducedLikelihoodManager

from ._multiple_runs import InversionPool


__all__ = [
    "BaseRegularization",
    "ReducedLikelihood",
    "ReducedLikelihoodManager",
    "BaseLikelihood",
    "LpNormRegularization",
    "QuadraticReg",
    "ModelCovariance",
    "GaussianPrior",
    "InversionPool",
    "SquaredExponentialKernel",
]
