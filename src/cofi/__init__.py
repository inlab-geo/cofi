from .base_problem import BaseProblem
from .inversion_options import InversionOptions
from .inversion import Inversion, InversionResult, SamplingResult


try:
    from . import _version

    __version__ = _version.__version__
except ImportError:
    pass


__all__ = [
    "BaseProblem",  # public API, basic usage
    "InversionOptions",  # public API, basic usage
    "Inversion",  # public API, basic usage
    "InversionResult",  # public API, for advanced usage (own solver)
    "SamplingResult",  # public API, for advanced usage (own solver)
]
