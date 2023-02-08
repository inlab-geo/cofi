import sys as _sys

from ._base_problem import BaseProblem
from ._inversion_options import InversionOptions
from ._inversion import Inversion, InversionResult, SamplingResult

from . import utils
from . import tools

from ._version import __version__


__all__ = [
    "BaseProblem",  # public API, basic usage
    "InversionOptions",  # public API, basic usage
    "Inversion",  # public API, basic usage
    "InversionResult",  # public API, for advanced usage (own solver)
    "SamplingResult",  # public API, for advanced usage (own solver)
]


# Set default logging handler to avoid "No handler found" warnings.
import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# alias for deprecated API
_sys.modules["cofi.solvers"] = tools
