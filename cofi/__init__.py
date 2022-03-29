from .base_problem import BaseProblem
from .inversion_options import InversionOptions
from .runner import InversionRunner, InversionResult

from . import inv_problems
from . import solvers


try:
    from . import _version

    __version__ = _version.__version__
except ImportError:
    pass


__all__ = [
    "BaseProblem",          # public API, basic usage
    "InversionOptions",     # public API, basic usage
    "InversionRunner",      # public API, basic usage
    "InversionResult",      # public API, for advanced usage (own solver)
]
