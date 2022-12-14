from pathlib import Path

from .base_problem import BaseProblem
from .inversion_options import InversionOptions
from .inversion import Inversion, InversionResult, SamplingResult

from . import utils
from . import solvers


__version__ = Path(__file__).with_name("VERSION").read_text().strip()

__all__ = [
    "BaseProblem",  # public API, basic usage
    "InversionOptions",  # public API, basic usage
    "Inversion",  # public API, basic usage
    "InversionResult",  # public API, for advanced usage (own solver)
    "SamplingResult",  # public API, for advanced usage (own solver)
]


# Set default logging handler to avoid "No handler found" warnings.
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
