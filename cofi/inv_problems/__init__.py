from .base_forward import BaseForward

from .polynomial_fwd import PolynomialForward

__all__ = [
    "BaseForward",
    "PolynomialForward",
]


# dispatch forward solver: {forward_name -> BaseForward}
forward_dispatch_table = {
    "polynomial": PolynomialForward
}

