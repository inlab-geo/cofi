from .model import Model, Parameter


class BaseForward:
    """Base class for all forward solvers in CoFI.

    All forward solvers must be sub-classes of this class and implements two methods:
    1. __init__
    2. misfit()
    """

    def __init__(self, func, init=None):
        if init:
            self.init = init
            self.init()
        self.objective = func

    def misfit(self, model: Model):
        pass
