class BaseInverser:
    """Base class for all inverse solvers (aka inversers) in CoFI.

    All inversers must be sub-classes of this class and implements two methods:
    1. __init__
    2. solve()
    """

    def __init__(self):
        pass

    def solve(self):
        pass
