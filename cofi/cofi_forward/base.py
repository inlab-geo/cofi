"""Base class for all forwarders."""

class BaseForwarder:
    def __init__(self, init=None):
        if init:
            self.init = init
            self.init()

    def misfit(self):
        pass
