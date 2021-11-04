from cofi.cofi_inverse.base_inverser import BaseInverser
from lib import rjmcmc

class RJ_MCMC(BaseInverser):
    def __init__(self, x, y, n):
        # TODO - type validation / conversion (e.g. for numpy ndarray)
        self.x = x
        self.y = y
        self.n = n
        
        if not (len(self.x) == len(self.y) and len(self.x) == len(self.n)):
            raise ValueError("Input lists must be of same length")

        self.data = rjmcmc.dataset1d(x, y, n)

    def solve(self):
        pass
