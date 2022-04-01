from abc import abstractmethod

import numpy as np


class BaseForward:
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, model: np.ndarray):
        raise NotImplementedError
