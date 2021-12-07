from cofi.cofi_objective import BaseObjective, Model

import numpy as np
from typing import Union


class MultiArmedBandit(BaseObjective):
    """The Multi-armed Bandit Model (bandit or MAB for short).
    The problem aims to maximize the sum of the collected rewards, given a set of
    distributions :math:`B = \{R_1, ..., R_k\}` where each distribution is associated 
    with the rewards delivered by one of the K levers.

    """
    def __init__(self):
        pass

    def misfit(self, model: Union[Model, np.ndarray]):
        pass


