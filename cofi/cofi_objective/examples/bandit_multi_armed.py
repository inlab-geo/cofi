from cofi.cofi_objective import BaseObjective, Model

import numpy as np
from typing import Union, Protocol
from numbers import Number


class _PullCallable(Protocol):
    def __call__(self, i: int) -> Union[bool, Number]:
        ...


class MultiArmedBandit(BaseObjective):
    """The Multi-armed Bandit Model (bandit or MAB for short).

    The problem aims to maximize the sum of the collected rewards, given a set of
    distributions :math:`B = \{R_1, ..., R_k\}` where each distribution is associated 
    with the rewards delivered by one of the :math:`K` levers. 

    """

    def __init__(self, n_bandits: int, pull_bandit: _PullCallable):
        self.n_bandits = n_bandits
        self.pull_bandit = pull_bandit

    def misfit(self, model: Union[Model, np.ndarray]):
        pass
 
    def propose(self, ):
        pass

