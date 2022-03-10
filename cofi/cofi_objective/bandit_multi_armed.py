from typing import Callable, Union
from numbers import Number

import numpy as np

from .. import Model, BaseObjective


class MultiArmedBandit(BaseObjective):
    """The Multi-armed Bandit Model (bandit or MAB for short).

    The problem aims to maximize the sum of the collected rewards, given a set of
    distributions :math:`B = \{R_1, ..., R_k\}` where each distribution is associated
    with the rewards delivered by one of the :math:`K` levers.

    """

    def __init__(
        self, n_bandits: int, pull_bandit: Callable[[int], Union[bool, Number]]
    ):
        # self.n_bandits = n_bandits
        # self.pull_bandit = pull_bandit
        pass

    def misfit(self, model: Union[Model, np.ndarray]):
        pass

    def propose(
        self,
    ):
        pass
