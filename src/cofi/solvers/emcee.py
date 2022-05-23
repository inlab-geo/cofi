import numpy as np
import emcee

from . import BaseSolver


class EmceeSolver(BaseSolver):
    documentation_links = []
    short_description = ""
    components_used = []
    required_in_problem = []
    optional_in_problem = dict()
    required_in_options = []
    optional_in_options = dict()

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)
        self._assign_args()

    def _assign_args(self):
        inv_problem = self.inv_problem
        # assign components in problem to args
        # assign options
        self._assign_args()

    def __call__(self) -> dict:
        return None

