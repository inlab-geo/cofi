from cofi.cofi_objective import Model, BaseObjective
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Union, Callable
import numpy as np
import time


class DumbDescent(BaseObjective):
    """
    A greedy dumb descent where we just take a small step in every direction, and
    only accept if it improves our misfit.

    ALL inverters must take model and forward as the first two arguments
    """

    def __init__(
        self,
        model: Model,
        forward: Union[BaseObjective, Callable],
        step: np.float,
        time: np.float,
    ):
        self.start = model
        self.step = step
        self.current = deepcopy(model)
        self.objective = forward
        self.best_misfit = None
        self.best_info = None
        self.time = time

    def get_misfit(self, model: Model) -> Union[float, tuple]:
        if isinstance(self.objective, BaseObjective):
            user_vals = self.objective.get_misfit(model)
        else:
            user_vals = self.objective(*[param.value for param in model.params])
        if isinstance(user_vals, tuple):
            return user_vals
        else:
            return user_vals, None

    def solve(self) -> tuple:
        misfits_by_time = []
        if self.best_info is None:
            user_vals = self.get_misfit(self.current)
            self.best_misfit = float("inf") if np.isnan(user_vals[0]) else user_vals[0]
            self.best_info = user_vals[1:]
            misfits_by_time.append((0.0, self.best_misfit))
        t0 = time.time()
        t1 = t0
        while t1 - t0 < self.time:
            newmodel = deepcopy(self.current)
            for p in newmodel.params:
                if isinstance(p.value, np.ndarray):
                    p.value = (
                        p.value + (np.random.random(p.value.shape) - 0.5) * self.step
                    )
                else:
                    p.value = p.value + (np.random.random() - 0.5) * self.step
            user_vals = self.get_misfit(newmodel)

            misfit = user_vals[0]
            other = user_vals[1:]
            if np.isnan(misfit):
                pass  # invalid model, ignore
            else:
                misfits_by_time.append((time.time() - t0, misfit))
                if misfit < self.best_misfit:
                    self.current = newmodel
                    self.best_misfit = user_vals[0]
                    self.best_info = other
            t1 = time.time()

        # The result
        result_dict = dict(
            model=self.current, misfit=self.best_misfit, misfit_by_time=misfits_by_time
        )
        if self.best_info is not None:
            if isinstance(self.best_info, tuple):
                for i, item in enumerate(self.best_info):
                    result_dict[f"cofi_misfit_results_for_best_model_{i+1}"] = item
            else:
                result_dict["cofi_misfit_results_for_best_model"] = self.best_info
        return result_dict
