from abc import ABC
from .model import Model, Parameter
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Union, Callable
import numpy as np
import time

class ObjectiveFunction:
    
    def __init__(self, func):
        self.obj = func

    def get_misfit(self, model: Model) -> tuple:
        return self.obj(*[p.value for p in model.params])



class DumbDescent:
    """
    A greedy dumb descent where we just take a small step in every direction, and
    only accept if it improves our misfit.
    """
    def __init__(self, start: Model, step: np.float, userobj: Union[ObjectiveFunction, Callable]):
        self.start = start
        self.step = step
        self.current = deepcopy(start)
        self.objective = userobj
        self.best_misfit = None
        self.best_info = None

    def get_misfit(self, model: Model) -> Union[float, tuple]:
        if isinstance(self.objective, ObjectiveFunction):
            user_vals = self.objective.get_misfit(model)
        else:
            user_vals = self.objective(*[param.value for param in model.params])
        if isinstance(user_vals, tuple):
            print(user_vals[0])
            return user_vals
        else:
            print(user_vals)
            return user_vals, None
    
    def run(self, time_seconds: float) -> tuple:
        t0 = time.time()
        t1 = t0
        if self.best_info is None:
            user_vals = self.get_misfit(self.current)
            self.best_misfit = float('inf') if np.isnan(user_vals[0]) else user_vals[0]
            self.best_info = user_vals[1:]
        while t1-t0 < time_seconds:
            newmodel = deepcopy(self.current)
            for p in newmodel.params:
                if isinstance(p.value, np.ndarray):
                    p.value = p.value + np.random.random(p.value.shape)*self.step
                else:
                    p.value = p.value + np.random.random()*self.step
            user_vals = self.get_misfit(newmodel)

            misfit = user_vals[0]
            other = user_vals[1:]
            if np.isnan(misfit):
                print(f"  got misfit of 'nan' in descent... ignoring model")
            elif misfit < self.best_misfit:
                self.current = newmodel
                self.best_misfit = user_vals[0]
                self.best_info = other
            t1 = time.time()
        if self.best_info is None:
            return self.current, self.best_misfit
        else:
            return tuple([self.current, self.best_misfit] + list(self.best_info))





