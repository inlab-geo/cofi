from cofi.cofi_forward.model_params import Model
from cofi.cofi_inverse import BaseInverse
from cofi.cofi_forward import BaseObjectiveFunction, Model, Parameter

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(BaseInverse):
    def __init__(self, initial_model: Model, objective: BaseObjectiveFunction):
        self.initial = initial_model


