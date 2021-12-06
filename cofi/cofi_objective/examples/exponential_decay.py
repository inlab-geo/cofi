from cofi.cofi_objective import BaseObjective, Model

import numpy as np
from typing import Union


class ExpDecay(BaseObjective):
    """Defines the problem of exponential decay (sum of exponentials)

    Must implement the 'misfit' function.
    Depending on solvers, the following functions may also need to be provided:
    'residuals(model)', 'jacobian(model)', 'gradient(model)', 'hessian(model)'

    Apart from that, 'data_x()', 'data_y()', 'initial_model()' are also required for
    some solvers.
    """

    def __init__(self, data_x, data_y, initial_model: Union[Model, np.ndarray]):
        self.x = np.asanyarray(data_x)
        self.y = np.asanyarray(data_y)
        if isinstance(initial_model, Model):
            initial_model = initial_model.values()
        self.m0 = np.asanyarray(initial_model)
        self.n_params = self.m0.shape[0]

        if self.n_params % 2 != 0:
            raise ValueError(f"Exponential decay sums need to have an even number of parameters, but got {self.n_params} instead")

        self._last_validated_model = None


    def _forward(self, model: Union[Model, np.ndarray], ret_model=False):
        model = self._validate_model(model)
    
        yhat = np.zeros_like(self.x)
        for i in range(int(self.n_params/2)):
            yhat += model[i*2] * np.exp(-model[i*2+1] * self.x)
        return (yhat, model) if ret_model else yhat


    def residuals(self, model: Union[Model, np.ndarray]):
        yhat = self._forward(model)
        return yhat - self.y


    def misfit(self, model: Union[Model, np.ndarray]):
        residuals = self.residuals(model)
        return residuals @ residuals

    
    def jacobian(self, model: Union[Model, np.ndarray]):
        model = self._validate_model(model)
        
        jac = np.zeros([np.shape(self.x)[0], self.n_params])
        for i in range(int(self.n_params/2)):
            for j in range(self.x.shape[0]):
                jac[j,i*2] = np.exp(-model[i*2+1]*self.x[j])
                jac[j,i*2+1] = -model[i*2] * self.x[j] * np.exp(-model[i*2+1]*self.x[j])
        return jac


    def gradient(self, model: Union[Model, np.ndarray]):
        yhat, model = self._forward(model, True)
        jac = self.jacobian(model)
        return jac.T @ (yhat - self.y)


    def hessian(self, model: Union[Model, np.ndarray]):
        # using the standard approximation (J^T J)
        jac = self.jacobian(model)
        hessian = jac.T @ jac
        return hessian

    
    def data_x(self):
        return self.x


    def data_y(self):
        return self.y

    
    def initial_model(self):
        return self.m0

    
    def params_size(self):
        return self.n_params


    def _validate_model(self, model: Union[Model, np.ndarray]) -> np.ndarray:
        if model is self._last_validated_model:   # validated already (and converted if needed)
            return model

        if isinstance(model, Model):
            n_params = model.length()
            model = np.asanyarray(model.values())
        else:
            model = np.asanyarray(model)
            n_params = model.shape[0]
        
        if n_params != self.n_params:
            raise ValueError(f"Model length doesn't match initialisation, expected {self.n_params} parameters but got {n_params} instead")
        
        self._last_validated_model = model
        return model
