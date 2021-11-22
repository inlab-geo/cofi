from cofi.cofi_inverse import LinearRegression
from cofi.cofi_forward import BaseObjectiveFunction, Model, Parameter

import numpy as np

x = np.array([1,2,3])
X = np.array([x ** o for o in range(3)]).T
print(x.shape)
print(X)