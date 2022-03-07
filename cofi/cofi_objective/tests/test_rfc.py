import pytest
from cofi.cofi_objective import ReceiverFunctionObjective, ReceiverFunction
from cofi.optimisers import ScipyOptimiserSolver

import numpy as np
import matplotlib.pyplot as plt


model = np.array(
    [[1, 4.0, 1.7], [3.5, 4.3, 1.7], [8.0, 4.2, 2.0], [20, 6, 1.7], [45, 6.2, 1.7]]
)

rfc_fwd = ReceiverFunction()
t, rf_data = rfc_fwd.calc(model, sn=0.1)
t2, rf_data2 = rfc_fwd.calc(
    model, sn=0.5
)  # Receiver function with added correlated noise
plt.plot(t, rf_data, label="No noise RF")
plt.plot(t2, rf_data2, "r-", label="Noisy RF")
plt.xlabel("Time/s")
plt.ylabel("Amplitude")
plt.legend()
# plt.show()

rfc_obj = ReceiverFunctionObjective(t, rf_data)
model_guess = np.array(
    [[1, 4.0, 1.0], [3.5, 4.3, 1.7], [8.0, 4.2, 2.0], [20, 6, 1.7], [45, 6.2, 1.7]]
)
rfc_obj.misfit(model_guess)
rfc_obj.misfit(model)

# Test using ScipyOptimiserSolver
model_guess = np.array(
    [[1, 4.0, 1.0], [3.5, 4.3, 1.7], [8.0, 4.2, 2.0], [20, 6, 1.7], [45, 6.2, 1.7]]
)
rfc_obj = ReceiverFunctionObjective(t, rf_data, model_guess)
solver = ScipyOptimiserSolver(rfc_obj)
model = solver.solve("Nelder-Mead")

with pytest.raises(ValueError, match=r".*ensure the time array matches your data.*"):
    rfc_obj.misfit(model, time_shift=6)

with pytest.raises(ValueError, match=r".*dimension should be.*"):
    rfc_fwd.calc(np.array([[1, 2]]))

with pytest.raises(ValueError, match=r".*maximum depth.*"):
    rfc_fwd.calc(np.array([[61, 1, 2]]))
