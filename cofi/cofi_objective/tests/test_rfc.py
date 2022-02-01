import pytest
from cofi.cofi_objective.examples import ReceiverFunctionObjective, ReceiverFunction

import numpy as np


model = np.array([[1,4.0,1.7],
                  [3.5,4.3,1.7],
                  [8.0,4.2,2.0],
                  [20, 6,1.7],
                  [45,6.2,1.7]])

rfc_fwd = ReceiverFunction()
t, rf_data = rfc_fwd.calc(model, sn=0.1)

rfc_obj = ReceiverFunctionObjective(t, rf_data)
model_guess = np.array([[1,4.0,1.0],
                  [3.5,4.3,1.7],
                  [8.0,4.2,2.0],
                  [20, 6,1.7],
                  [45,6.2,1.7]])
rfc_obj.misfit(model_guess)
rfc_obj.misfit(model)

with pytest.raises(ValueError, match=r".*ensure the time array matches your data.*"):
    rfc_obj.misfit(model, time_shift=6)

with pytest.raises(ValueError, match=r".*dimension should be.*"):
    rfc_fwd.calc(np.array([[1,2]]))

with pytest.raises(ValueError, match=r".*maximum depth.*"):
    rfc_fwd.calc(np.array([[61,1,2]]))


