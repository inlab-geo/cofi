from cofi.cofi_objective import BaseObjective, Model
from cofi.cofi_objective.base_forward import BaseForward
from .lib_rfc import rf
from .lib_rfc import _rfc

import numpy as np
from typing import Union
import matplotlib.pyplot as plt


class ReceiverFunctionObjective(BaseObjective):
    """Receiver functions are a class of seismic data used to study discontinuities
       (layering) in the Earth's crust. At each discontinuity, P-to-S conversions
       occur, introducing complexity in the waveform. By deconvolving horizontal- 
       and vertical-channel waveforms from earthquakes at teleseismic distances,
       we can isolate information about these conversions, and hence learn about
       the crustal structure. This deconvolved signal is the receiver function, and 
       has a highly non-linear dependencies on the local crustal properties.

       The model that this objective takes in is of dimension [nlayers,3]. The 
       values in model[:,0] give the depths of discontinuities in the model, while
       model[:,1] contains the S-wave speed above the interface. model[:,2] is the
       ratio of S-wave speed to P-wave speed. The maximum depth of discontinuity
       that can be considered is 60km.
    """
    def __init__(self):
        pass

    def misfit(self, model: Union[Model, np.ndarray]):
        model = np.asanyarray(model)


class ReceiverFunction(BaseForward):
    def __init__(self):
        pass

    def solve(self, model: Union[Model, np.ndarray]) -> np.ndarray:
        model = np.array([[1,4.0,1.7],
                  [3.5,4.3,1.7],
                  [8.0,4.2,2.0],
                  [20, 6,1.7],
                  [45,6.2,1.7]])
        t, rfunc = rf.rfcalc(self._validate_model(model))
        px = np.zeros([2*len(model),2])
        py = np.zeros([2*len(model),2])
        n=len(model)
        px[0::2,0],px[1::2,0],px[1::2,1],px[2::2,1] = model[:,1],model[:,1],model[:,0],model[:-1,0]
        plt.figure(figsize=(4,6))
        plt.xlabel('Vs (km/s)')
        plt.ylabel('Depth (km)')
        plt.gca().invert_yaxis()
        plt.plot(px[:,0],px[:,1],'y-')
        plt.show()


    def _validate_model(self, model: Union[Model, np.ndarray]) -> np.ndarray:
        model = np.asanyarray(model)
        if model.shape[1] != 3:
            raise ValueError(f"Model dimension should be (nlayers,3) but instead got {model.shape}")
        if np.any(model[:,0] > 60):
            raise ValueError(f"The first column of model represents depths of discontinuities and the maximum depth that can be considered is 60km")
        return model

