from typing import Tuple, Union

import numpy as np

from .. import BaseProblem
from . import _rfc


class ReceiverFunctionObjective(BaseProblem):
    """Receiver functions are a class of seismic data used to study discontinuities
    (layering) in the Earth's crust. At each discontinuity, P-to-S conversions
    occur, introducing complexity in the waveform. By deconvolving horizontal-
    and vertical-channel waveforms from earthquakes at teleseismic distances,
    we can isolate information about these conversions, and hence learn about
    the crustal structure. This deconvolved signal is the receiver function, and
    has a highly non-linear dependencies on the local crustal properties.

    The model that this objective takes in is of dimension [nlayers,3]:
    - model[:,0] gives the depths of discontinuities in the model,
    - model[:,1] contains the S-wave speed above the interface,
    - model[:,2] is the ratio of S-wave speed to P-wave speed.
    The maximum depth of discontinuity that can be considered is 60km.
    """

    def __init__(self, t, rf_data, initial_model=None):
        self.fwd = ReceiverFunction()
        self.set_dataset(t, rf_data)
        self.initial_model = initial_model

    def objective(
        self,
        model: np.ndarray,
        mtype=0,
        fs=25.0,
        gauss_a=2.5,
        water_c=0.0001,
        angle=35.0,
        time_shift=5.0,
        ndatar=626,
        v60=8.043,
        seed=1,
    ):
        model = self.fwd._validate_model(model)
        t, rf_calculated = self.fwd.calc(
            model, 0, mtype, fs, gauss_a, water_c, angle, time_shift, ndatar, v60, seed
        )
        if not np.array_equal(t, self.t):
            raise ValueError("Please ensure the time array matches your data")
        return np.linalg.norm(rf_calculated - self.rf_data)

    def initial_model(self):
        return self.initial_model

    # def log_likelihood(self, model, )


class ReceiverFunction:
    def __init__(self):
        pass

    def calc(
        self,
        model: np.ndarray,
        sn=0.0,
        mtype=0,
        fs=25.0,
        gauss_a=2.5,
        water_c=0.0001,
        angle=35.0,
        time_shift=5.0,
        ndatar=626,
        v60=8.043,
        seed=1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        model = self._validate_model(model)
        if sn == 0.0:
            t, rfunc = _rfc.rfcalc_nonoise(
                model, mtype, fs, gauss_a, water_c, angle, time_shift, ndatar, v60
            )
        else:
            t, rfunc = _rfc.rfcalc_noise(
                model,
                mtype,
                sn,
                fs,
                gauss_a,
                water_c,
                angle,
                time_shift,
                ndatar,
                v60,
                seed,
            )
        return t, rfunc

    def _validate_model(self, model: np.ndarray) -> np.ndarray:
        model = np.asanyarray(model)
        try:
            model = model.reshape([-1, 3])
        except:
            raise ValueError(
                f"Model dimension should be (nlayers,3) but instead got {model.shape}"
            )
        if np.any(model[:, 0] > 60):
            raise ValueError(
                f"The first column of model represents depths of discontinuities and"
                f" the maximum depth that can be considered is 60km"
            )
        return model
