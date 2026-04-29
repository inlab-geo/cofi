from typing import List, Dict
import warnings
import joblib
import os
import numpy
import operator
import matplotlib.pyplot as plt

import pyp223
import vtem_max_geom_lib
import pickle
import hashlib



# ------- wrap forward operator
class ForwardWrapper:

    def __init__(
        self,
        sample_model_params: Dict[str, numpy.ndarray],
        problem_setup: Dict[str, numpy.ndarray],
        system_spec: Dict[str, numpy.ndarray],
        survey_setup: Dict[str, numpy.ndarray],
        params_to_invert: List[str] = None,
        data_returned: List[str] = ["vertical", "inline"],
    ):
        self.sample_model_params = sample_model_params
        self.problem_setup = problem_setup
        self.system_spec = system_spec

        print(params_to_invert)

        self.n_fiducials = survey_setup["tx"].size
        if self.n_fiducials == 1:
            self.survey_setup = survey_setup
        else:
            self.survey_setup = []
            for i in range(self.n_fiducials):
                self.survey_setup.append(
                    {k: numpy.array([v[i]]) for k, v in survey_setup.items()}
                )

        if params_to_invert is None:
            params_to_invert = sorted(self.sample_model_params.keys())
        else:
            for p in params_to_invert:
                if p not in self.sample_model_params:
                    raise ValueError(f"Invalid parameter name: {p}")
        self.params_to_invert = sorted(params_to_invert)

        self.data_returned = []
        for d in data_returned:
            if d not in ["vertical", "inline"]:
                raise ValueError(
                    f"Invalid data return type: {d}. Must be either 'inline' or 'vertical'"
                )
            self.data_returned.append(d)
        if len(self.data_returned) == 2:
            self.data_returned = [
                "vertical",
                "inline",
            ]  # make sure the order is correct

        self._init_param_length()
        self.leroiair = pyp223.LeroiAir()

    def __call__(
        self, model: numpy.ndarray, return_lengths: bool = False
    ) -> numpy.ndarray:
        
        model_dict = self.model_dict(model)
        #model_dict=model
        model_dict = {
            k: numpy.deg2rad(v) if k in ["pdzm", "pdip"] else v
            for k, v in model_dict.items()
        }
        pbres = model_dict["res"][-1]
        leroiair_failure_count = 0
        xmodl = numpy.zeros([self.system_spec["nchnl"] * self.system_spec["ncmp"]])
        if self.n_fiducials == 1:
            dpred = self._call_forward(
                pbres,
                leroiair_failure_count,
                xmodl,
                model_dict,
                self.survey_setup,
            )
        else:
            # https://joblib.readthedocs.io/en/stable/parallel.html#thread-based-parallelism-vs-process-based-parallelism
            # calling compiled extension so using `prefer="threads"`
            dpred_all = joblib.Parallel(n_jobs=self.n_fiducials, prefer="threads")(
                joblib.delayed(self._call_forward)(
                    pbres, leroiair_failure_count, xmodl, model_dict, survey_setup
                )
                for survey_setup in self.survey_setup
            )
            dpred = numpy.concatenate(dpred_all)
            if return_lengths:
                return dpred, [len(d) for d in dpred_all]
        return dpred

    def _call_forward(self, pbres, failure_count, xmodl, model_dict, survey_setup):
        dpred = self.leroiair.formod_vtem_max_data(
            pbres=pbres,
            leroiair_failure_count=failure_count,
            xmodl=xmodl,
            **model_dict,
            **{k: v for k, v in survey_setup.items() if "id" not in k},
            **self.problem_setup,
            **self.system_spec,
        )
        if len(self.data_returned) == 2:
            dpred = dpred.reshape(-1)
        elif "vertical" in self.data_returned:
            dpred = dpred[:, 0]
        else:
            dpred = dpred[:, 1]
        dpred[dpred < 0.0001] = 0.0001
        return numpy.log(dpred)

    def jacobian(self, model: numpy.ndarray, relative_step=0.1) -> numpy.ndarray:
        original_dpred = self.__call__(model)
        jac = numpy.zeros((len(original_dpred), len(model)))
        for i in range(len(model)):
            perturbed_model = model.copy()
            step = relative_step * model[i]
            perturbed_model[i] += step
            perturbed_dpred = self.__call__(perturbed_model)
            derivative = (perturbed_dpred - original_dpred) / step
            jac[:, i] = derivative
        return jac

    def _init_param_length(self):
        nlyr = self.problem_setup["nlyr"]
        nstat = self.problem_setup["nstat"]
        nplt = self.problem_setup["nplt"]
        self.param_length = {
            "res": nlyr,
            "thk": (nlyr - 1) * nstat,
            "peast": nplt,
            "pnorth": nplt,
            "ptop": nplt,
            "pres": nplt,
            "plngth1": nplt,
            "plngth2": nplt,
            "pwdth1": nplt,
            "pwdth2": nplt,
            "pdzm": nplt,
            "pdip": nplt,
        }

    def model_dict(self, model: numpy.ndarray) -> Dict[str, numpy.ndarray]:
        model_dict = dict(self.sample_model_params)

        i = 0
        for p in self.params_to_invert:
            try:
                model_dict[p] = model[i : i + self.param_length[p]]
            except IndexError:
                raise ValueError(
                    f"Invalid model length. Expected {sum(self.param_length)} in "
                    f"total for parameters: {self.params_to_invert}"
                )
            i += self.param_length[p]
        return model_dict

    def model_vector(self, model: Dict[str, numpy.ndarray]) -> numpy.ndarray:
        return numpy.concatenate([model[p] for p in self.params_to_invert])



def plot_survey_map(survey_setup):
    _, ax = plt.subplots(1, 1)
    ax.plot(survey_setup['tx'],survey_setup['ty'],'.r')
    
    for i, txt in enumerate(survey_setup['fiducial_id']):
        if i%100==0:
            ax.annotate(txt, (survey_setup['tx'][i], survey_setup['ty'][i]))

# ------- wrap plotting functions
def plot_transient(model, forward, label, ax1=None, ax2=None, **kwargs):
    vertical_returned = "vertical" in forward.data_returned
    inline_returned = "inline" in forward.data_returned
    if ax1 is None:
        if vertical_returned and inline_returned:
            _, (ax1, ax2) = plt.subplots(1, 2)
        else:
            _, ax1 = plt.subplots(1, 1)
    x = (forward.system_spec["topn"] + forward.system_spec["tcls"]) / 2
    if forward.n_fiducials == 1:
        data = forward(model)
        _plot_data(
            x, data, vertical_returned, inline_returned, ax1, ax2, label=label, **kwargs
        )
    else:
        data, data_lengths = forward(model, return_lengths=True)
        i = 0
        for length in data_lengths:
            current_data = data[i : i + length]
            if i == 0:
                _plot_data(
                    x,
                    current_data,
                    vertical_returned,
                    inline_returned,
                    ax1,
                    ax2,
                    label=label,
                    **kwargs,
                )
            else:
                _plot_data(
                    x,
                    current_data,
                    vertical_returned,
                    inline_returned,
                    ax1,
                    ax2,
                    **kwargs,
                )
            i += length


def plot_field_data(data_x, data_obs, label, ax, **kwargs):
    if len(data_obs) != len(data_x):  # there are more than one transmitters
        n_fiducials = len(data_obs) // len(data_x)
        data_length = len(data_x)
        for i in range(n_fiducials):
            data_x_transmitter = data_x
            data_obs_transmitter = data_obs[i * data_length : (i + 1) * data_length]
            if i == 0:
                _plot_data(
                    data_x_transmitter,
                    data_obs_transmitter,
                    True,
                    False,
                    ax,
                    None,
                    label=label,
                    **kwargs,
                )
            else:
                _plot_data(
                    data_x_transmitter,
                    data_obs_transmitter,
                    True,
                    False,
                    ax,
                    None,
                    **kwargs,
                )
    else:
        _plot_data(data_x, data_obs, True, False, ax, None, label=label, **kwargs)


def _plot_data(
    x,
    data,
    vertical_returned,
    inline_returned,
    ax1,
    ax2,
    xlabel="Time (s)",
    ylabel="Response (pT/s)",
    **kwargs,
):
    if vertical_returned and inline_returned:
        data = data.reshape(-1, 2)
        data = numpy.exp(data)
        vertical = data[:, 0]
        inline = data[:, 1]
        ax1.loglog(x, vertical, **kwargs)
        ax2.loglog(x, inline, **kwargs)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
    else:
        data = numpy.exp(data)
        ax1.loglog(x, data, **kwargs)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)


def plot_predicted_profile(
    model,
    forward,
    label,
    gate_idx: list = None,
    line_id: list = None,
    ax=None,
    cmp='vertical',
    **kwargs,
):
    if forward.n_fiducials == 1:
        raise ValueError("This function is only for multiple transmitters")
    if ax is None:
        _, ax = plt.subplots(1, 1)
    x = numpy.array(
        [forward.survey_setup[i]["tx"] for i in range(forward.n_fiducials)]
    )
    
    ax.semilogy()
    old_data_returned = forward.data_returned
    forward.data_returned = [cmp]
    data, data_lengths = forward(model, return_lengths=True)
    idx_to_draw = range(data_lengths[0]) if gate_idx is None else gate_idx
    if line_id is not None:
        idx_to_draw_line_id = [
            i
            for i in range(forward.n_fiducials)
            if forward.survey_setup[i]["line_id"]
            in line_id
        ]
        x = x[idx_to_draw_line_id]
    labeled = False
    for i in idx_to_draw:
        y = numpy.array([data[j] for j in range(i, len(data), data_lengths[0])])
        if line_id is not None:
            y = y[idx_to_draw_line_id]
        if cmp=="vertical":
        	ylabel="Vertical component (pT/s)"
        else:
        	ylabel="Inline component (pT/s)"
        
        if not labeled:
        
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "Horizontal distance (m)",
                ylabel,
                label=label,
                **kwargs,
            )
            labeled = True
        else:
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "Horizontal distance (m)",
                ylabel,
                **kwargs,
            )
    forward.data_returned = old_data_returned


def plot_observed_profile(
    survey_setup,
    data_obs,
    label,
    gate_idx: list = None,
    line_id: list = None,
    annotate_fiducial_id: bool = False,
    ax=None,
    **kwargs,
):
    if survey_setup["tx"].size == 1:
        raise ValueError("This function is only for multiple transmitters")
    if ax is None:
        _, ax = plt.subplots(1, 1)
    survey_setup, _, data_obs = get_subset_of_survey(
        survey_setup, None, data_obs, gate_idx, line_id
    )
    x = survey_setup["tx"]
    n_gates = data_obs.size // x.size
    labled = False
    for i in range(n_gates):
        y = data_obs[i::n_gates]
        if not labled:
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "Horizontal distance (m)",
                "Vertical component (pT/s)",
                label=label,
                **kwargs,
            )
            labled = True
            if annotate_fiducial_id:
                for j in range(0, x.size, 200):
                    ax.annotate(str(survey_setup["fiducial_id"][j]), (x[j], y[j]), fontsize=8)
        else:
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "Horizontal distance (m)",
                "Vertical component (pT/s)",
                **kwargs,
            )

def get_subset_of_survey(
    survey_setup,
    system_spec,
    data_obs,
    gate_idx=None,
    line_id=None,
    fiducial_id=None,
):
    x = survey_setup["tx"]
    n_gates_total = data_obs.size // x.size
    data_obs = data_obs.reshape((x.size, n_gates_total))
    if gate_idx is None:
        gate_idx = range(n_gates_total)
    if fiducial_id is not None:
        if line_id is not None:
            warnings.warn(
                "Both line_id and fiducial_id are provided. "
                "Using fiducial_id."
            )
        fiducial_idx = [
            i
            for i in range(x.size)
            if survey_setup["fiducial_id"][i] in fiducial_id
        ]
    else:
        if line_id is None:
            line_id = list(set(survey_setup["line_id"]))
        fiducial_idx = [
            i
            for i in range(x.size)
            if survey_setup["line_id"][i] in line_id
        ]
    new_survey_setup = {
        k: v[fiducial_idx] for k, v in survey_setup.items()
    }
    if system_spec is None:
        new_system_spec = None
    else:
        new_system_spec = dict(system_spec)
        new_system_spec["nchnl"] = len(gate_idx)
        new_system_spec["topn"] = system_spec["topn"][gate_idx]
        new_system_spec["tcls"] = system_spec["tcls"][gate_idx]
    new_data_obs = numpy.zeros(
        (
            len(fiducial_idx),
            len(gate_idx),
        )
    )
    for i, idx in enumerate(gate_idx):
        new_data_obs[:, i] = data_obs[fiducial_idx, idx]
    new_data_obs = new_data_obs.flatten()
    return new_survey_setup, new_system_spec, new_data_obs


def gmt_plate_faces(fpt, forward, problem_setup, model, surface_elevation=400):
    f = numpy.zeros([6, 4, 3])
    fh = open(fpt + ".xy", "w")
    fh.close()
    fh = open(fpt + ".xz", "w")
    fh.close()
    fh = open(fpt + ".zy", "w")
    fh.close()

    model = forward.model_dict(model)

    for i in range(problem_setup["nplt"]):
        f[:, :, :] = vtem_max_geom_lib.get_plate_faces_from_orientation(
            model["peast"][i],
            model["pnorth"][i],
            surface_elevation - model["ptop"][i],
            model["plngth1"][i],
            model["plngth2"][i],
            model["pwdth1"][i],
            model["pwdth2"][i],
            problem_setup["pthk"][i],
            numpy.deg2rad(model["pdzm"][i]),
            numpy.deg2rad(model["pdip"][i]),
            numpy.deg2rad(problem_setup["plng"][i]),
        )

        fd = {}
        for i in range(6):
            fd[i] = numpy.mean(f[i, :, 2])
        fds = sorted(fd.items(), key=operator.itemgetter(1), reverse=False)
        fh = open(fpt + ".xy", "a")
        for tlp in fds:
            i = tlp[0]
            fh.write(">\n")
            for j in range(4):
                fh.write("{} {}\n".format(f[i, j, 0], f[i, j, 1]))
        fh.close()

        fd = {}
        for i in range(6):
            fd[i] = numpy.mean(f[i, :, 1])
        fds = sorted(fd.items(), key=operator.itemgetter(1), reverse=True)
        fh = open(fpt + ".xz", "a")
        for tlp in fds:
            i = tlp[0]
            fh.write(">\n")
            for j in range(4):
                fh.write("{} {}\n".format(f[i, j, 0], f[i, j, 2]))
        fh.close()

        fd = {}
        for i in range(6):
            fd[i] = numpy.mean(f[i, :, 0])
        fds = sorted(fd.items(), key=operator.itemgetter(1), reverse=False)
        fh = open(fpt + ".zy", "a")
        for tlp in fds:
            i = tlp[0]
            fh.write(">\n")
            for j in range(4):
                fh.write("{} {}\n".format(f[i, j, 2], f[i, j, 1]))
        fh.close()


def plot_plate_face(
    full_fpth, forward, ax, cleanup=True, surface_elevation=400, **plotting_kwargs
):
    with open(full_fpth, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    for i in range(0, len(lines), 5):
        x = [float(l.split()[0]) for l in lines[i + 1 : i + 5]]
        y = [float(l.split()[1]) for l in lines[i + 1 : i + 5]]
        if i == 5:
            plotting_kwargs = {k: v for k, v in plotting_kwargs.items() if k != "label"}
            ax.plot(x, y, **plotting_kwargs)
        else:
            ax.plot(x, y, **plotting_kwargs)
    if "xy" in full_fpth:
        if forward.n_fiducials == 1:
            tx = forward.survey_setup["tx"]
            ty = forward.survey_setup["ty"]
            ax.plot(tx, ty, "o", color="orange")
        else:
            tx_min = float("inf")
            tx_max = float("-inf")
            for i in range(forward.n_fiducials):
                survey_setup = forward.survey_setup[i]
                ax.plot(
                    survey_setup["tx"],
                    survey_setup["ty"],
                    "o",
                    color="orange",
                )
                tx_min = min(tx_min, survey_setup["tx"])
                tx_max = max(tx_max, survey_setup["tx"])
            ax.set_xlim(tx_min - 10, tx_max + 10)
    elif "xz" in full_fpth:
        ax.axhline(surface_elevation, color="black", linestyle="--")
    elif "zy" in full_fpth:
        ax.axvline(surface_elevation, color="black", linestyle="--")
        if not (ax.xaxis_inverted()):
	        ax.invert_xaxis()
    if cleanup == True:
        os.remove(full_fpth)
    elif cleanup == "all":
        for ext in [".xy", ".xz", ".zy"]:
            os.remove(full_fpth[:-3] + ext)


def plot_plate_faces(
    fpt, forward, model, ax1, ax2, ax3, surface_elevation=400, **plotting_kwargs
):
    gmt_plate_faces(fpt, forward, forward.problem_setup, model, surface_elevation)
    plot_plate_face(
        fpt + ".xy", forward, ax1, True, surface_elevation, **plotting_kwargs
    )
    plot_plate_face(
        fpt + ".zy", forward, ax2, True, surface_elevation, **plotting_kwargs
    )
    plot_plate_face(
        fpt + ".xz", forward, ax3, True, surface_elevation, **plotting_kwargs
    )
    ax1.set_xlabel("Inline (m)")
    ax1.set_ylabel("Crossline (m)")
    ax2.set_xlabel("Elevation (m)")
    ax2.set_ylabel("Crossline (m)")
    ax3.set_xlabel("Inline (m)")
    ax3.set_ylabel("Elevation (m)")


def plot_plate_faces_single(fpt, option, forward, model, ax, **plotting_kwargs):
    gmt_plate_faces(fpt, forward, forward.problem_setup, model)
    plot_plate_face(fpt + "." + option, forward, ax, "all", **plotting_kwargs)
    if option == "xy":
        ax.set_xlabel("Inline (m)")
        ax.set_ylabel("Crossline (m)")
    elif option == "zy":
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel("Crossline (m)")
        ax.invert_xaxis()

    else:
        ax.set_xlabel("Inline (m)")
        ax.set_ylabel("Elevation (m)")
    if "x" in option and forward.n_fiducials > 1:
        tx_min = float("inf")
        tx_max = float("-inf")
        for i in range(forward.n_fiducials):
            tx = forward.survey_setup[i]["tx"]
            tx_min = min(tx_min, tx)
            tx_max = max(tx_max, tx)
        ax.set_xlim(tx_min - 10, tx_max + 10)
        
# ------- problem setup
problem_setup = {
    "nlyr": 2,  # number of layers (icl. halfspace)
    "nstat": 1,  # number of fiducials/stations
    "nplt": 1,  # number of thin plates
    "cellw": 25,  # cell width
    "pthk": numpy.array([1]),  # plates thickness
    "plng": numpy.deg2rad(numpy.array([0])),  # plates plunge (orientation)
}

# ------- system specification with waveform and gates read from file
system_spec = {
    "ncmp": 2,  # system spec: number of components
    "cmp": 2,  # system spec: active components
    "ntrn": 3,  # system spec: number of transmitter turns
    "txarea": 531,  # system spec: transmitter area
    "ampt": 0,  # system spec: amplitude type AMPS 0
}

# ------- example survey settings
tx_min = 115
tx_max = 281
tx_interval = 15
n_fiducials = (tx_max - tx_min - 1) // tx_interval + 1
tx = numpy.arange(tx_min, tx_max, tx_interval)
survey_setup = {
    "tx": tx,  # transmitter easting/x-position
    "ty": numpy.array([100] * n_fiducials),  # transmitter northing/y-position
    "tz": numpy.array([50] * n_fiducials),  # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90] * n_fiducials)),  # transmitter azimuth
    "tincl": numpy.deg2rad(
        numpy.array([6] * n_fiducials)
    ),  # transmitter inclination
    "rx": tx,  # receiver easting/x-position
    "ry": numpy.array([100] * n_fiducials),  # receiver northing/y-position
    "rz": numpy.array([50] * n_fiducials),  # receiver height/z-position
    "trdx": numpy.array([0] * n_fiducials),  # transmitter receiver separation inline
    "trdy": numpy.array(
        [0] * n_fiducials
    ),  # transmitter receiver separation crossline
    "trdz": numpy.array(
        [0] * n_fiducials
    ),  # transmitter receiver separation vertical
}


# ------- read survey data
def default_gates_and_waveform(file_name="LeroiAir.cfl") -> dict:
    
    nsx=3200
    swx=numpy.array(
      [0.00000000e+00, 5.20833333e-06, 1.04166667e-05, 1.56250000e-05,
       2.08333333e-05, 2.60416667e-05, 3.12500000e-05, 3.64583333e-05,
       4.16666666e-05, 4.68750000e-05, 5.20833333e-05, 5.72916666e-05,
       6.25000000e-05, 6.77083333e-05, 7.29166666e-05, 7.81249999e-05,
       8.33333333e-05, 8.85416666e-05, 9.37499999e-05, 9.89583333e-05,
       1.04166667e-04, 1.09375000e-04, 1.14583333e-04, 1.19791667e-04,
       1.25000000e-04, 1.30208333e-04, 1.35416667e-04, 1.40625000e-04,
       1.45833333e-04, 1.51041667e-04, 1.56250000e-04, 1.61458333e-04,
       1.66666667e-04, 1.71875000e-04, 1.77083333e-04, 1.82291667e-04,
       1.87500000e-04, 1.92708333e-04, 1.97916667e-04, 2.03125000e-04,
       2.08333333e-04, 2.13541667e-04, 2.18750000e-04, 2.23958333e-04,
       2.29166667e-04, 2.34375000e-04, 2.39583333e-04, 2.44791667e-04,
       2.50000000e-04, 2.55208333e-04, 2.60416667e-04, 2.65625000e-04,
       2.70833333e-04, 2.76041666e-04, 2.81250000e-04, 2.86458333e-04,
       2.91666666e-04, 2.96875000e-04, 3.02083333e-04, 3.07291666e-04,
       3.12500000e-04, 3.17708333e-04, 3.22916666e-04, 3.28125000e-04,
       3.33333333e-04, 3.38541666e-04, 3.43750000e-04, 3.48958333e-04,
       3.54166666e-04, 3.59375000e-04, 3.64583333e-04, 3.69791666e-04,
       3.75000000e-04, 3.80208333e-04, 3.85416666e-04, 3.90625000e-04,
       3.95833333e-04, 4.01041666e-04, 4.06250000e-04, 4.11458333e-04,
       4.16666666e-04, 4.21875000e-04, 4.27083333e-04, 4.32291666e-04,
       4.37500000e-04, 4.42708333e-04, 4.47916666e-04, 4.53125000e-04,
       4.58333333e-04, 4.63541666e-04, 4.68750000e-04, 4.73958333e-04,
       4.79166666e-04, 4.84375000e-04, 4.89583333e-04, 4.94791666e-04,
       5.00000000e-04, 5.05208333e-04, 5.10416666e-04, 5.15625000e-04,
       5.20833333e-04, 5.26041666e-04, 5.31250000e-04, 5.36458333e-04,
       5.41666666e-04, 5.46875000e-04, 5.52083333e-04, 5.57291666e-04,
       5.62500000e-04, 5.67708333e-04, 5.72916666e-04, 5.78125000e-04,
       5.83333333e-04, 5.88541666e-04, 5.93750000e-04, 5.98958333e-04,
       6.04166666e-04, 6.09375000e-04, 6.14583333e-04, 6.19791666e-04,
       6.25000000e-04, 6.30208333e-04, 6.35416666e-04, 6.40625000e-04,
       6.45833333e-04, 6.51041666e-04, 6.56250000e-04, 6.61458333e-04,
       6.66666666e-04, 6.71875000e-04, 6.77083333e-04, 6.82291666e-04,
       6.87500000e-04, 6.92708333e-04, 6.97916666e-04, 7.03125000e-04,
       7.08333333e-04, 7.13541666e-04, 7.18750000e-04, 7.23958333e-04,
       7.29166666e-04, 7.34375000e-04, 7.39583333e-04, 7.44791666e-04,
       7.50000000e-04, 7.55208333e-04, 7.60416666e-04, 7.65625000e-04,
       7.70833333e-04, 7.76041666e-04, 7.81249999e-04, 7.86458333e-04,
       7.91666666e-04, 7.96874999e-04, 8.02083333e-04, 8.07291666e-04,
       8.12499999e-04, 8.17708333e-04, 8.22916666e-04, 8.28124999e-04,
       8.33333333e-04, 8.38541666e-04, 8.43749999e-04, 8.48958333e-04,
       8.54166666e-04, 8.59374999e-04, 8.64583333e-04, 8.69791666e-04,
       8.74999999e-04, 8.80208333e-04, 8.85416666e-04, 8.90624999e-04,
       8.95833333e-04, 9.01041666e-04, 9.06249999e-04, 9.11458333e-04,
       9.16666666e-04, 9.21874999e-04, 9.27083333e-04, 9.32291666e-04,
       9.37499999e-04, 9.42708333e-04, 9.47916666e-04, 9.53124999e-04,
       9.58333333e-04, 9.63541666e-04, 9.68749999e-04, 9.73958333e-04,
       9.79166666e-04, 9.84374999e-04, 9.89583333e-04, 9.94791666e-04,
       9.99999999e-04, 1.00520833e-03, 1.01041667e-03, 1.01562500e-03,
       1.02083333e-03, 1.02604167e-03, 1.03125000e-03, 1.03645833e-03,
       1.04166667e-03, 1.04687500e-03, 1.05208333e-03, 1.05729167e-03,
       1.06250000e-03, 1.06770833e-03, 1.07291667e-03, 1.07812500e-03,
       1.08333333e-03, 1.08854167e-03, 1.09375000e-03, 1.09895833e-03,
       1.10416667e-03, 1.10937500e-03, 1.11458333e-03, 1.11979167e-03,
       1.12500000e-03, 1.13020833e-03, 1.13541667e-03, 1.14062500e-03,
       1.14583333e-03, 1.15104167e-03, 1.15625000e-03, 1.16145833e-03,
       1.16666667e-03, 1.17187500e-03, 1.17708333e-03, 1.18229167e-03,
       1.18750000e-03, 1.19270833e-03, 1.19791667e-03, 1.20312500e-03,
       1.20833333e-03, 1.21354167e-03, 1.21875000e-03, 1.22395833e-03,
       1.22916667e-03, 1.23437500e-03, 1.23958333e-03, 1.24479167e-03,
       1.25000000e-03, 1.25520833e-03, 1.26041667e-03, 1.26562500e-03,
       1.27083333e-03, 1.27604167e-03, 1.28125000e-03, 1.28645833e-03,
       1.29166667e-03, 1.29687500e-03, 1.30208333e-03, 1.30729167e-03,
       1.31250000e-03, 1.31770833e-03, 1.32291667e-03, 1.32812500e-03,
       1.33333333e-03, 1.33854167e-03, 1.34375000e-03, 1.34895833e-03,
       1.35416667e-03, 1.35937500e-03, 1.36458333e-03, 1.36979167e-03,
       1.37500000e-03, 1.38020833e-03, 1.38541667e-03, 1.39062500e-03,
       1.39583333e-03, 1.40104167e-03, 1.40625000e-03, 1.41145833e-03,
       1.41666667e-03, 1.42187500e-03, 1.42708333e-03, 1.43229167e-03,
       1.43750000e-03, 1.44270833e-03, 1.44791667e-03, 1.45312500e-03,
       1.45833333e-03, 1.46354167e-03, 1.46875000e-03, 1.47395833e-03,
       1.47916667e-03, 1.48437500e-03, 1.48958333e-03, 1.49479167e-03,
       1.50000000e-03, 1.50520833e-03, 1.51041667e-03, 1.51562500e-03,
       1.52083333e-03, 1.52604167e-03, 1.53125000e-03, 1.53645833e-03,
       1.54166667e-03, 1.54687500e-03, 1.55208333e-03, 1.55729167e-03,
       1.56250000e-03, 1.56770833e-03, 1.57291667e-03, 1.57812500e-03,
       1.58333333e-03, 1.58854167e-03, 1.59375000e-03, 1.59895833e-03,
       1.60416667e-03, 1.60937500e-03, 1.61458333e-03, 1.61979167e-03,
       1.62500000e-03, 1.63020833e-03, 1.63541667e-03, 1.64062500e-03,
       1.64583333e-03, 1.65104167e-03, 1.65625000e-03, 1.66145833e-03,
       1.66666667e-03, 1.67187500e-03, 1.67708333e-03, 1.68229167e-03,
       1.68750000e-03, 1.69270833e-03, 1.69791667e-03, 1.70312500e-03,
       1.70833333e-03, 1.71354167e-03, 1.71875000e-03, 1.72395833e-03,
       1.72916667e-03, 1.73437500e-03, 1.73958333e-03, 1.74479167e-03,
       1.75000000e-03, 1.75520833e-03, 1.76041667e-03, 1.76562500e-03,
       1.77083333e-03, 1.77604167e-03, 1.78125000e-03, 1.78645833e-03,
       1.79166667e-03, 1.79687500e-03, 1.80208333e-03, 1.80729167e-03,
       1.81250000e-03, 1.81770833e-03, 1.82291667e-03, 1.82812500e-03,
       1.83333333e-03, 1.83854167e-03, 1.84375000e-03, 1.84895833e-03,
       1.85416667e-03, 1.85937500e-03, 1.86458333e-03, 1.86979167e-03,
       1.87500000e-03, 1.88020833e-03, 1.88541667e-03, 1.89062500e-03,
       1.89583333e-03, 1.90104167e-03, 1.90625000e-03, 1.91145833e-03,
       1.91666667e-03, 1.92187500e-03, 1.92708333e-03, 1.93229167e-03,
       1.93750000e-03, 1.94270833e-03, 1.94791667e-03, 1.95312500e-03,
       1.95833333e-03, 1.96354167e-03, 1.96875000e-03, 1.97395833e-03,
       1.97916667e-03, 1.98437500e-03, 1.98958333e-03, 1.99479167e-03,
       2.00000000e-03, 2.00520833e-03, 2.01041667e-03, 2.01562500e-03,
       2.02083333e-03, 2.02604167e-03, 2.03125000e-03, 2.03645833e-03,
       2.04166667e-03, 2.04687500e-03, 2.05208333e-03, 2.05729167e-03,
       2.06250000e-03, 2.06770833e-03, 2.07291667e-03, 2.07812500e-03,
       2.08333333e-03, 2.08854167e-03, 2.09375000e-03, 2.09895833e-03,
       2.10416667e-03, 2.10937500e-03, 2.11458333e-03, 2.11979167e-03,
       2.12500000e-03, 2.13020833e-03, 2.13541667e-03, 2.14062500e-03,
       2.14583333e-03, 2.15104167e-03, 2.15625000e-03, 2.16145833e-03,
       2.16666667e-03, 2.17187500e-03, 2.17708333e-03, 2.18229167e-03,
       2.18750000e-03, 2.19270833e-03, 2.19791667e-03, 2.20312500e-03,
       2.20833333e-03, 2.21354167e-03, 2.21875000e-03, 2.22395833e-03,
       2.22916667e-03, 2.23437500e-03, 2.23958333e-03, 2.24479167e-03,
       2.25000000e-03, 2.25520833e-03, 2.26041667e-03, 2.26562500e-03,
       2.27083333e-03, 2.27604167e-03, 2.28125000e-03, 2.28645833e-03,
       2.29166667e-03, 2.29687500e-03, 2.30208333e-03, 2.30729167e-03,
       2.31250000e-03, 2.31770833e-03, 2.32291667e-03, 2.32812500e-03,
       2.33333333e-03, 2.33854167e-03, 2.34375000e-03, 2.34895833e-03,
       2.35416667e-03, 2.35937500e-03, 2.36458333e-03, 2.36979167e-03,
       2.37500000e-03, 2.38020833e-03, 2.38541667e-03, 2.39062500e-03,
       2.39583333e-03, 2.40104167e-03, 2.40625000e-03, 2.41145833e-03,
       2.41666667e-03, 2.42187500e-03, 2.42708333e-03, 2.43229167e-03,
       2.43750000e-03, 2.44270833e-03, 2.44791667e-03, 2.45312500e-03,
       2.45833333e-03, 2.46354167e-03, 2.46875000e-03, 2.47395833e-03,
       2.47916667e-03, 2.48437500e-03, 2.48958333e-03, 2.49479167e-03,
       2.50000000e-03, 2.50520833e-03, 2.51041667e-03, 2.51562500e-03,
       2.52083333e-03, 2.52604167e-03, 2.53125000e-03, 2.53645833e-03,
       2.54166667e-03, 2.54687500e-03, 2.55208333e-03, 2.55729167e-03,
       2.56250000e-03, 2.56770833e-03, 2.57291667e-03, 2.57812500e-03,
       2.58333333e-03, 2.58854167e-03, 2.59375000e-03, 2.59895833e-03,
       2.60416667e-03, 2.60937500e-03, 2.61458333e-03, 2.61979166e-03,
       2.62500000e-03, 2.63020833e-03, 2.63541666e-03, 2.64062500e-03,
       2.64583333e-03, 2.65104166e-03, 2.65625000e-03, 2.66145833e-03,
       2.66666666e-03, 2.67187500e-03, 2.67708333e-03, 2.68229166e-03,
       2.68750000e-03, 2.69270833e-03, 2.69791666e-03, 2.70312500e-03,
       2.70833333e-03, 2.71354166e-03, 2.71875000e-03, 2.72395833e-03,
       2.72916666e-03, 2.73437500e-03, 2.73958333e-03, 2.74479166e-03,
       2.75000000e-03, 2.75520833e-03, 2.76041666e-03, 2.76562500e-03,
       2.77083333e-03, 2.77604166e-03, 2.78125000e-03, 2.78645833e-03,
       2.79166666e-03, 2.79687500e-03, 2.80208333e-03, 2.80729166e-03,
       2.81250000e-03, 2.81770833e-03, 2.82291666e-03, 2.82812500e-03,
       2.83333333e-03, 2.83854166e-03, 2.84375000e-03, 2.84895833e-03,
       2.85416666e-03, 2.85937500e-03, 2.86458333e-03, 2.86979166e-03,
       2.87500000e-03, 2.88020833e-03, 2.88541666e-03, 2.89062500e-03,
       2.89583333e-03, 2.90104166e-03, 2.90625000e-03, 2.91145833e-03,
       2.91666666e-03, 2.92187500e-03, 2.92708333e-03, 2.93229166e-03,
       2.93750000e-03, 2.94270833e-03, 2.94791666e-03, 2.95312500e-03,
       2.95833333e-03, 2.96354166e-03, 2.96875000e-03, 2.97395833e-03,
       2.97916666e-03, 2.98437500e-03, 2.98958333e-03, 2.99479166e-03,
       3.00000000e-03, 3.00520833e-03, 3.01041666e-03, 3.01562500e-03,
       3.02083333e-03, 3.02604166e-03, 3.03125000e-03, 3.03645833e-03,
       3.04166666e-03, 3.04687500e-03, 3.05208333e-03, 3.05729166e-03,
       3.06250000e-03, 3.06770833e-03, 3.07291666e-03, 3.07812500e-03,
       3.08333333e-03, 3.08854166e-03, 3.09375000e-03, 3.09895833e-03,
       3.10416666e-03, 3.10937500e-03, 3.11458333e-03, 3.11979166e-03,
       3.12500000e-03, 3.13020833e-03, 3.13541666e-03, 3.14062500e-03,
       3.14583333e-03, 3.15104166e-03, 3.15625000e-03, 3.16145833e-03,
       3.16666666e-03, 3.17187500e-03, 3.17708333e-03, 3.18229166e-03,
       3.18750000e-03, 3.19270833e-03, 3.19791666e-03, 3.20312500e-03,
       3.20833333e-03, 3.21354166e-03, 3.21875000e-03, 3.22395833e-03,
       3.22916666e-03, 3.23437500e-03, 3.23958333e-03, 3.24479166e-03,
       3.25000000e-03, 3.25520833e-03, 3.26041666e-03, 3.26562500e-03,
       3.27083333e-03, 3.27604166e-03, 3.28125000e-03, 3.28645833e-03,
       3.29166666e-03, 3.29687500e-03, 3.30208333e-03, 3.30729166e-03,
       3.31250000e-03, 3.31770833e-03, 3.32291666e-03, 3.32812500e-03,
       3.33333333e-03, 3.33854166e-03, 3.34375000e-03, 3.34895833e-03,
       3.35416666e-03, 3.35937500e-03, 3.36458333e-03, 3.36979166e-03,
       3.37500000e-03, 3.38020833e-03, 3.38541666e-03, 3.39062500e-03,
       3.39583333e-03, 3.40104166e-03, 3.40625000e-03, 3.41145833e-03,
       3.41666666e-03, 3.42187500e-03, 3.42708333e-03, 3.43229166e-03,
       3.43750000e-03, 3.44270833e-03, 3.44791666e-03, 3.45312500e-03,
       3.45833333e-03, 3.46354166e-03, 3.46875000e-03, 3.47395833e-03,
       3.47916666e-03, 3.48437500e-03, 3.48958333e-03, 3.49479166e-03,
       3.50000000e-03, 3.50520833e-03, 3.51041666e-03, 3.51562500e-03,
       3.52083333e-03, 3.52604166e-03, 3.53125000e-03, 3.53645833e-03,
       3.54166666e-03, 3.54687500e-03, 3.55208333e-03, 3.55729166e-03,
       3.56250000e-03, 3.56770833e-03, 3.57291666e-03, 3.57812500e-03,
       3.58333333e-03, 3.58854166e-03, 3.59375000e-03, 3.59895833e-03,
       3.60416666e-03, 3.60937500e-03, 3.61458333e-03, 3.61979166e-03,
       3.62500000e-03, 3.63020833e-03, 3.63541666e-03, 3.64062500e-03,
       3.64583333e-03, 3.65104166e-03, 3.65625000e-03, 3.66145833e-03,
       3.66666666e-03, 3.67187500e-03, 3.67708333e-03, 3.68229166e-03,
       3.68750000e-03, 3.69270833e-03, 3.69791666e-03, 3.70312500e-03,
       3.70833333e-03, 3.71354166e-03, 3.71875000e-03, 3.72395833e-03,
       3.72916666e-03, 3.73437500e-03, 3.73958333e-03, 3.74479166e-03,
       3.75000000e-03, 3.75520833e-03, 3.76041666e-03, 3.76562500e-03,
       3.77083333e-03, 3.77604166e-03, 3.78125000e-03, 3.78645833e-03,
       3.79166666e-03, 3.79687500e-03, 3.80208333e-03, 3.80729166e-03,
       3.81250000e-03, 3.81770833e-03, 3.82291666e-03, 3.82812500e-03,
       3.83333333e-03, 3.83854166e-03, 3.84375000e-03, 3.84895833e-03,
       3.85416666e-03, 3.85937500e-03, 3.86458333e-03, 3.86979166e-03,
       3.87500000e-03, 3.88020833e-03, 3.88541666e-03, 3.89062500e-03,
       3.89583333e-03, 3.90104166e-03, 3.90625000e-03, 3.91145833e-03,
       3.91666666e-03, 3.92187500e-03, 3.92708333e-03, 3.93229166e-03,
       3.93750000e-03, 3.94270833e-03, 3.94791666e-03, 3.95312500e-03,
       3.95833333e-03, 3.96354166e-03, 3.96875000e-03, 3.97395833e-03,
       3.97916666e-03, 3.98437500e-03, 3.98958333e-03, 3.99479166e-03,
       4.00000000e-03, 4.00520833e-03, 4.01041666e-03, 4.01562500e-03,
       4.02083333e-03, 4.02604166e-03, 4.03125000e-03, 4.03645833e-03,
       4.04166666e-03, 4.04687500e-03, 4.05208333e-03, 4.05729166e-03,
       4.06250000e-03, 4.06770833e-03, 4.07291666e-03, 4.07812500e-03,
       4.08333333e-03, 4.08854166e-03, 4.09375000e-03, 4.09895833e-03,
       4.10416666e-03, 4.10937500e-03, 4.11458333e-03, 4.11979166e-03,
       4.12500000e-03, 4.13020833e-03, 4.13541666e-03, 4.14062500e-03,
       4.14583333e-03, 4.15104166e-03, 4.15625000e-03, 4.16145833e-03,
       4.16666666e-03, 4.17187500e-03, 4.17708333e-03, 4.18229166e-03,
       4.18750000e-03, 4.19270833e-03, 4.19791666e-03, 4.20312500e-03,
       4.20833333e-03, 4.21354166e-03, 4.21875000e-03, 4.22395833e-03,
       4.22916666e-03, 4.23437500e-03, 4.23958333e-03, 4.24479166e-03,
       4.25000000e-03, 4.25520833e-03, 4.26041666e-03, 4.26562500e-03,
       4.27083333e-03, 4.27604166e-03, 4.28125000e-03, 4.28645833e-03,
       4.29166666e-03, 4.29687500e-03, 4.30208333e-03, 4.30729166e-03,
       4.31250000e-03, 4.31770833e-03, 4.32291666e-03, 4.32812500e-03,
       4.33333333e-03, 4.33854166e-03, 4.34375000e-03, 4.34895833e-03,
       4.35416666e-03, 4.35937500e-03, 4.36458333e-03, 4.36979166e-03,
       4.37500000e-03, 4.38020833e-03, 4.38541666e-03, 4.39062500e-03,
       4.39583333e-03, 4.40104166e-03, 4.40625000e-03, 4.41145833e-03,
       4.41666666e-03, 4.42187500e-03, 4.42708333e-03, 4.43229166e-03,
       4.43750000e-03, 4.44270833e-03, 4.44791666e-03, 4.45312500e-03,
       4.45833333e-03, 4.46354166e-03, 4.46875000e-03, 4.47395833e-03,
       4.47916666e-03, 4.48437500e-03, 4.48958333e-03, 4.49479166e-03,
       4.50000000e-03, 4.50520833e-03, 4.51041666e-03, 4.51562500e-03,
       4.52083333e-03, 4.52604166e-03, 4.53125000e-03, 4.53645833e-03,
       4.54166666e-03, 4.54687500e-03, 4.55208333e-03, 4.55729166e-03,
       4.56250000e-03, 4.56770833e-03, 4.57291666e-03, 4.57812500e-03,
       4.58333333e-03, 4.58854166e-03, 4.59375000e-03, 4.59895833e-03,
       4.60416666e-03, 4.60937500e-03, 4.61458333e-03, 4.61979166e-03,
       4.62500000e-03, 4.63020833e-03, 4.63541666e-03, 4.64062500e-03,
       4.64583333e-03, 4.65104166e-03, 4.65625000e-03, 4.66145833e-03,
       4.66666666e-03, 4.67187500e-03, 4.67708333e-03, 4.68229166e-03,
       4.68750000e-03, 4.69270833e-03, 4.69791666e-03, 4.70312500e-03,
       4.70833333e-03, 4.71354166e-03, 4.71875000e-03, 4.72395833e-03,
       4.72916666e-03, 4.73437500e-03, 4.73958333e-03, 4.74479166e-03,
       4.75000000e-03, 4.75520833e-03, 4.76041666e-03, 4.76562500e-03,
       4.77083333e-03, 4.77604166e-03, 4.78125000e-03, 4.78645833e-03,
       4.79166666e-03, 4.79687500e-03, 4.80208333e-03, 4.80729166e-03,
       4.81250000e-03, 4.81770833e-03, 4.82291666e-03, 4.82812500e-03,
       4.83333333e-03, 4.83854166e-03, 4.84375000e-03, 4.84895833e-03,
       4.85416666e-03, 4.85937500e-03, 4.86458333e-03, 4.86979166e-03,
       4.87500000e-03, 4.88020833e-03, 4.88541666e-03, 4.89062500e-03,
       4.89583333e-03, 4.90104166e-03, 4.90625000e-03, 4.91145833e-03,
       4.91666666e-03, 4.92187500e-03, 4.92708333e-03, 4.93229166e-03,
       4.93750000e-03, 4.94270833e-03, 4.94791666e-03, 4.95312500e-03,
       4.95833333e-03, 4.96354166e-03, 4.96875000e-03, 4.97395833e-03,
       4.97916666e-03, 4.98437500e-03, 4.98958333e-03, 4.99479166e-03,
       5.00000000e-03, 5.00520833e-03, 5.01041666e-03, 5.01562500e-03,
       5.02083333e-03, 5.02604166e-03, 5.03125000e-03, 5.03645833e-03,
       5.04166666e-03, 5.04687500e-03, 5.05208333e-03, 5.05729166e-03,
       5.06250000e-03, 5.06770833e-03, 5.07291666e-03, 5.07812500e-03,
       5.08333333e-03, 5.08854166e-03, 5.09375000e-03, 5.09895833e-03,
       5.10416666e-03, 5.10937500e-03, 5.11458333e-03, 5.11979166e-03,
       5.12500000e-03, 5.13020833e-03, 5.13541666e-03, 5.14062500e-03,
       5.14583333e-03, 5.15104166e-03, 5.15625000e-03, 5.16145833e-03,
       5.16666666e-03, 5.17187500e-03, 5.17708333e-03, 5.18229166e-03,
       5.18750000e-03, 5.19270833e-03, 5.19791666e-03, 5.20312500e-03,
       5.20833333e-03, 5.21354166e-03, 5.21875000e-03, 5.22395833e-03,
       5.22916666e-03, 5.23437500e-03, 5.23958333e-03, 5.24479166e-03,
       5.25000000e-03, 5.25520833e-03, 5.26041666e-03, 5.26562500e-03,
       5.27083333e-03, 5.27604166e-03, 5.28125000e-03, 5.28645833e-03,
       5.29166666e-03, 5.29687500e-03, 5.30208333e-03, 5.30729166e-03,
       5.31250000e-03, 5.31770833e-03, 5.32291666e-03, 5.32812500e-03,
       5.33333333e-03, 5.33854166e-03, 5.34375000e-03, 5.34895833e-03,
       5.35416666e-03, 5.35937500e-03, 5.36458333e-03, 5.36979166e-03,
       5.37500000e-03, 5.38020833e-03, 5.38541666e-03, 5.39062500e-03,
       5.39583333e-03, 5.40104166e-03, 5.40625000e-03, 5.41145833e-03,
       5.41666666e-03, 5.42187500e-03, 5.42708333e-03, 5.43229166e-03,
       5.43750000e-03, 5.44270833e-03, 5.44791666e-03, 5.45312500e-03,
       5.45833333e-03, 5.46354166e-03, 5.46875000e-03, 5.47395833e-03,
       5.47916666e-03, 5.48437500e-03, 5.48958333e-03, 5.49479166e-03,
       5.50000000e-03, 5.50520833e-03, 5.51041666e-03, 5.51562500e-03,
       5.52083333e-03, 5.52604166e-03, 5.53125000e-03, 5.53645833e-03,
       5.54166666e-03, 5.54687500e-03, 5.55208333e-03, 5.55729166e-03,
       5.56250000e-03, 5.56770833e-03, 5.57291666e-03, 5.57812500e-03,
       5.58333333e-03, 5.58854166e-03, 5.59375000e-03, 5.59895833e-03,
       5.60416666e-03, 5.60937500e-03, 5.61458333e-03, 5.61979166e-03,
       5.62500000e-03, 5.63020833e-03, 5.63541666e-03, 5.64062500e-03,
       5.64583333e-03, 5.65104166e-03, 5.65625000e-03, 5.66145833e-03,
       5.66666666e-03, 5.67187500e-03, 5.67708333e-03, 5.68229166e-03,
       5.68750000e-03, 5.69270833e-03, 5.69791666e-03, 5.70312500e-03,
       5.70833333e-03, 5.71354166e-03, 5.71875000e-03, 5.72395833e-03,
       5.72916666e-03, 5.73437500e-03, 5.73958333e-03, 5.74479166e-03,
       5.75000000e-03, 5.75520833e-03, 5.76041666e-03, 5.76562500e-03,
       5.77083333e-03, 5.77604166e-03, 5.78125000e-03, 5.78645833e-03,
       5.79166666e-03, 5.79687500e-03, 5.80208333e-03, 5.80729166e-03,
       5.81250000e-03, 5.81770833e-03, 5.82291666e-03, 5.82812500e-03,
       5.83333333e-03, 5.83854166e-03, 5.84375000e-03, 5.84895833e-03,
       5.85416666e-03, 5.85937500e-03, 5.86458333e-03, 5.86979166e-03,
       5.87500000e-03, 5.88020833e-03, 5.88541666e-03, 5.89062500e-03,
       5.89583333e-03, 5.90104166e-03, 5.90625000e-03, 5.91145833e-03,
       5.91666666e-03, 5.92187500e-03, 5.92708333e-03, 5.93229166e-03,
       5.93750000e-03, 5.94270833e-03, 5.94791666e-03, 5.95312500e-03,
       5.95833333e-03, 5.96354166e-03, 5.96875000e-03, 5.97395833e-03,
       5.97916666e-03, 5.98437500e-03, 5.98958333e-03, 5.99479166e-03,
       6.00000000e-03, 6.00520833e-03, 6.01041666e-03, 6.01562500e-03,
       6.02083333e-03, 6.02604166e-03, 6.03125000e-03, 6.03645833e-03,
       6.04166666e-03, 6.04687500e-03, 6.05208333e-03, 6.05729166e-03,
       6.06250000e-03, 6.06770833e-03, 6.07291666e-03, 6.07812500e-03,
       6.08333333e-03, 6.08854166e-03, 6.09375000e-03, 6.09895833e-03,
       6.10416666e-03, 6.10937500e-03, 6.11458333e-03, 6.11979166e-03,
       6.12500000e-03, 6.13020833e-03, 6.13541666e-03, 6.14062500e-03,
       6.14583333e-03, 6.15104166e-03, 6.15625000e-03, 6.16145833e-03,
       6.16666666e-03, 6.17187500e-03, 6.17708333e-03, 6.18229166e-03,
       6.18750000e-03, 6.19270833e-03, 6.19791666e-03, 6.20312500e-03,
       6.20833333e-03, 6.21354166e-03, 6.21875000e-03, 6.22395833e-03,
       6.22916666e-03, 6.23437500e-03, 6.23958333e-03, 6.24479166e-03,
       6.25000000e-03, 6.25520833e-03, 6.26041666e-03, 6.26562500e-03,
       6.27083333e-03, 6.27604166e-03, 6.28125000e-03, 6.28645833e-03,
       6.29166666e-03, 6.29687500e-03, 6.30208333e-03, 6.30729166e-03,
       6.31250000e-03, 6.31770833e-03, 6.32291666e-03, 6.32812500e-03,
       6.33333333e-03, 6.33854166e-03, 6.34375000e-03, 6.34895833e-03,
       6.35416666e-03, 6.35937500e-03, 6.36458333e-03, 6.36979166e-03,
       6.37500000e-03, 6.38020833e-03, 6.38541666e-03, 6.39062500e-03,
       6.39583333e-03, 6.40104166e-03, 6.40625000e-03, 6.41145833e-03,
       6.41666666e-03, 6.42187500e-03, 6.42708333e-03, 6.43229166e-03,
       6.43750000e-03, 6.44270833e-03, 6.44791666e-03, 6.45312500e-03,
       6.45833333e-03, 6.46354166e-03, 6.46875000e-03, 6.47395833e-03,
       6.47916666e-03, 6.48437500e-03, 6.48958333e-03, 6.49479166e-03,
       6.50000000e-03, 6.50520833e-03, 6.51041666e-03, 6.51562500e-03,
       6.52083333e-03, 6.52604166e-03, 6.53125000e-03, 6.53645833e-03,
       6.54166666e-03, 6.54687500e-03, 6.55208333e-03, 6.55729166e-03,
       6.56250000e-03, 6.56770833e-03, 6.57291666e-03, 6.57812500e-03,
       6.58333333e-03, 6.58854166e-03, 6.59375000e-03, 6.59895833e-03,
       6.60416666e-03, 6.60937500e-03, 6.61458333e-03, 6.61979166e-03,
       6.62500000e-03, 6.63020833e-03, 6.63541666e-03, 6.64062500e-03,
       6.64583333e-03, 6.65104166e-03, 6.65625000e-03, 6.66145833e-03,
       6.66666666e-03, 6.67187500e-03, 6.67708333e-03, 6.68229166e-03,
       6.68750000e-03, 6.69270833e-03, 6.69791666e-03, 6.70312500e-03,
       6.70833333e-03, 6.71354166e-03, 6.71875000e-03, 6.72395833e-03,
       6.72916666e-03, 6.73437500e-03, 6.73958333e-03, 6.74479166e-03,
       6.75000000e-03, 6.75520833e-03, 6.76041666e-03, 6.76562500e-03,
       6.77083333e-03, 6.77604166e-03, 6.78125000e-03, 6.78645833e-03,
       6.79166666e-03, 6.79687500e-03, 6.80208333e-03, 6.80729166e-03,
       6.81250000e-03, 6.81770833e-03, 6.82291666e-03, 6.82812500e-03,
       6.83333333e-03, 6.83854166e-03, 6.84375000e-03, 6.84895833e-03,
       6.85416666e-03, 6.85937500e-03, 6.86458333e-03, 6.86979166e-03,
       6.87500000e-03, 6.88020833e-03, 6.88541666e-03, 6.89062500e-03,
       6.89583333e-03, 6.90104166e-03, 6.90625000e-03, 6.91145833e-03,
       6.91666666e-03, 6.92187500e-03, 6.92708333e-03, 6.93229166e-03,
       6.93750000e-03, 6.94270833e-03, 6.94791666e-03, 6.95312500e-03,
       6.95833333e-03, 6.96354166e-03, 6.96875000e-03, 6.97395833e-03,
       6.97916666e-03, 6.98437500e-03, 6.98958333e-03, 6.99479166e-03,
       7.00000000e-03, 7.00520833e-03, 7.01041666e-03, 7.01562500e-03,
       7.02083333e-03, 7.02604166e-03, 7.03125000e-03, 7.03645833e-03,
       7.04166666e-03, 7.04687500e-03, 7.05208333e-03, 7.05729166e-03,
       7.06250000e-03, 7.06770833e-03, 7.07291666e-03, 7.07812500e-03,
       7.08333333e-03, 7.08854166e-03, 7.09375000e-03, 7.09895833e-03,
       7.10416666e-03, 7.10937500e-03, 7.11458333e-03, 7.11979166e-03,
       7.12500000e-03, 7.13020833e-03, 7.13541666e-03, 7.14062500e-03,
       7.14583333e-03, 7.15104166e-03, 7.15625000e-03, 7.16145833e-03,
       7.16666666e-03, 7.17187500e-03, 7.17708333e-03, 7.18229166e-03,
       7.18750000e-03, 7.19270833e-03, 7.19791666e-03, 7.20312500e-03,
       7.20833333e-03, 7.21354166e-03, 7.21875000e-03, 7.22395833e-03,
       7.22916666e-03, 7.23437500e-03, 7.23958333e-03, 7.24479166e-03,
       7.25000000e-03, 7.25520833e-03, 7.26041666e-03, 7.26562500e-03,
       7.27083333e-03, 7.27604166e-03, 7.28125000e-03, 7.28645833e-03,
       7.29166666e-03, 7.29687500e-03, 7.30208333e-03, 7.30729166e-03,
       7.31250000e-03, 7.31770833e-03, 7.32291666e-03, 7.32812500e-03,
       7.33333333e-03, 7.33854166e-03, 7.34375000e-03, 7.34895833e-03,
       7.35416666e-03, 7.35937500e-03, 7.36458333e-03, 7.36979166e-03,
       7.37500000e-03, 7.38020833e-03, 7.38541666e-03, 7.39062500e-03,
       7.39583333e-03, 7.40104166e-03, 7.40625000e-03, 7.41145833e-03,
       7.41666666e-03, 7.42187500e-03, 7.42708333e-03, 7.43229166e-03,
       7.43750000e-03, 7.44270833e-03, 7.44791666e-03, 7.45312500e-03,
       7.45833333e-03, 7.46354166e-03, 7.46875000e-03, 7.47395833e-03,
       7.47916666e-03, 7.48437500e-03, 7.48958333e-03, 7.49479166e-03,
       7.50000000e-03, 7.50520833e-03, 7.51041666e-03, 7.51562500e-03,
       7.52083333e-03, 7.52604166e-03, 7.53125000e-03, 7.53645833e-03,
       7.54166666e-03, 7.54687500e-03, 7.55208333e-03, 7.55729166e-03,
       7.56250000e-03, 7.56770833e-03, 7.57291666e-03, 7.57812500e-03,
       7.58333333e-03, 7.58854166e-03, 7.59375000e-03, 7.59895833e-03,
       7.60416666e-03, 7.60937500e-03, 7.61458333e-03, 7.61979166e-03,
       7.62500000e-03, 7.63020833e-03, 7.63541666e-03, 7.64062500e-03,
       7.64583333e-03, 7.65104166e-03, 7.65625000e-03, 7.66145833e-03,
       7.66666666e-03, 7.67187500e-03, 7.67708333e-03, 7.68229166e-03,
       7.68750000e-03, 7.69270833e-03, 7.69791666e-03, 7.70312500e-03,
       7.70833333e-03, 7.71354166e-03, 7.71875000e-03, 7.72395833e-03,
       7.72916666e-03, 7.73437500e-03, 7.73958333e-03, 7.74479166e-03,
       7.75000000e-03, 7.75520833e-03, 7.76041666e-03, 7.76562500e-03,
       7.77083333e-03, 7.77604166e-03, 7.78125000e-03, 7.78645833e-03,
       7.79166666e-03, 7.79687500e-03, 7.80208333e-03, 7.80729166e-03,
       7.81249999e-03, 7.81770833e-03, 7.82291666e-03, 7.82812499e-03,
       7.83333333e-03, 7.83854166e-03, 7.84374999e-03, 7.84895833e-03,
       7.85416666e-03, 7.85937499e-03, 7.86458333e-03, 7.86979166e-03,
       7.87499999e-03, 7.88020833e-03, 7.88541666e-03, 7.89062499e-03,
       7.89583333e-03, 7.90104166e-03, 7.90624999e-03, 7.91145833e-03,
       7.91666666e-03, 7.92187499e-03, 7.92708333e-03, 7.93229166e-03,
       7.93749999e-03, 7.94270833e-03, 7.94791666e-03, 7.95312499e-03,
       7.95833333e-03, 7.96354166e-03, 7.96874999e-03, 7.97395833e-03,
       7.97916666e-03, 7.98437499e-03, 7.98958333e-03, 7.99479166e-03,
       7.99999999e-03, 8.00520833e-03, 8.01041666e-03, 8.01562499e-03,
       8.02083333e-03, 8.02604166e-03, 8.03124999e-03, 8.03645833e-03,
       8.04166666e-03, 8.04687499e-03, 8.05208333e-03, 8.05729166e-03,
       8.06249999e-03, 8.06770833e-03, 8.07291666e-03, 8.07812499e-03,
       8.08333333e-03, 8.08854166e-03, 8.09374999e-03, 8.09895833e-03,
       8.10416666e-03, 8.10937499e-03, 8.11458333e-03, 8.11979166e-03,
       8.12499999e-03, 8.13020833e-03, 8.13541666e-03, 8.14062499e-03,
       8.14583333e-03, 8.15104166e-03, 8.15624999e-03, 8.16145833e-03,
       8.16666666e-03, 8.17187499e-03, 8.17708333e-03, 8.18229166e-03,
       8.18749999e-03, 8.19270833e-03, 8.19791666e-03, 8.20312499e-03,
       8.20833333e-03, 8.21354166e-03, 8.21874999e-03, 8.22395833e-03,
       8.22916666e-03, 8.23437499e-03, 8.23958333e-03, 8.24479166e-03,
       8.24999999e-03, 8.25520833e-03, 8.26041666e-03, 8.26562499e-03,
       8.27083333e-03, 8.27604166e-03, 8.28124999e-03, 8.28645833e-03,
       8.29166666e-03, 8.29687499e-03, 8.30208333e-03, 8.30729166e-03,
       8.31249999e-03, 8.31770833e-03, 8.32291666e-03, 8.32812499e-03,
       8.33333333e-03, 8.33854166e-03, 8.34374999e-03, 8.34895833e-03,
       8.35416666e-03, 8.35937499e-03, 8.36458333e-03, 8.36979166e-03,
       8.37499999e-03, 8.38020833e-03, 8.38541666e-03, 8.39062499e-03,
       8.39583333e-03, 8.40104166e-03, 8.40624999e-03, 8.41145833e-03,
       8.41666666e-03, 8.42187499e-03, 8.42708333e-03, 8.43229166e-03,
       8.43749999e-03, 8.44270833e-03, 8.44791666e-03, 8.45312499e-03,
       8.45833333e-03, 8.46354166e-03, 8.46874999e-03, 8.47395833e-03,
       8.47916666e-03, 8.48437499e-03, 8.48958333e-03, 8.49479166e-03,
       8.49999999e-03, 8.50520833e-03, 8.51041666e-03, 8.51562499e-03,
       8.52083333e-03, 8.52604166e-03, 8.53124999e-03, 8.53645833e-03,
       8.54166666e-03, 8.54687499e-03, 8.55208333e-03, 8.55729166e-03,
       8.56249999e-03, 8.56770833e-03, 8.57291666e-03, 8.57812499e-03,
       8.58333333e-03, 8.58854166e-03, 8.59374999e-03, 8.59895833e-03,
       8.60416666e-03, 8.60937499e-03, 8.61458333e-03, 8.61979166e-03,
       8.62499999e-03, 8.63020833e-03, 8.63541666e-03, 8.64062499e-03,
       8.64583333e-03, 8.65104166e-03, 8.65624999e-03, 8.66145833e-03,
       8.66666666e-03, 8.67187499e-03, 8.67708333e-03, 8.68229166e-03,
       8.68749999e-03, 8.69270833e-03, 8.69791666e-03, 8.70312499e-03,
       8.70833333e-03, 8.71354166e-03, 8.71874999e-03, 8.72395833e-03,
       8.72916666e-03, 8.73437499e-03, 8.73958333e-03, 8.74479166e-03,
       8.74999999e-03, 8.75520833e-03, 8.76041666e-03, 8.76562499e-03,
       8.77083333e-03, 8.77604166e-03, 8.78124999e-03, 8.78645833e-03,
       8.79166666e-03, 8.79687499e-03, 8.80208333e-03, 8.80729166e-03,
       8.81249999e-03, 8.81770833e-03, 8.82291666e-03, 8.82812499e-03,
       8.83333333e-03, 8.83854166e-03, 8.84374999e-03, 8.84895833e-03,
       8.85416666e-03, 8.85937499e-03, 8.86458333e-03, 8.86979166e-03,
       8.87499999e-03, 8.88020833e-03, 8.88541666e-03, 8.89062499e-03,
       8.89583333e-03, 8.90104166e-03, 8.90624999e-03, 8.91145833e-03,
       8.91666666e-03, 8.92187499e-03, 8.92708333e-03, 8.93229166e-03,
       8.93749999e-03, 8.94270833e-03, 8.94791666e-03, 8.95312499e-03,
       8.95833333e-03, 8.96354166e-03, 8.96874999e-03, 8.97395833e-03,
       8.97916666e-03, 8.98437499e-03, 8.98958333e-03, 8.99479166e-03,
       8.99999999e-03, 9.00520833e-03, 9.01041666e-03, 9.01562499e-03,
       9.02083333e-03, 9.02604166e-03, 9.03124999e-03, 9.03645833e-03,
       9.04166666e-03, 9.04687499e-03, 9.05208333e-03, 9.05729166e-03,
       9.06249999e-03, 9.06770833e-03, 9.07291666e-03, 9.07812499e-03,
       9.08333333e-03, 9.08854166e-03, 9.09374999e-03, 9.09895833e-03,
       9.10416666e-03, 9.10937499e-03, 9.11458333e-03, 9.11979166e-03,
       9.12499999e-03, 9.13020833e-03, 9.13541666e-03, 9.14062499e-03,
       9.14583333e-03, 9.15104166e-03, 9.15624999e-03, 9.16145833e-03,
       9.16666666e-03, 9.17187499e-03, 9.17708333e-03, 9.18229166e-03,
       9.18749999e-03, 9.19270833e-03, 9.19791666e-03, 9.20312499e-03,
       9.20833333e-03, 9.21354166e-03, 9.21874999e-03, 9.22395833e-03,
       9.22916666e-03, 9.23437499e-03, 9.23958333e-03, 9.24479166e-03,
       9.24999999e-03, 9.25520833e-03, 9.26041666e-03, 9.26562499e-03,
       9.27083333e-03, 9.27604166e-03, 9.28124999e-03, 9.28645833e-03,
       9.29166666e-03, 9.29687499e-03, 9.30208333e-03, 9.30729166e-03,
       9.31249999e-03, 9.31770833e-03, 9.32291666e-03, 9.32812499e-03,
       9.33333333e-03, 9.33854166e-03, 9.34374999e-03, 9.34895833e-03,
       9.35416666e-03, 9.35937499e-03, 9.36458333e-03, 9.36979166e-03,
       9.37499999e-03, 9.38020833e-03, 9.38541666e-03, 9.39062499e-03,
       9.39583333e-03, 9.40104166e-03, 9.40624999e-03, 9.41145833e-03,
       9.41666666e-03, 9.42187499e-03, 9.42708333e-03, 9.43229166e-03,
       9.43749999e-03, 9.44270833e-03, 9.44791666e-03, 9.45312499e-03,
       9.45833333e-03, 9.46354166e-03, 9.46874999e-03, 9.47395833e-03,
       9.47916666e-03, 9.48437499e-03, 9.48958333e-03, 9.49479166e-03,
       9.49999999e-03, 9.50520833e-03, 9.51041666e-03, 9.51562499e-03,
       9.52083333e-03, 9.52604166e-03, 9.53124999e-03, 9.53645833e-03,
       9.54166666e-03, 9.54687499e-03, 9.55208333e-03, 9.55729166e-03,
       9.56249999e-03, 9.56770833e-03, 9.57291666e-03, 9.57812499e-03,
       9.58333333e-03, 9.58854166e-03, 9.59374999e-03, 9.59895833e-03,
       9.60416666e-03, 9.60937499e-03, 9.61458333e-03, 9.61979166e-03,
       9.62499999e-03, 9.63020833e-03, 9.63541666e-03, 9.64062499e-03,
       9.64583333e-03, 9.65104166e-03, 9.65624999e-03, 9.66145833e-03,
       9.66666666e-03, 9.67187499e-03, 9.67708333e-03, 9.68229166e-03,
       9.68749999e-03, 9.69270833e-03, 9.69791666e-03, 9.70312499e-03,
       9.70833333e-03, 9.71354166e-03, 9.71874999e-03, 9.72395833e-03,
       9.72916666e-03, 9.73437499e-03, 9.73958333e-03, 9.74479166e-03,
       9.74999999e-03, 9.75520833e-03, 9.76041666e-03, 9.76562499e-03,
       9.77083333e-03, 9.77604166e-03, 9.78124999e-03, 9.78645833e-03,
       9.79166666e-03, 9.79687499e-03, 9.80208333e-03, 9.80729166e-03,
       9.81249999e-03, 9.81770833e-03, 9.82291666e-03, 9.82812499e-03,
       9.83333333e-03, 9.83854166e-03, 9.84374999e-03, 9.84895833e-03,
       9.85416666e-03, 9.85937499e-03, 9.86458333e-03, 9.86979166e-03,
       9.87499999e-03, 9.88020833e-03, 9.88541666e-03, 9.89062499e-03,
       9.89583333e-03, 9.90104166e-03, 9.90624999e-03, 9.91145833e-03,
       9.91666666e-03, 9.92187499e-03, 9.92708333e-03, 9.93229166e-03,
       9.93749999e-03, 9.94270833e-03, 9.94791666e-03, 9.95312499e-03,
       9.95833333e-03, 9.96354166e-03, 9.96874999e-03, 9.97395833e-03,
       9.97916666e-03, 9.98437499e-03, 9.98958333e-03, 9.99479166e-03,
       9.99999999e-03, 1.00052083e-02, 1.00104167e-02, 1.00156250e-02,
       1.00208333e-02, 1.00260417e-02, 1.00312500e-02, 1.00364583e-02,
       1.00416667e-02, 1.00468750e-02, 1.00520833e-02, 1.00572917e-02,
       1.00625000e-02, 1.00677083e-02, 1.00729167e-02, 1.00781250e-02,
       1.00833333e-02, 1.00885417e-02, 1.00937500e-02, 1.00989583e-02,
       1.01041667e-02, 1.01093750e-02, 1.01145833e-02, 1.01197917e-02,
       1.01250000e-02, 1.01302083e-02, 1.01354167e-02, 1.01406250e-02,
       1.01458333e-02, 1.01510417e-02, 1.01562500e-02, 1.01614583e-02,
       1.01666667e-02, 1.01718750e-02, 1.01770833e-02, 1.01822917e-02,
       1.01875000e-02, 1.01927083e-02, 1.01979167e-02, 1.02031250e-02,
       1.02083333e-02, 1.02135417e-02, 1.02187500e-02, 1.02239583e-02,
       1.02291667e-02, 1.02343750e-02, 1.02395833e-02, 1.02447917e-02,
       1.02500000e-02, 1.02552083e-02, 1.02604167e-02, 1.02656250e-02,
       1.02708333e-02, 1.02760417e-02, 1.02812500e-02, 1.02864583e-02,
       1.02916667e-02, 1.02968750e-02, 1.03020833e-02, 1.03072917e-02,
       1.03125000e-02, 1.03177083e-02, 1.03229167e-02, 1.03281250e-02,
       1.03333333e-02, 1.03385417e-02, 1.03437500e-02, 1.03489583e-02,
       1.03541667e-02, 1.03593750e-02, 1.03645833e-02, 1.03697917e-02,
       1.03750000e-02, 1.03802083e-02, 1.03854167e-02, 1.03906250e-02,
       1.03958333e-02, 1.04010417e-02, 1.04062500e-02, 1.04114583e-02,
       1.04166667e-02, 1.04218750e-02, 1.04270833e-02, 1.04322917e-02,
       1.04375000e-02, 1.04427083e-02, 1.04479167e-02, 1.04531250e-02,
       1.04583333e-02, 1.04635417e-02, 1.04687500e-02, 1.04739583e-02,
       1.04791667e-02, 1.04843750e-02, 1.04895833e-02, 1.04947917e-02,
       1.05000000e-02, 1.05052083e-02, 1.05104167e-02, 1.05156250e-02,
       1.05208333e-02, 1.05260417e-02, 1.05312500e-02, 1.05364583e-02,
       1.05416667e-02, 1.05468750e-02, 1.05520833e-02, 1.05572917e-02,
       1.05625000e-02, 1.05677083e-02, 1.05729167e-02, 1.05781250e-02,
       1.05833333e-02, 1.05885417e-02, 1.05937500e-02, 1.05989583e-02,
       1.06041667e-02, 1.06093750e-02, 1.06145833e-02, 1.06197917e-02,
       1.06250000e-02, 1.06302083e-02, 1.06354167e-02, 1.06406250e-02,
       1.06458333e-02, 1.06510417e-02, 1.06562500e-02, 1.06614583e-02,
       1.06666667e-02, 1.06718750e-02, 1.06770833e-02, 1.06822917e-02,
       1.06875000e-02, 1.06927083e-02, 1.06979167e-02, 1.07031250e-02,
       1.07083333e-02, 1.07135417e-02, 1.07187500e-02, 1.07239583e-02,
       1.07291667e-02, 1.07343750e-02, 1.07395833e-02, 1.07447917e-02,
       1.07500000e-02, 1.07552083e-02, 1.07604167e-02, 1.07656250e-02,
       1.07708333e-02, 1.07760417e-02, 1.07812500e-02, 1.07864583e-02,
       1.07916667e-02, 1.07968750e-02, 1.08020833e-02, 1.08072917e-02,
       1.08125000e-02, 1.08177083e-02, 1.08229167e-02, 1.08281250e-02,
       1.08333333e-02, 1.08385417e-02, 1.08437500e-02, 1.08489583e-02,
       1.08541667e-02, 1.08593750e-02, 1.08645833e-02, 1.08697917e-02,
       1.08750000e-02, 1.08802083e-02, 1.08854167e-02, 1.08906250e-02,
       1.08958333e-02, 1.09010417e-02, 1.09062500e-02, 1.09114583e-02,
       1.09166667e-02, 1.09218750e-02, 1.09270833e-02, 1.09322917e-02,
       1.09375000e-02, 1.09427083e-02, 1.09479167e-02, 1.09531250e-02,
       1.09583333e-02, 1.09635417e-02, 1.09687500e-02, 1.09739583e-02,
       1.09791667e-02, 1.09843750e-02, 1.09895833e-02, 1.09947917e-02,
       1.10000000e-02, 1.10052083e-02, 1.10104167e-02, 1.10156250e-02,
       1.10208333e-02, 1.10260417e-02, 1.10312500e-02, 1.10364583e-02,
       1.10416667e-02, 1.10468750e-02, 1.10520833e-02, 1.10572917e-02,
       1.10625000e-02, 1.10677083e-02, 1.10729167e-02, 1.10781250e-02,
       1.10833333e-02, 1.10885417e-02, 1.10937500e-02, 1.10989583e-02,
       1.11041667e-02, 1.11093750e-02, 1.11145833e-02, 1.11197917e-02,
       1.11250000e-02, 1.11302083e-02, 1.11354167e-02, 1.11406250e-02,
       1.11458333e-02, 1.11510417e-02, 1.11562500e-02, 1.11614583e-02,
       1.11666667e-02, 1.11718750e-02, 1.11770833e-02, 1.11822917e-02,
       1.11875000e-02, 1.11927083e-02, 1.11979167e-02, 1.12031250e-02,
       1.12083333e-02, 1.12135417e-02, 1.12187500e-02, 1.12239583e-02,
       1.12291667e-02, 1.12343750e-02, 1.12395833e-02, 1.12447917e-02,
       1.12500000e-02, 1.12552083e-02, 1.12604167e-02, 1.12656250e-02,
       1.12708333e-02, 1.12760417e-02, 1.12812500e-02, 1.12864583e-02,
       1.12916667e-02, 1.12968750e-02, 1.13020833e-02, 1.13072917e-02,
       1.13125000e-02, 1.13177083e-02, 1.13229167e-02, 1.13281250e-02,
       1.13333333e-02, 1.13385417e-02, 1.13437500e-02, 1.13489583e-02,
       1.13541667e-02, 1.13593750e-02, 1.13645833e-02, 1.13697917e-02,
       1.13750000e-02, 1.13802083e-02, 1.13854167e-02, 1.13906250e-02,
       1.13958333e-02, 1.14010417e-02, 1.14062500e-02, 1.14114583e-02,
       1.14166667e-02, 1.14218750e-02, 1.14270833e-02, 1.14322917e-02,
       1.14375000e-02, 1.14427083e-02, 1.14479167e-02, 1.14531250e-02,
       1.14583333e-02, 1.14635417e-02, 1.14687500e-02, 1.14739583e-02,
       1.14791667e-02, 1.14843750e-02, 1.14895833e-02, 1.14947917e-02,
       1.15000000e-02, 1.15052083e-02, 1.15104167e-02, 1.15156250e-02,
       1.15208333e-02, 1.15260417e-02, 1.15312500e-02, 1.15364583e-02,
       1.15416667e-02, 1.15468750e-02, 1.15520833e-02, 1.15572917e-02,
       1.15625000e-02, 1.15677083e-02, 1.15729167e-02, 1.15781250e-02,
       1.15833333e-02, 1.15885417e-02, 1.15937500e-02, 1.15989583e-02,
       1.16041667e-02, 1.16093750e-02, 1.16145833e-02, 1.16197917e-02,
       1.16250000e-02, 1.16302083e-02, 1.16354167e-02, 1.16406250e-02,
       1.16458333e-02, 1.16510417e-02, 1.16562500e-02, 1.16614583e-02,
       1.16666667e-02, 1.16718750e-02, 1.16770833e-02, 1.16822917e-02,
       1.16875000e-02, 1.16927083e-02, 1.16979167e-02, 1.17031250e-02,
       1.17083333e-02, 1.17135417e-02, 1.17187500e-02, 1.17239583e-02,
       1.17291667e-02, 1.17343750e-02, 1.17395833e-02, 1.17447917e-02,
       1.17500000e-02, 1.17552083e-02, 1.17604167e-02, 1.17656250e-02,
       1.17708333e-02, 1.17760417e-02, 1.17812500e-02, 1.17864583e-02,
       1.17916667e-02, 1.17968750e-02, 1.18020833e-02, 1.18072917e-02,
       1.18125000e-02, 1.18177083e-02, 1.18229167e-02, 1.18281250e-02,
       1.18333333e-02, 1.18385417e-02, 1.18437500e-02, 1.18489583e-02,
       1.18541667e-02, 1.18593750e-02, 1.18645833e-02, 1.18697917e-02,
       1.18750000e-02, 1.18802083e-02, 1.18854167e-02, 1.18906250e-02,
       1.18958333e-02, 1.19010417e-02, 1.19062500e-02, 1.19114583e-02,
       1.19166667e-02, 1.19218750e-02, 1.19270833e-02, 1.19322917e-02,
       1.19375000e-02, 1.19427083e-02, 1.19479167e-02, 1.19531250e-02,
       1.19583333e-02, 1.19635417e-02, 1.19687500e-02, 1.19739583e-02,
       1.19791667e-02, 1.19843750e-02, 1.19895833e-02, 1.19947917e-02,
       1.20000000e-02, 1.20052083e-02, 1.20104167e-02, 1.20156250e-02,
       1.20208333e-02, 1.20260417e-02, 1.20312500e-02, 1.20364583e-02,
       1.20416667e-02, 1.20468750e-02, 1.20520833e-02, 1.20572917e-02,
       1.20625000e-02, 1.20677083e-02, 1.20729167e-02, 1.20781250e-02,
       1.20833333e-02, 1.20885417e-02, 1.20937500e-02, 1.20989583e-02,
       1.21041667e-02, 1.21093750e-02, 1.21145833e-02, 1.21197917e-02,
       1.21250000e-02, 1.21302083e-02, 1.21354167e-02, 1.21406250e-02,
       1.21458333e-02, 1.21510417e-02, 1.21562500e-02, 1.21614583e-02,
       1.21666667e-02, 1.21718750e-02, 1.21770833e-02, 1.21822917e-02,
       1.21875000e-02, 1.21927083e-02, 1.21979167e-02, 1.22031250e-02,
       1.22083333e-02, 1.22135417e-02, 1.22187500e-02, 1.22239583e-02,
       1.22291667e-02, 1.22343750e-02, 1.22395833e-02, 1.22447917e-02,
       1.22500000e-02, 1.22552083e-02, 1.22604167e-02, 1.22656250e-02,
       1.22708333e-02, 1.22760417e-02, 1.22812500e-02, 1.22864583e-02,
       1.22916667e-02, 1.22968750e-02, 1.23020833e-02, 1.23072917e-02,
       1.23125000e-02, 1.23177083e-02, 1.23229167e-02, 1.23281250e-02,
       1.23333333e-02, 1.23385417e-02, 1.23437500e-02, 1.23489583e-02,
       1.23541667e-02, 1.23593750e-02, 1.23645833e-02, 1.23697917e-02,
       1.23750000e-02, 1.23802083e-02, 1.23854167e-02, 1.23906250e-02,
       1.23958333e-02, 1.24010417e-02, 1.24062500e-02, 1.24114583e-02,
       1.24166667e-02, 1.24218750e-02, 1.24270833e-02, 1.24322917e-02,
       1.24375000e-02, 1.24427083e-02, 1.24479167e-02, 1.24531250e-02,
       1.24583333e-02, 1.24635417e-02, 1.24687500e-02, 1.24739583e-02,
       1.24791667e-02, 1.24843750e-02, 1.24895833e-02, 1.24947917e-02,
       1.25000000e-02, 1.25052083e-02, 1.25104167e-02, 1.25156250e-02,
       1.25208333e-02, 1.25260417e-02, 1.25312500e-02, 1.25364583e-02,
       1.25416667e-02, 1.25468750e-02, 1.25520833e-02, 1.25572917e-02,
       1.25625000e-02, 1.25677083e-02, 1.25729167e-02, 1.25781250e-02,
       1.25833333e-02, 1.25885417e-02, 1.25937500e-02, 1.25989583e-02,
       1.26041667e-02, 1.26093750e-02, 1.26145833e-02, 1.26197917e-02,
       1.26250000e-02, 1.26302083e-02, 1.26354167e-02, 1.26406250e-02,
       1.26458333e-02, 1.26510417e-02, 1.26562500e-02, 1.26614583e-02,
       1.26666667e-02, 1.26718750e-02, 1.26770833e-02, 1.26822917e-02,
       1.26875000e-02, 1.26927083e-02, 1.26979167e-02, 1.27031250e-02,
       1.27083333e-02, 1.27135417e-02, 1.27187500e-02, 1.27239583e-02,
       1.27291667e-02, 1.27343750e-02, 1.27395833e-02, 1.27447917e-02,
       1.27500000e-02, 1.27552083e-02, 1.27604167e-02, 1.27656250e-02,
       1.27708333e-02, 1.27760417e-02, 1.27812500e-02, 1.27864583e-02,
       1.27916667e-02, 1.27968750e-02, 1.28020833e-02, 1.28072917e-02,
       1.28125000e-02, 1.28177083e-02, 1.28229167e-02, 1.28281250e-02,
       1.28333333e-02, 1.28385417e-02, 1.28437500e-02, 1.28489583e-02,
       1.28541667e-02, 1.28593750e-02, 1.28645833e-02, 1.28697917e-02,
       1.28750000e-02, 1.28802083e-02, 1.28854167e-02, 1.28906250e-02,
       1.28958333e-02, 1.29010417e-02, 1.29062500e-02, 1.29114583e-02,
       1.29166667e-02, 1.29218750e-02, 1.29270833e-02, 1.29322917e-02,
       1.29375000e-02, 1.29427083e-02, 1.29479167e-02, 1.29531250e-02,
       1.29583333e-02, 1.29635417e-02, 1.29687500e-02, 1.29739583e-02,
       1.29791667e-02, 1.29843750e-02, 1.29895833e-02, 1.29947917e-02,
       1.30000000e-02, 1.30052083e-02, 1.30104167e-02, 1.30156250e-02,
       1.30208333e-02, 1.30260417e-02, 1.30312500e-02, 1.30364583e-02,
       1.30416667e-02, 1.30468750e-02, 1.30520833e-02, 1.30572917e-02,
       1.30625000e-02, 1.30677083e-02, 1.30729167e-02, 1.30781250e-02,
       1.30833333e-02, 1.30885417e-02, 1.30937500e-02, 1.30989583e-02,
       1.31041667e-02, 1.31093750e-02, 1.31145833e-02, 1.31197917e-02,
       1.31250000e-02, 1.31302083e-02, 1.31354167e-02, 1.31406250e-02,
       1.31458333e-02, 1.31510417e-02, 1.31562500e-02, 1.31614583e-02,
       1.31666667e-02, 1.31718750e-02, 1.31770833e-02, 1.31822917e-02,
       1.31875000e-02, 1.31927083e-02, 1.31979167e-02, 1.32031250e-02,
       1.32083333e-02, 1.32135417e-02, 1.32187500e-02, 1.32239583e-02,
       1.32291667e-02, 1.32343750e-02, 1.32395833e-02, 1.32447917e-02,
       1.32500000e-02, 1.32552083e-02, 1.32604167e-02, 1.32656250e-02,
       1.32708333e-02, 1.32760417e-02, 1.32812500e-02, 1.32864583e-02,
       1.32916667e-02, 1.32968750e-02, 1.33020833e-02, 1.33072917e-02,
       1.33125000e-02, 1.33177083e-02, 1.33229167e-02, 1.33281250e-02,
       1.33333333e-02, 1.33385417e-02, 1.33437500e-02, 1.33489583e-02,
       1.33541667e-02, 1.33593750e-02, 1.33645833e-02, 1.33697917e-02,
       1.33750000e-02, 1.33802083e-02, 1.33854167e-02, 1.33906250e-02,
       1.33958333e-02, 1.34010417e-02, 1.34062500e-02, 1.34114583e-02,
       1.34166667e-02, 1.34218750e-02, 1.34270833e-02, 1.34322917e-02,
       1.34375000e-02, 1.34427083e-02, 1.34479167e-02, 1.34531250e-02,
       1.34583333e-02, 1.34635417e-02, 1.34687500e-02, 1.34739583e-02,
       1.34791667e-02, 1.34843750e-02, 1.34895833e-02, 1.34947917e-02,
       1.35000000e-02, 1.35052083e-02, 1.35104167e-02, 1.35156250e-02,
       1.35208333e-02, 1.35260417e-02, 1.35312500e-02, 1.35364583e-02,
       1.35416667e-02, 1.35468750e-02, 1.35520833e-02, 1.35572917e-02,
       1.35625000e-02, 1.35677083e-02, 1.35729167e-02, 1.35781250e-02,
       1.35833333e-02, 1.35885417e-02, 1.35937500e-02, 1.35989583e-02,
       1.36041667e-02, 1.36093750e-02, 1.36145833e-02, 1.36197917e-02,
       1.36250000e-02, 1.36302083e-02, 1.36354167e-02, 1.36406250e-02,
       1.36458333e-02, 1.36510417e-02, 1.36562500e-02, 1.36614583e-02,
       1.36666667e-02, 1.36718750e-02, 1.36770833e-02, 1.36822917e-02,
       1.36875000e-02, 1.36927083e-02, 1.36979167e-02, 1.37031250e-02,
       1.37083333e-02, 1.37135417e-02, 1.37187500e-02, 1.37239583e-02,
       1.37291667e-02, 1.37343750e-02, 1.37395833e-02, 1.37447917e-02,
       1.37500000e-02, 1.37552083e-02, 1.37604167e-02, 1.37656250e-02,
       1.37708333e-02, 1.37760417e-02, 1.37812500e-02, 1.37864583e-02,
       1.37916667e-02, 1.37968750e-02, 1.38020833e-02, 1.38072917e-02,
       1.38125000e-02, 1.38177083e-02, 1.38229167e-02, 1.38281250e-02,
       1.38333333e-02, 1.38385417e-02, 1.38437500e-02, 1.38489583e-02,
       1.38541667e-02, 1.38593750e-02, 1.38645833e-02, 1.38697917e-02,
       1.38750000e-02, 1.38802083e-02, 1.38854167e-02, 1.38906250e-02,
       1.38958333e-02, 1.39010417e-02, 1.39062500e-02, 1.39114583e-02,
       1.39166667e-02, 1.39218750e-02, 1.39270833e-02, 1.39322917e-02,
       1.39375000e-02, 1.39427083e-02, 1.39479167e-02, 1.39531250e-02,
       1.39583333e-02, 1.39635417e-02, 1.39687500e-02, 1.39739583e-02,
       1.39791667e-02, 1.39843750e-02, 1.39895833e-02, 1.39947917e-02,
       1.40000000e-02, 1.40052083e-02, 1.40104167e-02, 1.40156250e-02,
       1.40208333e-02, 1.40260417e-02, 1.40312500e-02, 1.40364583e-02,
       1.40416667e-02, 1.40468750e-02, 1.40520833e-02, 1.40572917e-02,
       1.40625000e-02, 1.40677083e-02, 1.40729167e-02, 1.40781250e-02,
       1.40833333e-02, 1.40885417e-02, 1.40937500e-02, 1.40989583e-02,
       1.41041667e-02, 1.41093750e-02, 1.41145833e-02, 1.41197917e-02,
       1.41250000e-02, 1.41302083e-02, 1.41354167e-02, 1.41406250e-02,
       1.41458333e-02, 1.41510417e-02, 1.41562500e-02, 1.41614583e-02,
       1.41666667e-02, 1.41718750e-02, 1.41770833e-02, 1.41822917e-02,
       1.41875000e-02, 1.41927083e-02, 1.41979167e-02, 1.42031250e-02,
       1.42083333e-02, 1.42135417e-02, 1.42187500e-02, 1.42239583e-02,
       1.42291667e-02, 1.42343750e-02, 1.42395833e-02, 1.42447917e-02,
       1.42500000e-02, 1.42552083e-02, 1.42604167e-02, 1.42656250e-02,
       1.42708333e-02, 1.42760417e-02, 1.42812500e-02, 1.42864583e-02,
       1.42916667e-02, 1.42968750e-02, 1.43020833e-02, 1.43072917e-02,
       1.43125000e-02, 1.43177083e-02, 1.43229167e-02, 1.43281250e-02,
       1.43333333e-02, 1.43385417e-02, 1.43437500e-02, 1.43489583e-02,
       1.43541667e-02, 1.43593750e-02, 1.43645833e-02, 1.43697917e-02,
       1.43750000e-02, 1.43802083e-02, 1.43854167e-02, 1.43906250e-02,
       1.43958333e-02, 1.44010417e-02, 1.44062500e-02, 1.44114583e-02,
       1.44166667e-02, 1.44218750e-02, 1.44270833e-02, 1.44322917e-02,
       1.44375000e-02, 1.44427083e-02, 1.44479167e-02, 1.44531250e-02,
       1.44583333e-02, 1.44635417e-02, 1.44687500e-02, 1.44739583e-02,
       1.44791667e-02, 1.44843750e-02, 1.44895833e-02, 1.44947917e-02,
       1.45000000e-02, 1.45052083e-02, 1.45104167e-02, 1.45156250e-02,
       1.45208333e-02, 1.45260417e-02, 1.45312500e-02, 1.45364583e-02,
       1.45416667e-02, 1.45468750e-02, 1.45520833e-02, 1.45572917e-02,
       1.45625000e-02, 1.45677083e-02, 1.45729167e-02, 1.45781250e-02,
       1.45833333e-02, 1.45885417e-02, 1.45937500e-02, 1.45989583e-02,
       1.46041667e-02, 1.46093750e-02, 1.46145833e-02, 1.46197917e-02,
       1.46250000e-02, 1.46302083e-02, 1.46354167e-02, 1.46406250e-02,
       1.46458333e-02, 1.46510417e-02, 1.46562500e-02, 1.46614583e-02,
       1.46666667e-02, 1.46718750e-02, 1.46770833e-02, 1.46822917e-02,
       1.46875000e-02, 1.46927083e-02, 1.46979167e-02, 1.47031250e-02,
       1.47083333e-02, 1.47135417e-02, 1.47187500e-02, 1.47239583e-02,
       1.47291667e-02, 1.47343750e-02, 1.47395833e-02, 1.47447917e-02,
       1.47500000e-02, 1.47552083e-02, 1.47604167e-02, 1.47656250e-02,
       1.47708333e-02, 1.47760417e-02, 1.47812500e-02, 1.47864583e-02,
       1.47916667e-02, 1.47968750e-02, 1.48020833e-02, 1.48072917e-02,
       1.48125000e-02, 1.48177083e-02, 1.48229167e-02, 1.48281250e-02,
       1.48333333e-02, 1.48385417e-02, 1.48437500e-02, 1.48489583e-02,
       1.48541667e-02, 1.48593750e-02, 1.48645833e-02, 1.48697917e-02,
       1.48750000e-02, 1.48802083e-02, 1.48854167e-02, 1.48906250e-02,
       1.48958333e-02, 1.49010417e-02, 1.49062500e-02, 1.49114583e-02,
       1.49166667e-02, 1.49218750e-02, 1.49270833e-02, 1.49322917e-02,
       1.49375000e-02, 1.49427083e-02, 1.49479167e-02, 1.49531250e-02,
       1.49583333e-02, 1.49635417e-02, 1.49687500e-02, 1.49739583e-02,
       1.49791667e-02, 1.49843750e-02, 1.49895833e-02, 1.49947917e-02,
       1.50000000e-02, 1.50052083e-02, 1.50104167e-02, 1.50156250e-02,
       1.50208333e-02, 1.50260417e-02, 1.50312500e-02, 1.50364583e-02,
       1.50416667e-02, 1.50468750e-02, 1.50520833e-02, 1.50572917e-02,
       1.50625000e-02, 1.50677083e-02, 1.50729167e-02, 1.50781250e-02,
       1.50833333e-02, 1.50885417e-02, 1.50937500e-02, 1.50989583e-02,
       1.51041667e-02, 1.51093750e-02, 1.51145833e-02, 1.51197917e-02,
       1.51250000e-02, 1.51302083e-02, 1.51354167e-02, 1.51406250e-02,
       1.51458333e-02, 1.51510417e-02, 1.51562500e-02, 1.51614583e-02,
       1.51666667e-02, 1.51718750e-02, 1.51770833e-02, 1.51822917e-02,
       1.51875000e-02, 1.51927083e-02, 1.51979167e-02, 1.52031250e-02,
       1.52083333e-02, 1.52135417e-02, 1.52187500e-02, 1.52239583e-02,
       1.52291667e-02, 1.52343750e-02, 1.52395833e-02, 1.52447917e-02,
       1.52500000e-02, 1.52552083e-02, 1.52604167e-02, 1.52656250e-02,
       1.52708333e-02, 1.52760417e-02, 1.52812500e-02, 1.52864583e-02,
       1.52916667e-02, 1.52968750e-02, 1.53020833e-02, 1.53072917e-02,
       1.53125000e-02, 1.53177083e-02, 1.53229167e-02, 1.53281250e-02,
       1.53333333e-02, 1.53385417e-02, 1.53437500e-02, 1.53489583e-02,
       1.53541667e-02, 1.53593750e-02, 1.53645833e-02, 1.53697917e-02,
       1.53750000e-02, 1.53802083e-02, 1.53854167e-02, 1.53906250e-02,
       1.53958333e-02, 1.54010417e-02, 1.54062500e-02, 1.54114583e-02,
       1.54166667e-02, 1.54218750e-02, 1.54270833e-02, 1.54322917e-02,
       1.54375000e-02, 1.54427083e-02, 1.54479167e-02, 1.54531250e-02,
       1.54583333e-02, 1.54635417e-02, 1.54687500e-02, 1.54739583e-02,
       1.54791667e-02, 1.54843750e-02, 1.54895833e-02, 1.54947917e-02,
       1.55000000e-02, 1.55052083e-02, 1.55104167e-02, 1.55156250e-02,
       1.55208333e-02, 1.55260417e-02, 1.55312500e-02, 1.55364583e-02,
       1.55416667e-02, 1.55468750e-02, 1.55520833e-02, 1.55572917e-02,
       1.55625000e-02, 1.55677083e-02, 1.55729167e-02, 1.55781250e-02,
       1.55833333e-02, 1.55885417e-02, 1.55937500e-02, 1.55989583e-02,
       1.56041667e-02, 1.56093750e-02, 1.56145833e-02, 1.56197917e-02,
       1.56250000e-02, 1.56302083e-02, 1.56354167e-02, 1.56406250e-02,
       1.56458333e-02, 1.56510417e-02, 1.56562500e-02, 1.56614583e-02,
       1.56666667e-02, 1.56718750e-02, 1.56770833e-02, 1.56822917e-02,
       1.56875000e-02, 1.56927083e-02, 1.56979167e-02, 1.57031250e-02,
       1.57083333e-02, 1.57135417e-02, 1.57187500e-02, 1.57239583e-02,
       1.57291667e-02, 1.57343750e-02, 1.57395833e-02, 1.57447917e-02,
       1.57500000e-02, 1.57552083e-02, 1.57604167e-02, 1.57656250e-02,
       1.57708333e-02, 1.57760417e-02, 1.57812500e-02, 1.57864583e-02,
       1.57916667e-02, 1.57968750e-02, 1.58020833e-02, 1.58072917e-02,
       1.58125000e-02, 1.58177083e-02, 1.58229167e-02, 1.58281250e-02,
       1.58333333e-02, 1.58385417e-02, 1.58437500e-02, 1.58489583e-02,
       1.58541667e-02, 1.58593750e-02, 1.58645833e-02, 1.58697917e-02,
       1.58750000e-02, 1.58802083e-02, 1.58854167e-02, 1.58906250e-02,
       1.58958333e-02, 1.59010417e-02, 1.59062500e-02, 1.59114583e-02,
       1.59166667e-02, 1.59218750e-02, 1.59270833e-02, 1.59322917e-02,
       1.59375000e-02, 1.59427083e-02, 1.59479167e-02, 1.59531250e-02,
       1.59583333e-02, 1.59635417e-02, 1.59687500e-02, 1.59739583e-02,
       1.59791667e-02, 1.59843750e-02, 1.59895833e-02, 1.59947917e-02,
       1.60000000e-02, 1.60052083e-02, 1.60104167e-02, 1.60156250e-02,
       1.60208333e-02, 1.60260417e-02, 1.60312500e-02, 1.60364583e-02,
       1.60416667e-02, 1.60468750e-02, 1.60520833e-02, 1.60572917e-02,
       1.60625000e-02, 1.60677083e-02, 1.60729167e-02, 1.60781250e-02,
       1.60833333e-02, 1.60885417e-02, 1.60937500e-02, 1.60989583e-02,
       1.61041667e-02, 1.61093750e-02, 1.61145833e-02, 1.61197917e-02,
       1.61250000e-02, 1.61302083e-02, 1.61354167e-02, 1.61406250e-02,
       1.61458333e-02, 1.61510417e-02, 1.61562500e-02, 1.61614583e-02,
       1.61666667e-02, 1.61718750e-02, 1.61770833e-02, 1.61822917e-02,
       1.61875000e-02, 1.61927083e-02, 1.61979167e-02, 1.62031250e-02,
       1.62083333e-02, 1.62135417e-02, 1.62187500e-02, 1.62239583e-02,
       1.62291667e-02, 1.62343750e-02, 1.62395833e-02, 1.62447917e-02,
       1.62500000e-02, 1.62552083e-02, 1.62604167e-02, 1.62656250e-02,
       1.62708333e-02, 1.62760417e-02, 1.62812500e-02, 1.62864583e-02,
       1.62916667e-02, 1.62968750e-02, 1.63020833e-02, 1.63072917e-02,
       1.63125000e-02, 1.63177083e-02, 1.63229167e-02, 1.63281250e-02,
       1.63333333e-02, 1.63385417e-02, 1.63437500e-02, 1.63489583e-02,
       1.63541667e-02, 1.63593750e-02, 1.63645833e-02, 1.63697917e-02,
       1.63750000e-02, 1.63802083e-02, 1.63854167e-02, 1.63906250e-02,
       1.63958333e-02, 1.64010417e-02, 1.64062500e-02, 1.64114583e-02,
       1.64166667e-02, 1.64218750e-02, 1.64270833e-02, 1.64322917e-02,
       1.64375000e-02, 1.64427083e-02, 1.64479167e-02, 1.64531250e-02,
       1.64583333e-02, 1.64635417e-02, 1.64687500e-02, 1.64739583e-02,
       1.64791667e-02, 1.64843750e-02, 1.64895833e-02, 1.64947917e-02,
       1.65000000e-02, 1.65052083e-02, 1.65104167e-02, 1.65156250e-02,
       1.65208333e-02, 1.65260417e-02, 1.65312500e-02, 1.65364583e-02,
       1.65416667e-02, 1.65468750e-02, 1.65520833e-02, 1.65572917e-02,
       1.65625000e-02, 1.65677083e-02, 1.65729167e-02, 1.65781250e-02,
       1.65833333e-02, 1.65885417e-02, 1.65937500e-02, 1.65989583e-02,
       1.66041667e-02, 1.66093750e-02, 1.66145833e-02, 1.66197917e-02,
       1.66250000e-02, 1.66302083e-02, 1.66354167e-02, 1.66406250e-02,
       1.66458333e-02, 1.66510417e-02, 1.66562500e-02, 1.66614583e-02])

 
    waveform=numpy.array(
      [ 4.96392e-01,  7.83771e-01,  1.16763e+00,  1.65027e+00,
        2.22544e+00,  2.87985e+00,  3.59619e+00,  4.35663e+00,
        5.14554e+00,  5.95102e+00,  6.76511e+00,  7.58307e+00,
        8.40243e+00,  9.22200e+00,  1.00413e+01,  1.08602e+01,
        1.16786e+01,  1.24965e+01,  1.33140e+01,  1.41309e+01,
        1.49474e+01,  1.57635e+01,  1.65792e+01,  1.73944e+01,
        1.82092e+01,  1.90236e+01,  1.98375e+01,  2.06511e+01,
        2.14642e+01,  2.22768e+01,  2.30890e+01,  2.39007e+01,
        2.47120e+01,  2.55228e+01,  2.63332e+01,  2.71431e+01,
        2.79527e+01,  2.87618e+01,  2.95704e+01,  3.03787e+01,
        3.11865e+01,  3.19938e+01,  3.28007e+01,  3.36071e+01,
        3.44130e+01,  3.52184e+01,  3.60233e+01,  3.68278e+01,
        3.76319e+01,  3.84355e+01,  3.92386e+01,  4.00413e+01,
        4.08435e+01,  4.16453e+01,  4.24465e+01,  4.32472e+01,
        4.40475e+01,  4.48473e+01,  4.56465e+01,  4.64452e+01,
        4.72434e+01,  4.80412e+01,  4.88384e+01,  4.96352e+01,
        5.04315e+01,  5.12272e+01,  5.20224e+01,  5.28171e+01,
        5.36114e+01,  5.44050e+01,  5.51981e+01,  5.59907e+01,
        5.67827e+01,  5.75743e+01,  5.83652e+01,  5.91557e+01,
        5.99456e+01,  6.07349e+01,  6.15237e+01,  6.23120e+01,
        6.30997e+01,  6.38869e+01,  6.46734e+01,  6.54594e+01,
        6.62449e+01,  6.70297e+01,  6.78140e+01,  6.85977e+01,
        6.93809e+01,  7.01635e+01,  7.09454e+01,  7.17268e+01,
        7.25076e+01,  7.32878e+01,  7.40675e+01,  7.48465e+01,
        7.56249e+01,  7.64027e+01,  7.71799e+01,  7.79565e+01,
        7.87325e+01,  7.95078e+01,  8.02826e+01,  8.10567e+01,
        8.18301e+01,  8.26030e+01,  8.33753e+01,  8.41469e+01,
        8.49179e+01,  8.56882e+01,  8.64578e+01,  8.72269e+01,
        8.79953e+01,  8.87630e+01,  8.95301e+01,  9.02965e+01,
        9.10623e+01,  9.18274e+01,  9.25918e+01,  9.33556e+01,
        9.41187e+01,  9.48812e+01,  9.56429e+01,  9.64040e+01,
        9.71644e+01,  9.79240e+01,  9.86830e+01,  9.94414e+01,
        1.00199e+02,  1.00956e+02,  1.01712e+02,  1.02468e+02,
        1.03223e+02,  1.03977e+02,  1.04730e+02,  1.05483e+02,
        1.06235e+02,  1.06986e+02,  1.07737e+02,  1.08486e+02,
        1.09235e+02,  1.09984e+02,  1.10731e+02,  1.11478e+02,
        1.12224e+02,  1.12970e+02,  1.13714e+02,  1.14458e+02,
        1.15202e+02,  1.15944e+02,  1.16686e+02,  1.17427e+02,
        1.18167e+02,  1.18906e+02,  1.19645e+02,  1.20382e+02,
        1.21119e+02,  1.21856e+02,  1.22591e+02,  1.23326e+02,
        1.24060e+02,  1.24793e+02,  1.25525e+02,  1.26257e+02,
        1.26988e+02,  1.27718e+02,  1.28447e+02,  1.29175e+02,
        1.29903e+02,  1.30630e+02,  1.31356e+02,  1.32081e+02,
        1.32805e+02,  1.33529e+02,  1.34251e+02,  1.34973e+02,
        1.35694e+02,  1.36414e+02,  1.37134e+02,  1.37852e+02,
        1.38570e+02,  1.39287e+02,  1.40003e+02,  1.40718e+02,
        1.41433e+02,  1.42146e+02,  1.42859e+02,  1.43571e+02,
        1.44281e+02,  1.44992e+02,  1.45701e+02,  1.46409e+02,
        1.47117e+02,  1.47823e+02,  1.48529e+02,  1.49234e+02,
        1.49938e+02,  1.50641e+02,  1.51343e+02,  1.52044e+02,
        1.52745e+02,  1.53444e+02,  1.54143e+02,  1.54841e+02,
        1.55537e+02,  1.56233e+02,  1.56928e+02,  1.57623e+02,
        1.58316e+02,  1.59008e+02,  1.59700e+02,  1.60390e+02,
        1.61080e+02,  1.61768e+02,  1.62456e+02,  1.63143e+02,
        1.63829e+02,  1.64513e+02,  1.65198e+02,  1.65880e+02,
        1.66563e+02,  1.67244e+02,  1.67924e+02,  1.68604e+02,
        1.69282e+02,  1.69959e+02,  1.70636e+02,  1.71311e+02,
        1.71986e+02,  1.72659e+02,  1.73332e+02,  1.74003e+02,
        1.74674e+02,  1.75344e+02,  1.76013e+02,  1.76680e+02,
        1.77347e+02,  1.78013e+02,  1.78678e+02,  1.79342e+02,
        1.80004e+02,  1.80666e+02,  1.81327e+02,  1.81987e+02,
        1.82646e+02,  1.83304e+02,  1.83960e+02,  1.84616e+02,
        1.85271e+02,  1.85925e+02,  1.86578e+02,  1.87230e+02,
        1.87881e+02,  1.88530e+02,  1.89179e+02,  1.89827e+02,
        1.90474e+02,  1.91119e+02,  1.91764e+02,  1.92408e+02,
        1.93050e+02,  1.93692e+02,  1.94332e+02,  1.94972e+02,
        1.95610e+02,  1.96248e+02,  1.96884e+02,  1.97519e+02,
        1.98153e+02,  1.98787e+02,  1.99419e+02,  2.00050e+02,
        2.00680e+02,  2.01309e+02,  2.01937e+02,  2.02564e+02,
        2.03189e+02,  2.03814e+02,  2.04438e+02,  2.05060e+02,
        2.05681e+02,  2.06302e+02,  2.06922e+02,  2.07540e+02,
        2.08157e+02,  2.08773e+02,  2.09388e+02,  2.10001e+02,
        2.10614e+02,  2.11226e+02,  2.11837e+02,  2.12446e+02,
        2.13054e+02,  2.13662e+02,  2.14268e+02,  2.14873e+02,
        2.15477e+02,  2.16079e+02,  2.16681e+02,  2.17282e+02,
        2.17881e+02,  2.18480e+02,  2.19077e+02,  2.19673e+02,
        2.20268e+02,  2.20862e+02,  2.21454e+02,  2.22046e+02,
        2.22636e+02,  2.23226e+02,  2.23814e+02,  2.24401e+02,
        2.24986e+02,  2.25571e+02,  2.26155e+02,  2.26737e+02,
        2.27318e+02,  2.27899e+02,  2.28477e+02,  2.29055e+02,
        2.29632e+02,  2.30207e+02,  2.30782e+02,  2.31355e+02,
        2.31927e+02,  2.32498e+02,  2.33067e+02,  2.33636e+02,
        2.34203e+02,  2.34769e+02,  2.35334e+02,  2.35898e+02,
        2.36460e+02,  2.37022e+02,  2.37582e+02,  2.38141e+02,
        2.38699e+02,  2.39255e+02,  2.39811e+02,  2.40365e+02,
        2.40918e+02,  2.41470e+02,  2.42020e+02,  2.42570e+02,
        2.43118e+02,  2.43665e+02,  2.44211e+02,  2.44756e+02,
        2.45299e+02,  2.45841e+02,  2.46382e+02,  2.46922e+02,
        2.47460e+02,  2.47998e+02,  2.48533e+02,  2.49069e+02,
        2.49602e+02,  2.50134e+02,  2.50665e+02,  2.51195e+02,
        2.51724e+02,  2.52251e+02,  2.52778e+02,  2.53303e+02,
        2.53827e+02,  2.54349e+02,  2.54870e+02,  2.55390e+02,
        2.55909e+02,  2.56427e+02,  2.56943e+02,  2.57458e+02,
        2.57972e+02,  2.58484e+02,  2.58995e+02,  2.59505e+02,
        2.60014e+02,  2.60521e+02,  2.61028e+02,  2.61533e+02,
        2.62036e+02,  2.62539e+02,  2.63040e+02,  2.63540e+02,
        2.64038e+02,  2.64536e+02,  2.65032e+02,  2.65526e+02,
        2.66020e+02,  2.66512e+02,  2.67003e+02,  2.67493e+02,
        2.67981e+02,  2.68468e+02,  2.68954e+02,  2.69438e+02,
        2.69921e+02,  2.70403e+02,  2.70884e+02,  2.71363e+02,
        2.71841e+02,  2.72318e+02,  2.72793e+02,  2.73267e+02,
        2.73740e+02,  2.74211e+02,  2.74682e+02,  2.75150e+02,
        2.75618e+02,  2.76084e+02,  2.76549e+02,  2.77013e+02,
        2.77475e+02,  2.77936e+02,  2.78396e+02,  2.78854e+02,
        2.79311e+02,  2.79767e+02,  2.80221e+02,  2.80674e+02,
        2.81126e+02,  2.81577e+02,  2.82026e+02,  2.82473e+02,
        2.82920e+02,  2.83365e+02,  2.83809e+02,  2.84251e+02,
        2.84692e+02,  2.85131e+02,  2.85570e+02,  2.86007e+02,
        2.86443e+02,  2.86877e+02,  2.87310e+02,  2.87742e+02,
        2.88172e+02,  2.88601e+02,  2.89029e+02,  2.89455e+02,
        2.89880e+02,  2.90303e+02,  2.90725e+02,  2.91146e+02,
        2.91566e+02,  2.91984e+02,  2.92401e+02,  2.92816e+02,
        2.93231e+02,  2.93643e+02,  2.94054e+02,  2.94465e+02,
        2.94873e+02,  2.95280e+02,  2.95686e+02,  2.96091e+02,
        2.96494e+02,  2.96895e+02,  2.97296e+02,  2.97695e+02,
        2.98092e+02,  2.98488e+02,  2.98884e+02,  2.99277e+02,
        2.99669e+02,  3.00060e+02,  3.00450e+02,  3.00838e+02,
        3.01225e+02,  3.01611e+02,  3.01997e+02,  3.02380e+02,
        3.02764e+02,  3.03146e+02,  3.03527e+02,  3.03907e+02,
        3.04286e+02,  3.04665e+02,  3.05043e+02,  3.05420e+02,
        3.05796e+02,  3.06172e+02,  3.06547e+02,  3.06922e+02,
        3.07296e+02,  3.07670e+02,  3.08044e+02,  3.08417e+02,
        3.08790e+02,  3.09163e+02,  3.09535e+02,  3.09907e+02,
        3.10279e+02,  3.10651e+02,  3.11022e+02,  3.11393e+02,
        3.11764e+02,  3.12135e+02,  3.12506e+02,  3.12876e+02,
        3.13247e+02,  3.13617e+02,  3.13986e+02,  3.14357e+02,
        3.14726e+02,  3.15095e+02,  3.15464e+02,  3.15833e+02,
        3.16201e+02,  3.16569e+02,  3.16937e+02,  3.17305e+02,
        3.17672e+02,  3.18039e+02,  3.18405e+02,  3.18772e+02,
        3.19138e+02,  3.19504e+02,  3.19870e+02,  3.20234e+02,
        3.20599e+02,  3.20964e+02,  3.21327e+02,  3.21692e+02,
        3.22055e+02,  3.22418e+02,  3.22781e+02,  3.23143e+02,
        3.23506e+02,  3.23868e+02,  3.24229e+02,  3.24590e+02,
        3.24950e+02,  3.25311e+02,  3.25671e+02,  3.26031e+02,
        3.26391e+02,  3.26750e+02,  3.27109e+02,  3.27467e+02,
        3.27825e+02,  3.28183e+02,  3.28541e+02,  3.28899e+02,
        3.29256e+02,  3.29612e+02,  3.29968e+02,  3.30324e+02,
        3.30681e+02,  3.31036e+02,  3.31391e+02,  3.31745e+02,
        3.32100e+02,  3.32454e+02,  3.32808e+02,  3.33162e+02,
        3.33515e+02,  3.33868e+02,  3.34221e+02,  3.34573e+02,
        3.34925e+02,  3.35276e+02,  3.35628e+02,  3.35979e+02,
        3.36330e+02,  3.36681e+02,  3.37030e+02,  3.37381e+02,
        3.37730e+02,  3.38079e+02,  3.38429e+02,  3.38777e+02,
        3.39125e+02,  3.39473e+02,  3.39821e+02,  3.40168e+02,
        3.40515e+02,  3.40862e+02,  3.41209e+02,  3.41555e+02,
        3.41900e+02,  3.42246e+02,  3.42591e+02,  3.42936e+02,
        3.43281e+02,  3.43625e+02,  3.43969e+02,  3.44313e+02,
        3.44656e+02,  3.44999e+02,  3.45342e+02,  3.45684e+02,
        3.46026e+02,  3.46368e+02,  3.46710e+02,  3.47051e+02,
        3.47392e+02,  3.47732e+02,  3.48073e+02,  3.48413e+02,
        3.48753e+02,  3.49092e+02,  3.49430e+02,  3.49770e+02,
        3.50108e+02,  3.50446e+02,  3.50784e+02,  3.51121e+02,
        3.51459e+02,  3.51796e+02,  3.52132e+02,  3.52469e+02,
        3.52804e+02,  3.53140e+02,  3.53475e+02,  3.53811e+02,
        3.54145e+02,  3.54480e+02,  3.54814e+02,  3.55147e+02,
        3.55481e+02,  3.55815e+02,  3.56147e+02,  3.56480e+02,
        3.56812e+02,  3.57144e+02,  3.57476e+02,  3.57808e+02,
        3.58138e+02,  3.58469e+02,  3.58800e+02,  3.59129e+02,
        3.59459e+02,  3.59787e+02,  3.60113e+02,  3.60433e+02,
        3.60740e+02,  3.61026e+02,  3.61274e+02,  3.61466e+02,
        3.61581e+02,  3.61598e+02,  3.61503e+02,  3.61291e+02,
        3.60968e+02,  3.60549e+02,  3.60054e+02,  3.59509e+02,
        3.58933e+02,  3.58342e+02,  3.57746e+02,  3.57150e+02,
        3.56553e+02,  3.55955e+02,  3.55352e+02,  3.54742e+02,
        3.54125e+02,  3.53501e+02,  3.52870e+02,  3.52236e+02,
        3.51601e+02,  3.50969e+02,  3.50340e+02,  3.49716e+02,
        3.49096e+02,  3.48478e+02,  3.47859e+02,  3.47238e+02,
        3.46611e+02,  3.45978e+02,  3.45340e+02,  3.44697e+02,
        3.44051e+02,  3.43404e+02,  3.42759e+02,  3.42116e+02,
        3.41476e+02,  3.40839e+02,  3.40204e+02,  3.39568e+02,
        3.38929e+02,  3.38287e+02,  3.37640e+02,  3.36988e+02,
        3.36332e+02,  3.35673e+02,  3.35013e+02,  3.34353e+02,
        3.33695e+02,  3.33040e+02,  3.32386e+02,  3.31733e+02,
        3.31082e+02,  3.30428e+02,  3.29771e+02,  3.29110e+02,
        3.28446e+02,  3.27777e+02,  3.27105e+02,  3.26432e+02,
        3.25759e+02,  3.25087e+02,  3.24416e+02,  3.23746e+02,
        3.23077e+02,  3.22408e+02,  3.21739e+02,  3.21068e+02,
        3.20393e+02,  3.19715e+02,  3.19034e+02,  3.18351e+02,
        3.17666e+02,  3.16980e+02,  3.16294e+02,  3.15609e+02,
        3.14925e+02,  3.14241e+02,  3.13558e+02,  3.12873e+02,
        3.12186e+02,  3.11498e+02,  3.10807e+02,  3.10113e+02,
        3.09417e+02,  3.08719e+02,  3.08021e+02,  3.07323e+02,
        3.06625e+02,  3.05927e+02,  3.05230e+02,  3.04532e+02,
        3.03833e+02,  3.03134e+02,  3.02432e+02,  3.01728e+02,
        3.01021e+02,  3.00313e+02,  2.99604e+02,  2.98893e+02,
        2.98182e+02,  2.97471e+02,  2.96760e+02,  2.96049e+02,
        2.95338e+02,  2.94627e+02,  2.93914e+02,  2.93200e+02,
        2.92484e+02,  2.91766e+02,  2.91046e+02,  2.90324e+02,
        2.89601e+02,  2.88878e+02,  2.88155e+02,  2.87432e+02,
        2.86708e+02,  2.85984e+02,  2.85260e+02,  2.84535e+02,
        2.83808e+02,  2.83080e+02,  2.82351e+02,  2.81620e+02,
        2.80887e+02,  2.80154e+02,  2.79419e+02,  2.78685e+02,
        2.77949e+02,  2.77214e+02,  2.76478e+02,  2.75742e+02,
        2.75005e+02,  2.74266e+02,  2.73527e+02,  2.72786e+02,
        2.72044e+02,  2.71301e+02,  2.70556e+02,  2.69811e+02,
        2.69065e+02,  2.68319e+02,  2.67573e+02,  2.66826e+02,
        2.66078e+02,  2.65330e+02,  2.64581e+02,  2.63831e+02,
        2.63079e+02,  2.62326e+02,  2.61572e+02,  2.60818e+02,
        2.60062e+02,  2.59306e+02,  2.58549e+02,  2.57792e+02,
        2.57035e+02,  2.56277e+02,  2.55518e+02,  2.54759e+02,
        2.53998e+02,  2.53236e+02,  2.52474e+02,  2.51710e+02,
        2.50945e+02,  2.50180e+02,  2.49414e+02,  2.48647e+02,
        2.47880e+02,  2.47113e+02,  2.46345e+02,  2.45576e+02,
        2.44807e+02,  2.44036e+02,  2.43265e+02,  2.42493e+02,
        2.41720e+02,  2.40946e+02,  2.40172e+02,  2.39396e+02,
        2.38621e+02,  2.37844e+02,  2.37067e+02,  2.36290e+02,
        2.35512e+02,  2.34734e+02,  2.33954e+02,  2.33174e+02,
        2.32393e+02,  2.31611e+02,  2.30829e+02,  2.30045e+02,
        2.29261e+02,  2.28477e+02,  2.27692e+02,  2.26906e+02,
        2.26120e+02,  2.25334e+02,  2.24546e+02,  2.23759e+02,
        2.22970e+02,  2.22181e+02,  2.21390e+02,  2.20600e+02,
        2.19808e+02,  2.19016e+02,  2.18223e+02,  2.17430e+02,
        2.16637e+02,  2.15842e+02,  2.15048e+02,  2.14253e+02,
        2.13457e+02,  2.12661e+02,  2.11863e+02,  2.11066e+02,
        2.10267e+02,  2.09468e+02,  2.08669e+02,  2.07868e+02,
        2.07068e+02,  2.06267e+02,  2.05465e+02,  2.04663e+02,
        2.03860e+02,  2.03057e+02,  2.02254e+02,  2.01449e+02,
        2.00644e+02,  1.99839e+02,  1.99033e+02,  1.98226e+02,
        1.97419e+02,  1.96612e+02,  1.95804e+02,  1.94995e+02,
        1.94186e+02,  1.93377e+02,  1.92567e+02,  1.91757e+02,
        1.90946e+02,  1.90134e+02,  1.89322e+02,  1.88510e+02,
        1.87697e+02,  1.86884e+02,  1.86070e+02,  1.85256e+02,
        1.84441e+02,  1.83626e+02,  1.82810e+02,  1.81994e+02,
        1.81178e+02,  1.80361e+02,  1.79544e+02,  1.78726e+02,
        1.77908e+02,  1.77089e+02,  1.76270e+02,  1.75450e+02,
        1.74630e+02,  1.73810e+02,  1.72989e+02,  1.72168e+02,
        1.71347e+02,  1.70525e+02,  1.69703e+02,  1.68880e+02,
        1.68057e+02,  1.67233e+02,  1.66409e+02,  1.65585e+02,
        1.64760e+02,  1.63935e+02,  1.63110e+02,  1.62284e+02,
        1.61458e+02,  1.60632e+02,  1.59805e+02,  1.58978e+02,
        1.58151e+02,  1.57323e+02,  1.56495e+02,  1.55666e+02,
        1.54837e+02,  1.54008e+02,  1.53179e+02,  1.52349e+02,
        1.51519e+02,  1.50689e+02,  1.49858e+02,  1.49027e+02,
        1.48195e+02,  1.47364e+02,  1.46532e+02,  1.45700e+02,
        1.44867e+02,  1.44035e+02,  1.43201e+02,  1.42368e+02,
        1.41535e+02,  1.40701e+02,  1.39867e+02,  1.39032e+02,
        1.38198e+02,  1.37363e+02,  1.36528e+02,  1.35692e+02,
        1.34857e+02,  1.34021e+02,  1.33184e+02,  1.32348e+02,
        1.31511e+02,  1.30675e+02,  1.29837e+02,  1.29000e+02,
        1.28163e+02,  1.27325e+02,  1.26487e+02,  1.25649e+02,
        1.24811e+02,  1.23972e+02,  1.23134e+02,  1.22295e+02,
        1.21456e+02,  1.20616e+02,  1.19777e+02,  1.18937e+02,
        1.18098e+02,  1.17258e+02,  1.16418e+02,  1.15577e+02,
        1.14737e+02,  1.13896e+02,  1.13056e+02,  1.12215e+02,
        1.11373e+02,  1.10532e+02,  1.09691e+02,  1.08849e+02,
        1.08008e+02,  1.07166e+02,  1.06324e+02,  1.05482e+02,
        1.04640e+02,  1.03798e+02,  1.02956e+02,  1.02113e+02,
        1.01271e+02,  1.00428e+02,  9.95857e+01,  9.87429e+01,
        9.79000e+01,  9.70569e+01,  9.62139e+01,  9.53708e+01,
        9.45277e+01,  9.36844e+01,  9.28411e+01,  9.19977e+01,
        9.11543e+01,  9.03108e+01,  8.94672e+01,  8.86235e+01,
        8.77799e+01,  8.69361e+01,  8.60924e+01,  8.52486e+01,
        8.44049e+01,  8.35610e+01,  8.27172e+01,  8.18734e+01,
        8.10295e+01,  8.01856e+01,  7.93417e+01,  7.84977e+01,
        7.76538e+01,  7.68099e+01,  7.59660e+01,  7.51221e+01,
        7.42782e+01,  7.34343e+01,  7.25905e+01,  7.17466e+01,
        7.09029e+01,  7.00591e+01,  6.92153e+01,  6.83716e+01,
        6.75279e+01,  6.66843e+01,  6.58407e+01,  6.49972e+01,
        6.41537e+01,  6.33103e+01,  6.24669e+01,  6.16236e+01,
        6.07804e+01,  5.99373e+01,  5.90942e+01,  5.82512e+01,
        5.74083e+01,  5.65655e+01,  5.57228e+01,  5.48801e+01,
        5.40376e+01,  5.31952e+01,  5.23528e+01,  5.15106e+01,
        5.06685e+01,  4.98265e+01,  4.89846e+01,  4.81428e+01,
        4.73012e+01,  4.64597e+01,  4.56183e+01,  4.47771e+01,
        4.39360e+01,  4.30951e+01,  4.22543e+01,  4.14136e+01,
        4.05731e+01,  3.97328e+01,  3.88926e+01,  3.80526e+01,
        3.72128e+01,  3.63731e+01,  3.55336e+01,  3.46943e+01,
        3.38552e+01,  3.30163e+01,  3.21775e+01,  3.13390e+01,
        3.05006e+01,  2.96625e+01,  2.88246e+01,  2.79868e+01,
        2.71493e+01,  2.63121e+01,  2.54750e+01,  2.46382e+01,
        2.38016e+01,  2.29652e+01,  2.21290e+01,  2.12931e+01,
        2.04575e+01,  1.96221e+01,  1.87870e+01,  1.79521e+01,
        1.71175e+01,  1.62832e+01,  1.54491e+01,  1.46153e+01,
        1.37818e+01,  1.29486e+01,  1.21157e+01,  1.12831e+01,
        1.04511e+01,  9.61972e+00,  8.78981e+00,  7.96280e+00,
        7.14143e+00,  6.33040e+00,  5.53679e+00,  4.77024e+00,
        4.04245e+00,  3.36596e+00,  2.75248e+00,  2.21101e+00,
        1.74646e+00,  1.35898e+00,  1.04428e+00,  7.94686e-01,
        6.00647e-01,  4.52122e-01,  3.39690e-01,  2.55183e-01,
        1.91918e-01,  1.44636e-01,  1.09306e-01,  8.28876e-02,
        6.31054e-02,  4.82631e-02,  3.70987e-02,  2.86739e-02,
        2.22912e-02,  1.74318e-02,  1.37104e-02,  1.08400e-02,
        8.60762e-03,  6.85455e-03,  5.46278e-03,  4.34447e-03,
        3.43413e-03,  2.68296e-03,  2.05445e-03,  1.52128e-03,
        1.06291e-03,  6.63917e-04,  3.12607e-04,  1.25581e-07,
       -2.80289e-04, -5.33842e-04, -7.64578e-04, -9.75678e-04,
       -1.16967e-03, -1.34860e-03, -1.51413e-03, -1.66762e-03,
       -1.81025e-03, -1.94298e-03, -2.06667e-03, -2.18203e-03,
       -2.28964e-03, -2.38999e-03, -2.48349e-03, -2.57047e-03,
       -2.65119e-03, -2.72594e-03, -2.79494e-03, -2.85848e-03,
       -2.91681e-03, -2.97023e-03, -3.01908e-03, -3.06371e-03,
       -3.10451e-03, -3.14188e-03, -3.17621e-03, -3.20790e-03,
       -3.23726e-03, -3.26460e-03, -3.29014e-03, -3.31407e-03,
       -3.33652e-03, -3.35758e-03, -3.37732e-03, -3.39579e-03,
       -3.41301e-03, -3.42905e-03, -3.44395e-03, -3.45781e-03,
       -3.47072e-03, -3.48282e-03, -3.49424e-03, -3.50511e-03,
       -3.51556e-03, -3.52569e-03, -3.53555e-03, -3.54519e-03,
       -3.55463e-03, -3.56381e-03, -3.57272e-03, -3.58128e-03,
       -3.58944e-03, -3.59715e-03, -3.60436e-03, -3.61108e-03,
       -3.61732e-03, -3.62313e-03, -3.62859e-03, -3.63378e-03,
       -3.63880e-03, -3.64372e-03, -3.64862e-03, -3.65356e-03,
       -3.65854e-03, -3.66356e-03, -3.66859e-03, -3.67355e-03,
       -3.67838e-03, -3.68300e-03, -3.68735e-03, -3.69140e-03,
       -3.69513e-03, -3.69854e-03, -3.70167e-03, -3.70457e-03,
       -3.70732e-03, -3.70999e-03, -3.71264e-03, -3.71534e-03,
       -3.71814e-03, -3.72105e-03, -3.72406e-03, -3.72714e-03,
       -3.73025e-03, -3.73331e-03, -3.73628e-03, -3.73909e-03,
       -3.74171e-03, -3.74412e-03, -3.74632e-03, -3.74836e-03,
       -3.75026e-03, -3.75210e-03, -3.75393e-03, -3.75581e-03,
       -3.75780e-03, -3.75991e-03, -3.76217e-03, -3.76455e-03,
       -3.76703e-03, -3.76955e-03, -3.77206e-03, -3.77450e-03,
       -3.77683e-03, -3.77900e-03, -3.78099e-03, -3.78279e-03,
       -3.78441e-03, -3.78589e-03, -3.78727e-03, -3.78862e-03,
       -3.78999e-03, -3.79143e-03, -3.79299e-03, -3.79467e-03,
       -3.79647e-03, -3.79837e-03, -3.80032e-03, -3.80227e-03,
       -3.80415e-03, -3.80595e-03, -3.80761e-03, -3.80912e-03,
       -3.81048e-03, -3.81172e-03, -3.81286e-03, -3.81396e-03,
       -3.81507e-03, -3.81623e-03, -3.81749e-03, -3.81886e-03,
       -3.82037e-03, -3.82201e-03, -3.82375e-03, -3.82556e-03,
       -3.82740e-03, -3.82920e-03, -3.83094e-03, -3.83257e-03,
       -3.83406e-03, -3.83542e-03, -3.83665e-03, -3.83779e-03,
       -3.83887e-03, -3.83993e-03, -3.84101e-03, -3.84215e-03,
       -3.84336e-03, -3.84466e-03, -3.84605e-03, -3.84752e-03,
       -3.84903e-03, -3.85056e-03, -3.85205e-03, -3.85348e-03,
       -3.85481e-03, -3.85602e-03, -3.85712e-03, -3.85810e-03,
       -3.85900e-03, -3.85984e-03, -3.86067e-03, -3.86155e-03,
       -3.86250e-03, -3.86354e-03, -3.86470e-03, -3.86598e-03,
       -3.86734e-03, -3.86876e-03, -3.87020e-03, -3.87162e-03,
       -3.87299e-03, -3.87426e-03, -3.87543e-03, -3.87649e-03,
       -3.87746e-03, -3.87835e-03, -3.87920e-03, -3.88005e-03,
       -3.88092e-03, -3.88186e-03, -3.88287e-03, -3.88396e-03,
       -3.88511e-03, -3.88633e-03, -3.88756e-03, -3.88879e-03,
       -3.88998e-03, -3.89111e-03, -3.89214e-03, -3.89308e-03,
       -3.89391e-03, -3.89464e-03, -3.89529e-03, -3.89588e-03,
       -3.89647e-03, -3.89708e-03, -3.89775e-03, -3.89850e-03,
       -3.89934e-03, -3.90029e-03, -3.90132e-03, -3.90240e-03,
       -3.90351e-03, -3.90460e-03, -3.90565e-03, -3.90663e-03,
       -3.90753e-03, -3.90833e-03, -3.90905e-03, -3.90973e-03,
       -3.91037e-03, -3.91100e-03, -3.91168e-03, -3.91240e-03,
       -3.91318e-03, -3.91403e-03, -3.91494e-03, -3.91590e-03,
       -3.91689e-03, -3.91787e-03, -3.91884e-03, -3.91975e-03,
       -3.92059e-03, -3.92136e-03, -3.92204e-03, -3.92266e-03,
       -3.92322e-03, -3.92375e-03, -3.92427e-03, -3.92481e-03,
       -3.92539e-03, -3.92603e-03, -3.92673e-03, -3.92749e-03,
       -3.92831e-03, -3.92918e-03, -3.93007e-03, -3.93096e-03,
       -3.93184e-03, -3.93269e-03, -3.93349e-03, -3.93424e-03,
       -3.93494e-03, -3.93559e-03, -3.93622e-03, -3.93683e-03,
       -3.93745e-03, -3.93810e-03, -3.93880e-03, -3.93955e-03,
       -3.94036e-03, -3.94120e-03, -3.94206e-03, -3.94292e-03,
       -3.94374e-03, -3.94449e-03, -3.94518e-03, -3.94579e-03,
       -3.94632e-03, -3.94679e-03, -3.94721e-03, -3.94761e-03,
       -3.94799e-03, -3.94838e-03, -3.94880e-03, -3.94925e-03,
       -3.94975e-03, -3.95028e-03, -3.95086e-03, -3.95146e-03,
       -3.95208e-03, -3.95271e-03, -3.95332e-03, -3.95392e-03,
       -3.95449e-03, -3.95502e-03, -3.95553e-03, -3.95602e-03,
       -3.95651e-03, -3.95701e-03, -3.95753e-03, -3.95808e-03,
       -3.95868e-03, -3.95931e-03, -3.95999e-03, -3.96071e-03,
       -3.96145e-03, -3.96220e-03, -3.96295e-03, -3.96368e-03,
       -3.96436e-03, -3.96499e-03, -3.96555e-03, -3.96605e-03,
       -3.96648e-03, -3.96688e-03, -3.96724e-03, -3.96760e-03,
       -3.96796e-03, -3.96834e-03, -3.96875e-03, -3.96919e-03,
       -3.96967e-03, -3.97016e-03, -3.97068e-03, -3.97120e-03,
       -3.97169e-03, -3.97217e-03, -3.97261e-03, -3.97302e-03,
       -3.97340e-03, -3.97376e-03, -3.97412e-03, -3.97449e-03,
       -3.97490e-03, -3.97533e-03, -3.97580e-03, -3.97631e-03,
       -3.97686e-03, -3.97743e-03, -3.97801e-03, -3.97862e-03,
       -3.97921e-03, -3.97979e-03, -3.98034e-03, -3.98085e-03,
       -3.98132e-03, -3.98175e-03, -3.98213e-03, -3.98249e-03,
       -3.98284e-03, -3.98320e-03, -3.98358e-03, -3.98397e-03,
       -3.98439e-03, -3.98483e-03, -3.98529e-03, -3.98574e-03,
       -3.98618e-03, -3.98659e-03, -3.98698e-03, -3.98733e-03,
       -3.98766e-03, -3.98796e-03, -3.98824e-03, -3.98853e-03,
       -3.98882e-03, -3.98914e-03, -3.98950e-03, -3.98990e-03,
       -3.99033e-03, -3.99081e-03, -3.99132e-03, -3.99185e-03,
       -3.99241e-03, -3.99296e-03, -3.99353e-03, -3.99408e-03,
       -3.99461e-03, -3.99511e-03, -3.99559e-03, -3.99604e-03,
       -3.99645e-03, -3.99684e-03, -3.99722e-03, -3.99759e-03,
       -3.99797e-03, -3.99835e-03, -3.99874e-03, -3.99914e-03,
       -3.99956e-03, -3.99998e-03, -4.00040e-03, -4.00081e-03,
       -4.00120e-03, -4.00157e-03, -4.00191e-03, -4.00222e-03,
       -4.00249e-03, -4.00274e-03, -4.00298e-03, -4.00322e-03,
       -4.00348e-03, -4.00376e-03, -4.00408e-03, -4.00444e-03,
       -4.00484e-03, -4.00526e-03, -4.00569e-03, -4.00613e-03,
       -4.00656e-03, -4.00697e-03, -4.00735e-03, -4.00772e-03,
       -4.00806e-03, -4.00839e-03, -4.00870e-03, -4.00901e-03,
       -4.00933e-03, -4.00966e-03, -4.01000e-03, -4.01037e-03,
       -4.01075e-03, -4.01114e-03, -4.01153e-03, -4.01191e-03,
       -4.01228e-03, -4.01264e-03, -4.01298e-03, -4.01330e-03,
       -4.01359e-03, -4.01387e-03, -4.01411e-03, -4.01433e-03,
       -4.01454e-03, -4.01473e-03, -4.01492e-03, -4.01512e-03,
       -4.01534e-03, -4.01557e-03, -4.01584e-03, -4.01614e-03,
       -4.01647e-03, -4.01682e-03, -4.01720e-03, -4.01758e-03,
       -4.01797e-03, -4.01835e-03, -4.01873e-03, -4.01910e-03,
       -4.01946e-03, -4.01983e-03, -4.02019e-03, -4.02055e-03,
       -4.02092e-03, -4.02129e-03, -4.02167e-03, -4.02206e-03,
       -4.02246e-03, -4.02287e-03, -4.02328e-03, -4.02369e-03,
       -4.02408e-03, -4.02444e-03, -4.02476e-03, -4.02505e-03,
       -4.02532e-03, -4.02557e-03, -4.02580e-03, -4.02602e-03,
       -4.02624e-03, -4.02646e-03, -4.02670e-03, -4.02695e-03,
       -4.02722e-03, -4.02752e-03, -4.02784e-03, -4.02819e-03,
       -4.02855e-03, -4.02892e-03, -4.02929e-03, -4.02965e-03,
       -4.03000e-03, -4.03034e-03, -4.03069e-03, -4.03104e-03,
       -4.03141e-03, -4.03180e-03, -4.03222e-03, -4.03266e-03,
       -4.03311e-03, -4.03359e-03, -4.03407e-03, -4.03455e-03,
       -4.03504e-03, -4.03552e-03, -4.03597e-03, -4.03640e-03,
       -4.03679e-03, -4.03715e-03, -4.03748e-03, -4.03778e-03,
       -4.03807e-03, -4.03836e-03, -4.03867e-03, -4.03899e-03,
       -4.03932e-03, -4.03967e-03, -4.04004e-03, -4.04042e-03,
       -4.04080e-03, -4.04119e-03, -4.04157e-03, -4.04194e-03,
       -4.04230e-03, -4.04266e-03, -4.04300e-03, -4.04335e-03,
       -4.04371e-03, -4.04407e-03, -4.04444e-03, -4.04481e-03,
       -4.04520e-03, -4.04559e-03, -4.04599e-03, -4.04640e-03,
       -4.04682e-03, -4.04725e-03, -4.04768e-03, -4.04812e-03,
       -4.04855e-03, -4.04898e-03, -4.04940e-03, -4.04981e-03,
       -4.05022e-03, -4.05061e-03, -4.05098e-03, -4.05136e-03,
       -4.05173e-03, -4.05211e-03, -4.05249e-03, -4.05288e-03,
       -4.05328e-03, -4.05366e-03, -4.05405e-03, -4.05441e-03,
       -4.05477e-03, -4.05511e-03, -4.05544e-03, -4.05577e-03,
       -4.05608e-03, -4.05640e-03, -4.05672e-03, -4.05705e-03,
       -4.05739e-03, -4.05776e-03, -4.05816e-03, -4.05859e-03,
       -4.05905e-03, -4.05953e-03, -4.06003e-03, -4.06053e-03,
       -4.06105e-03, -4.06156e-03, -4.06206e-03, -4.06255e-03,
       -4.06302e-03, -4.06348e-03, -4.06392e-03, -4.06434e-03,
       -4.06475e-03, -4.06516e-03, -4.06557e-03, -4.06597e-03,
       -4.06638e-03, -4.06679e-03, -4.06721e-03, -4.06761e-03,
       -4.06802e-03, -4.06841e-03, -4.06878e-03, -4.06912e-03,
       -4.06945e-03, -4.06976e-03, -4.07006e-03, -4.07037e-03,
       -4.07068e-03, -4.07099e-03, -4.07132e-03, -4.07166e-03,
       -4.07202e-03, -4.07240e-03, -4.07281e-03, -4.07324e-03,
       -4.07370e-03, -4.07418e-03, -4.07467e-03, -4.07517e-03,
       -4.07566e-03, -4.07616e-03, -4.07666e-03, -4.07717e-03,
       -4.07767e-03, -4.07817e-03, -4.07865e-03, -4.07912e-03,
       -4.07958e-03, -4.08001e-03, -4.08044e-03, -4.08086e-03,
       -4.08126e-03, -4.08166e-03, -4.08204e-03, -4.08240e-03,
       -4.08274e-03, -4.08305e-03, -4.08333e-03, -4.08359e-03,
       -4.08384e-03, -4.08407e-03, -4.08430e-03, -4.08453e-03,
       -4.08477e-03, -4.08502e-03, -4.08530e-03, -4.08560e-03,
       -4.08592e-03, -4.08626e-03, -4.08663e-03, -4.08702e-03,
       -4.08741e-03, -4.08782e-03, -4.08824e-03, -4.08867e-03,
       -4.08909e-03, -4.08949e-03, -4.08989e-03, -4.09026e-03,
       -4.09062e-03, -4.09097e-03, -4.09131e-03, -4.09165e-03,
       -4.09200e-03, -4.09236e-03, -4.09272e-03, -4.09308e-03,
       -4.09344e-03, -4.09378e-03, -4.09412e-03, -4.09443e-03,
       -4.09473e-03, -4.09501e-03, -4.09527e-03, -4.09553e-03,
       -4.09577e-03, -4.09599e-03, -4.09620e-03, -4.09640e-03,
       -4.09660e-03, -4.09679e-03, -4.09700e-03, -4.09721e-03,
       -4.09745e-03, -4.09769e-03, -4.09796e-03, -4.09824e-03,
       -4.09854e-03, -4.09884e-03, -4.09915e-03, -4.09946e-03,
       -4.09977e-03, -4.10008e-03, -4.10040e-03, -4.10071e-03,
       -4.10102e-03, -4.10134e-03, -4.10165e-03, -4.10195e-03,
       -4.10224e-03, -4.10253e-03, -4.10280e-03, -4.10306e-03,
       -4.10330e-03, -4.10351e-03, -4.10372e-03, -4.10390e-03,
       -4.10407e-03, -4.10425e-03, -4.10443e-03, -4.10462e-03,
       -4.10483e-03, -4.10505e-03, -4.10528e-03, -4.10550e-03,
       -4.10573e-03, -4.10595e-03, -4.10616e-03, -4.10638e-03,
       -4.10659e-03, -4.10681e-03, -4.10702e-03, -4.10725e-03,
       -4.10748e-03, -4.10773e-03, -4.10797e-03, -4.10823e-03,
       -4.10849e-03, -4.10875e-03, -4.10901e-03, -4.10927e-03,
       -4.10953e-03, -4.10978e-03, -4.11004e-03, -4.11030e-03,
       -4.11056e-03, -4.11081e-03, -4.11108e-03, -4.11133e-03,
       -4.11158e-03, -4.11180e-03, -4.11201e-03, -4.11220e-03,
       -4.11236e-03, -4.11250e-03, -4.11263e-03, -4.11274e-03,
       -4.11283e-03, -4.11292e-03, -4.11299e-03, -4.11306e-03,
       -4.11313e-03, -4.11319e-03, -4.11325e-03, -4.11332e-03,
       -4.11339e-03, -4.11346e-03, -4.11353e-03, -4.11362e-03,
       -4.11371e-03, -4.11381e-03, -4.11393e-03, -4.11406e-03,
       -4.11418e-03, -4.11432e-03, -4.11446e-03, -4.11461e-03,
       -4.11477e-03, -4.11494e-03, -4.11513e-03, -4.11532e-03,
       -4.11553e-03, -4.11576e-03, -4.11599e-03, -4.11622e-03,
       -4.11645e-03, -4.11668e-03, -4.11689e-03, -4.11708e-03,
       -4.11725e-03, -4.11739e-03, -4.11751e-03, -4.11759e-03,
       -4.11764e-03, -4.11767e-03, -4.11768e-03, -4.11765e-03,
       -4.11761e-03, -4.11755e-03, -4.11748e-03, -4.11742e-03,
       -4.11736e-03, -4.11732e-03, -4.11730e-03, -4.11730e-03,
       -4.11732e-03, -4.11736e-03, -4.11742e-03, -4.11750e-03,
       -4.11760e-03, -4.11770e-03, -4.11781e-03, -4.11791e-03,
       -4.11801e-03, -4.11812e-03, -4.11822e-03, -4.11834e-03,
       -4.11846e-03, -4.11859e-03, -4.11873e-03, -4.11886e-03,
       -4.11898e-03, -4.11910e-03, -4.11919e-03, -4.11926e-03,
       -4.11930e-03, -4.11934e-03, -4.11935e-03, -4.11934e-03,
       -4.11933e-03, -4.11930e-03, -4.11926e-03, -4.11921e-03,
       -4.11915e-03, -4.11907e-03, -4.11899e-03, -4.11890e-03,
       -4.11880e-03, -4.11869e-03, -4.11858e-03, -4.11846e-03,
       -4.11835e-03, -4.11825e-03, -4.11816e-03, -4.11810e-03,
       -4.11806e-03, -4.11806e-03, -4.11808e-03, -4.11812e-03,
       -4.11817e-03, -4.11823e-03, -4.11828e-03, -4.11833e-03,
       -4.11838e-03, -4.11843e-03, -4.11847e-03, -4.11853e-03,
       -4.11859e-03, -4.11865e-03, -4.11871e-03, -4.11876e-03,
       -4.11878e-03, -4.11876e-03, -4.11872e-03, -4.11865e-03,
       -4.11855e-03, -4.11843e-03, -4.11830e-03, -4.11817e-03,
       -4.11804e-03, -4.11791e-03, -4.11778e-03, -4.11766e-03,
       -4.11753e-03, -4.11740e-03, -4.11728e-03, -4.11716e-03,
       -4.11703e-03, -4.11691e-03, -4.11680e-03, -4.11669e-03,
       -4.11661e-03, -4.11653e-03, -4.11649e-03, -4.11647e-03,
       -4.11647e-03, -4.11650e-03, -4.11654e-03, -4.11659e-03,
       -4.11665e-03, -4.11670e-03, -4.11676e-03, -4.11682e-03,
       -4.11689e-03, -4.11696e-03, -4.11704e-03, -4.11710e-03,
       -4.11717e-03, -4.11722e-03, -4.11726e-03, -4.11730e-03,
       -4.11733e-03, -4.11736e-03, -4.11740e-03, -4.11742e-03,
       -4.11745e-03, -4.11747e-03, -4.11749e-03, -4.11752e-03,
       -4.11755e-03, -4.11759e-03, -4.11763e-03, -4.11768e-03,
       -4.11773e-03, -4.11780e-03, -4.11789e-03, -4.11800e-03,
       -4.11813e-03, -4.11827e-03, -4.11842e-03, -4.11857e-03,
       -4.11874e-03, -4.11890e-03, -4.11908e-03, -4.11926e-03,
       -4.11944e-03, -4.11964e-03, -4.11986e-03, -4.12009e-03,
       -4.12034e-03, -4.12061e-03, -4.12090e-03, -4.12120e-03,
       -4.12151e-03, -4.12180e-03, -4.12208e-03, -4.12232e-03,
       -4.12255e-03, -4.12273e-03, -4.12290e-03, -4.12304e-03,
       -4.12317e-03, -4.12328e-03, -4.12339e-03, -4.12349e-03,
       -4.12359e-03, -4.12368e-03, -4.12378e-03, -4.12387e-03,
       -4.12397e-03, -4.12407e-03, -4.12419e-03, -4.12433e-03,
       -4.12448e-03, -4.12466e-03, -4.12484e-03, -4.12504e-03,
       -4.12525e-03, -4.12547e-03, -4.12570e-03, -4.12593e-03,
       -4.12618e-03, -4.12643e-03, -4.12670e-03, -4.12697e-03,
       -4.12724e-03, -4.12753e-03, -4.12781e-03, -4.12810e-03,
       -4.12838e-03, -4.12865e-03, -4.12891e-03, -4.12916e-03,
       -4.12939e-03, -4.12959e-03, -4.12978e-03, -4.12996e-03,
       -4.13011e-03, -4.13025e-03, -4.13039e-03, -4.13051e-03,
       -4.13062e-03, -4.13071e-03, -4.13080e-03, -4.13087e-03,
       -4.13095e-03, -4.13102e-03, -4.13109e-03, -4.13118e-03,
       -4.13127e-03, -4.13138e-03, -4.13149e-03, -4.13162e-03,
       -4.13176e-03, -4.13190e-03, -4.13204e-03, -4.13220e-03,
       -4.13235e-03, -4.13251e-03, -4.13268e-03, -4.13285e-03,
       -4.13304e-03, -4.13325e-03, -4.13347e-03, -4.13371e-03,
       -4.13394e-03, -4.13418e-03, -4.13441e-03, -4.13464e-03,
       -4.13484e-03, -4.13504e-03, -4.13522e-03, -4.13539e-03,
       -4.13555e-03, -4.13570e-03, -4.13585e-03, -4.13598e-03,
       -4.13610e-03, -4.13621e-03, -4.13631e-03, -4.13640e-03,
       -4.13649e-03, -4.13657e-03, -4.13666e-03, -4.13675e-03,
       -4.13685e-03, -4.13697e-03, -4.13709e-03, -4.13723e-03,
       -4.13738e-03, -4.13755e-03, -4.13773e-03, -4.13794e-03,
       -4.13816e-03, -4.13839e-03, -4.13864e-03, -4.13891e-03,
       -4.13918e-03, -4.13945e-03, -4.13973e-03, -4.14000e-03,
       -4.14025e-03, -4.14048e-03, -4.14069e-03, -4.14088e-03,
       -4.14105e-03, -4.14119e-03, -4.14131e-03, -4.14142e-03,
       -4.14151e-03, -4.14158e-03, -4.14165e-03, -4.14173e-03,
       -4.14179e-03, -4.14187e-03, -4.14195e-03, -4.14202e-03,
       -4.14208e-03, -4.14213e-03, -4.14218e-03, -4.14222e-03,
       -4.14227e-03, -4.14233e-03, -4.14241e-03, -4.14251e-03,
       -4.14264e-03, -4.14278e-03, -4.14294e-03, -4.14312e-03,
       -4.14330e-03, -4.14348e-03, -4.14367e-03, -4.14386e-03,
       -4.14404e-03, -4.14421e-03, -4.14438e-03, -4.14456e-03,
       -4.14473e-03, -4.14490e-03, -4.14508e-03, -4.14526e-03,
       -4.14545e-03, -4.14563e-03, -4.14580e-03, -4.14597e-03,
       -4.14612e-03, -4.14625e-03, -4.14637e-03, -4.14647e-03,
       -4.14654e-03, -4.14660e-03, -4.14664e-03, -4.14666e-03,
       -4.14667e-03, -4.14667e-03, -4.14666e-03, -4.14666e-03,
       -4.14667e-03, -4.14670e-03, -4.14676e-03, -4.14682e-03,
       -4.14690e-03, -4.14698e-03, -4.14707e-03, -4.14715e-03,
       -4.14724e-03, -4.14732e-03, -4.14740e-03, -4.14749e-03,
       -4.14757e-03, -4.14766e-03, -4.14773e-03, -4.14780e-03,
       -4.14787e-03, -4.14794e-03, -4.14800e-03, -4.14805e-03,
       -4.14810e-03, -4.14815e-03, -4.14819e-03, -4.14823e-03,
       -4.14825e-03, -4.14827e-03, -4.14828e-03, -4.14828e-03,
       -4.14826e-03, -4.14824e-03, -4.14820e-03, -4.14816e-03,
       -4.14811e-03, -4.14807e-03, -4.14802e-03, -4.14797e-03,
       -4.14792e-03, -4.14787e-03, -4.14782e-03, -4.14779e-03,
       -4.14775e-03, -4.14774e-03, -4.14774e-03, -4.14776e-03,
       -4.14780e-03, -4.14784e-03, -4.14790e-03, -4.14795e-03,
       -4.14801e-03, -4.14805e-03, -4.14810e-03, -4.14813e-03,
       -4.14817e-03, -4.14821e-03, -4.14826e-03, -4.14832e-03,
       -4.14838e-03, -4.14844e-03, -4.14851e-03, -4.14857e-03,
       -4.14861e-03, -4.14864e-03, -4.14866e-03, -4.14867e-03,
       -4.14866e-03, -4.14863e-03, -4.14860e-03, -4.14855e-03,
       -4.14848e-03, -4.14841e-03, -4.14834e-03, -4.14827e-03,
       -4.14821e-03, -4.14816e-03, -4.14811e-03, -4.14807e-03,
       -4.14803e-03, -4.14800e-03, -4.14797e-03, -4.14795e-03,
       -4.14792e-03, -4.14791e-03, -4.14789e-03, -4.14788e-03,
       -4.14787e-03, -4.14787e-03, -4.14787e-03, -4.14788e-03,
       -4.14789e-03, -4.14792e-03, -4.14795e-03, -4.14800e-03,
       -4.14804e-03, -4.14809e-03, -4.14814e-03, -4.14817e-03,
       -4.14820e-03, -4.14821e-03, -4.14821e-03, -4.14820e-03,
       -4.14818e-03, -4.14814e-03, -4.14809e-03, -4.14802e-03,
       -4.14793e-03, -4.14782e-03, -4.14770e-03, -4.14756e-03,
       -4.14743e-03, -4.14730e-03, -4.14717e-03, -4.14705e-03,
       -4.14694e-03, -4.14683e-03, -4.14674e-03, -4.14666e-03,
       -4.14661e-03, -4.14657e-03, -4.14656e-03, -4.14658e-03,
       -4.14662e-03, -4.14669e-03, -4.14677e-03, -4.14688e-03,
       -4.14699e-03, -4.14712e-03, -4.14726e-03, -4.14739e-03,
       -4.14751e-03, -4.14762e-03, -4.14772e-03, -4.14778e-03,
       -4.14782e-03, -4.14782e-03, -4.14780e-03, -4.14774e-03,
       -4.14766e-03, -4.14756e-03, -4.14745e-03, -4.14734e-03,
       -4.14723e-03, -4.14711e-03, -4.14700e-03, -4.14688e-03,
       -4.14676e-03, -4.14664e-03, -4.14652e-03, -4.14640e-03,
       -4.14628e-03, -4.14617e-03, -4.14606e-03, -4.14596e-03,
       -4.14587e-03, -4.14580e-03, -4.14575e-03, -4.14571e-03,
       -4.14570e-03, -4.14571e-03, -4.14574e-03, -4.14579e-03,
       -4.14585e-03, -4.14592e-03, -4.14599e-03, -4.14607e-03,
       -4.14613e-03, -4.14619e-03, -4.14624e-03, -4.14626e-03,
       -4.14627e-03, -4.14625e-03, -4.14622e-03, -4.14618e-03,
       -4.14613e-03, -4.14608e-03, -4.14603e-03, -4.14598e-03,
       -4.14591e-03, -4.14583e-03, -4.14574e-03, -4.14562e-03,
       -4.14549e-03, -4.14534e-03, -4.14519e-03, -4.14503e-03,
       -4.14487e-03, -4.14471e-03, -4.14457e-03, -4.14444e-03,
       -4.14434e-03, -4.14425e-03, -4.14419e-03, -4.14414e-03,
       -4.14412e-03, -4.14410e-03, -4.14409e-03, -4.14408e-03,
       -4.14407e-03, -4.14405e-03, -4.14403e-03, -4.14400e-03,
       -4.14396e-03, -4.14391e-03, -4.14387e-03, -4.14382e-03,
       -4.14377e-03, -4.14372e-03, -4.14366e-03, -4.14360e-03,
       -4.14353e-03, -4.14344e-03, -4.14334e-03, -4.14321e-03,
       -4.14307e-03, -4.14292e-03, -4.14276e-03, -4.14259e-03,
       -4.14242e-03, -4.14223e-03, -4.14205e-03, -4.14187e-03,
       -4.14169e-03, -4.14151e-03, -4.14134e-03, -4.14118e-03,
       -4.14104e-03, -4.14092e-03, -4.14082e-03, -4.14075e-03,
       -4.14070e-03, -4.14066e-03, -4.14063e-03, -4.14062e-03,
       -4.14060e-03, -4.14059e-03, -4.14057e-03, -4.14056e-03,
       -4.14055e-03, -4.14051e-03, -4.14048e-03, -4.14042e-03,
       -4.14035e-03, -4.14026e-03, -4.14016e-03, -4.14005e-03,
       -4.13994e-03, -4.13981e-03, -4.13968e-03, -4.13954e-03,
       -4.13939e-03, -4.13924e-03, -4.13907e-03, -4.13890e-03,
       -4.13872e-03, -4.13854e-03, -4.13836e-03, -4.13818e-03,
       -4.13800e-03, -4.13783e-03, -4.13766e-03, -4.13750e-03,
       -4.13734e-03, -4.13719e-03, -4.13705e-03, -4.13693e-03,
       -4.13683e-03, -4.13675e-03, -4.13670e-03, -4.13668e-03,
       -4.13668e-03, -4.13670e-03, -4.13675e-03, -4.13681e-03,
       -4.13686e-03, -4.13690e-03, -4.13694e-03, -4.13695e-03,
       -4.13694e-03, -4.13691e-03, -4.13686e-03, -4.13679e-03,
       -4.13671e-03, -4.13661e-03, -4.13650e-03, -4.13638e-03,
       -4.13625e-03, -4.13611e-03, -4.13595e-03, -4.13578e-03,
       -4.13561e-03, -4.13544e-03, -4.13526e-03, -4.13508e-03,
       -4.13491e-03, -4.13473e-03, -4.13456e-03, -4.13439e-03,
       -4.13422e-03, -4.13407e-03, -4.13394e-03, -4.13383e-03,
       -4.13375e-03, -4.13370e-03, -4.13367e-03, -4.13367e-03,
       -4.13369e-03, -4.13372e-03, -4.13376e-03, -4.13382e-03,
       -4.13388e-03, -4.13394e-03, -4.13399e-03, -4.13405e-03,
       -4.13410e-03, -4.13414e-03, -4.13416e-03, -4.13418e-03,
       -4.13417e-03, -4.13414e-03, -4.13409e-03, -4.13403e-03,
       -4.13394e-03, -4.13385e-03, -4.13375e-03, -4.13364e-03,
       -4.13351e-03, -4.13338e-03, -4.13324e-03, -4.13309e-03,
       -4.13294e-03, -4.13280e-03, -4.13265e-03, -4.13252e-03,
       -4.13239e-03, -4.13227e-03, -4.13216e-03, -4.13206e-03,
       -4.13197e-03, -4.13189e-03, -4.13181e-03, -4.13175e-03,
       -4.13169e-03, -4.13164e-03, -4.13161e-03, -4.13158e-03,
       -4.13155e-03, -4.13151e-03, -4.13148e-03, -4.13142e-03,
       -4.13136e-03, -4.13128e-03, -4.13120e-03, -4.13111e-03,
       -4.13103e-03, -4.13096e-03, -4.13090e-03, -4.13084e-03,
       -4.13078e-03, -4.13071e-03, -4.13062e-03, -4.13052e-03,
       -4.13039e-03, -4.13026e-03, -4.13012e-03, -4.12997e-03,
       -4.12983e-03, -4.12968e-03, -4.12955e-03, -4.12941e-03,
       -4.12929e-03, -4.12918e-03, -4.12907e-03, -4.12895e-03,
       -4.12884e-03, -4.12873e-03, -4.12863e-03, -4.12852e-03,
       -4.12843e-03, -4.12836e-03, -4.12830e-03, -4.12825e-03,
       -4.12819e-03, -4.12813e-03, -4.12807e-03, -4.12799e-03,
       -4.12792e-03, -4.12785e-03, -4.12778e-03, -4.12773e-03,
       -4.12769e-03, -4.12766e-03, -4.12764e-03, -4.12761e-03,
       -4.12758e-03, -4.12753e-03, -4.12746e-03, -4.12738e-03,
       -4.12728e-03, -4.12716e-03, -4.12704e-03, -4.12690e-03,
       -4.12676e-03, -4.12661e-03, -4.12646e-03, -4.12629e-03,
       -4.12613e-03, -4.12594e-03, -4.12576e-03, -4.12557e-03,
       -4.12538e-03, -4.12520e-03, -4.12504e-03, -4.12490e-03,
       -4.12479e-03, -4.12472e-03, -4.12468e-03, -4.12465e-03,
       -4.12465e-03, -4.12464e-03, -4.12464e-03, -4.12464e-03,
       -4.12463e-03, -4.12462e-03, -4.12460e-03, -4.12457e-03,
       -4.12454e-03, -4.12450e-03, -4.12446e-03, -4.12440e-03,
       -4.12432e-03, -4.12423e-03, -4.12413e-03, -4.12401e-03,
       -4.12389e-03, -4.12376e-03, -4.12362e-03, -4.12349e-03,
       -4.12335e-03, -4.12323e-03, -4.12311e-03, -4.12300e-03,
       -4.12288e-03, -4.12276e-03, -4.12264e-03, -4.12250e-03,
       -4.12235e-03, -4.12219e-03, -4.12204e-03, -4.12191e-03,
       -4.12180e-03, -4.12170e-03, -4.12164e-03, -4.12157e-03,
       -4.12153e-03, -4.12149e-03, -4.12145e-03, -4.12142e-03,
       -4.12140e-03, -4.12138e-03, -4.12136e-03, -4.12135e-03,
       -4.12133e-03, -4.12130e-03, -4.12127e-03, -4.12123e-03,
       -4.12117e-03, -4.12110e-03, -4.12102e-03, -4.12093e-03,
       -4.12082e-03, -4.12071e-03, -4.12057e-03, -4.12041e-03,
       -4.12024e-03, -4.12004e-03, -4.11983e-03, -4.11961e-03,
       -4.11940e-03, -4.11920e-03, -4.11901e-03, -4.11883e-03,
       -4.11866e-03, -4.11850e-03, -4.11835e-03, -4.11822e-03,
       -4.11810e-03, -4.11801e-03, -4.11793e-03, -4.11787e-03,
       -4.11784e-03, -4.11782e-03, -4.11782e-03, -4.11783e-03,
       -4.11785e-03, -4.11788e-03, -4.11791e-03, -4.11794e-03,
       -4.11796e-03, -4.11797e-03, -4.11798e-03, -4.11797e-03,
       -4.11795e-03, -4.11792e-03, -4.11789e-03, -4.11784e-03,
       -4.11778e-03, -4.11771e-03, -4.11763e-03, -4.11752e-03,
       -4.11740e-03, -4.11725e-03, -4.11708e-03, -4.11688e-03,
       -4.11666e-03, -4.11644e-03, -4.11621e-03, -4.11598e-03,
       -4.11578e-03, -4.11559e-03, -4.11543e-03, -4.11529e-03,
       -4.11518e-03, -4.11509e-03, -4.11501e-03, -4.11494e-03,
       -4.11487e-03, -4.11479e-03, -4.11470e-03, -4.11459e-03,
       -4.11447e-03, -4.11434e-03, -4.11423e-03, -4.11412e-03,
       -4.11403e-03, -4.11397e-03, -4.11392e-03, -4.11389e-03,
       -4.11386e-03, -4.11384e-03, -4.11380e-03, -4.11377e-03,
       -4.11371e-03, -4.11366e-03, -4.11361e-03, -4.11355e-03,
       -4.11348e-03, -4.11339e-03, -4.11330e-03, -4.11317e-03,
       -4.11303e-03, -4.11286e-03, -4.11269e-03, -4.11251e-03,
       -4.11233e-03, -4.11217e-03, -4.11201e-03, -4.11187e-03,
       -4.11174e-03, -4.11163e-03, -4.11153e-03, -4.11146e-03,
       -4.11139e-03, -4.11135e-03, -4.11132e-03, -4.11130e-03,
       -4.11129e-03, -4.11128e-03, -4.11126e-03, -4.11124e-03,
       -4.11121e-03, -4.11118e-03, -4.11114e-03, -4.11110e-03,
       -4.11105e-03, -4.11101e-03, -4.11097e-03, -4.11093e-03,
       -4.11089e-03, -4.11084e-03, -4.11078e-03, -4.11070e-03,
       -4.11061e-03, -4.11049e-03, -4.11035e-03, -4.11020e-03,
       -4.11003e-03, -4.10987e-03, -4.10969e-03, -4.10952e-03,
       -4.10935e-03, -4.10918e-03, -4.10903e-03, -4.10887e-03,
       -4.10873e-03, -4.10861e-03, -4.10851e-03, -4.10844e-03,
       -4.10838e-03, -4.10834e-03, -4.10832e-03, -4.10830e-03,
       -4.10829e-03, -4.10828e-03, -4.10827e-03, -4.10826e-03,
       -4.10825e-03, -4.10823e-03, -4.10822e-03, -4.10822e-03,
       -4.10822e-03, -4.10823e-03, -4.10824e-03, -4.10825e-03,
       -4.10825e-03, -4.10824e-03, -4.10821e-03, -4.10817e-03,
       -4.10812e-03, -4.10805e-03, -4.10798e-03, -4.10789e-03,
       -4.10779e-03, -4.10768e-03, -4.10756e-03, -4.10745e-03,
       -4.10733e-03, -4.10722e-03, -4.10711e-03, -4.10701e-03,
       -4.10691e-03, -4.10682e-03, -4.10672e-03, -4.10663e-03,
       -4.10656e-03, -4.10650e-03, -4.10645e-03, -4.10642e-03,
       -4.10640e-03, -4.10641e-03, -4.10642e-03, -4.10643e-03,
       -4.10645e-03, -4.10646e-03, -4.10647e-03, -4.10647e-03,
       -4.10647e-03, -4.10647e-03, -4.10647e-03, -4.10646e-03,
       -4.10646e-03, -4.10646e-03, -4.10645e-03, -4.10643e-03,
       -4.10639e-03, -4.10635e-03, -4.10627e-03, -4.10618e-03,
       -4.10607e-03, -4.10595e-03, -4.10581e-03, -4.10565e-03,
       -4.10549e-03, -4.10532e-03, -4.10515e-03, -4.10500e-03,
       -4.10486e-03, -4.10473e-03, -4.10462e-03, -4.10450e-03,
       -4.10439e-03, -4.10429e-03, -4.10421e-03, -4.10414e-03,
       -4.10409e-03, -4.10407e-03, -4.10406e-03, -4.10407e-03,
       -4.10410e-03, -4.10413e-03, -4.10416e-03, -4.10420e-03,
       -4.10423e-03, -4.10426e-03, -4.10429e-03, -4.10431e-03,
       -4.10432e-03, -4.10431e-03, -4.10429e-03, -4.10425e-03,
       -4.10421e-03, -4.10414e-03, -4.10407e-03, -4.10399e-03,
       -4.10391e-03, -4.10381e-03, -4.10372e-03, -4.10362e-03,
       -4.10352e-03, -4.10342e-03, -4.10333e-03, -4.10324e-03,
       -4.10315e-03, -4.10306e-03, -4.10299e-03, -4.10292e-03,
       -4.10287e-03, -4.10283e-03, -4.10280e-03, -4.10278e-03,
       -4.10278e-03, -4.10277e-03, -4.10278e-03, -4.10281e-03,
       -4.10285e-03, -4.10292e-03, -4.10300e-03, -4.10309e-03,
       -4.10319e-03, -4.10329e-03, -4.10339e-03, -4.10349e-03,
       -4.10359e-03, -4.10369e-03, -4.10377e-03, -4.10385e-03,
       -4.10390e-03, -4.10394e-03, -4.10395e-03, -4.10394e-03,
       -4.10390e-03, -4.10384e-03, -4.10376e-03, -4.10365e-03,
       -4.10354e-03, -4.10342e-03, -4.10330e-03, -4.10319e-03,
       -4.10309e-03, -4.10300e-03, -4.10293e-03, -4.10286e-03,
       -4.10280e-03, -4.10276e-03, -4.10272e-03, -4.10269e-03,
       -4.10267e-03, -4.10267e-03, -4.10268e-03, -4.10271e-03,
       -4.10275e-03, -4.10280e-03, -4.10286e-03, -4.10294e-03,
       -4.10302e-03, -4.10312e-03, -4.10322e-03, -4.10334e-03,
       -4.10346e-03, -4.10357e-03, -4.10368e-03, -4.10378e-03,
       -4.10386e-03, -4.10392e-03, -4.10395e-03, -4.10397e-03,
       -4.10396e-03, -4.10393e-03, -4.10390e-03, -4.10386e-03,
       -4.10382e-03, -4.10377e-03, -4.10374e-03, -4.10371e-03,
       -4.10367e-03, -4.10364e-03, -4.10360e-03, -4.10354e-03,
       -4.10347e-03, -4.10339e-03, -4.10330e-03, -4.10321e-03,
       -4.10314e-03, -4.10309e-03, -4.10307e-03, -4.10308e-03,
       -4.10312e-03, -4.10318e-03, -4.10325e-03, -4.10333e-03,
       -4.10342e-03, -4.10350e-03, -4.10359e-03, -4.10369e-03,
       -4.10378e-03, -4.10387e-03, -4.10395e-03, -4.10402e-03,
       -4.10407e-03, -4.10411e-03, -4.10412e-03, -4.10412e-03,
       -4.10409e-03, -4.10405e-03, -4.10400e-03, -4.10394e-03,
       -4.10387e-03, -4.10379e-03, -4.10371e-03, -4.10362e-03,
       -4.10352e-03, -4.10341e-03, -4.10329e-03, -4.10316e-03,
       -4.10303e-03, -4.10289e-03, -4.10276e-03, -4.10263e-03,
       -4.10252e-03, -4.10244e-03, -4.10239e-03, -4.10237e-03,
       -4.10237e-03, -4.10240e-03, -4.10246e-03, -4.10252e-03,
       -4.10258e-03, -4.10265e-03, -4.10269e-03, -4.10273e-03,
       -4.10275e-03, -4.10277e-03, -4.10277e-03, -4.10276e-03,
       -4.10276e-03, -4.10275e-03, -4.10274e-03, -4.10274e-03,
       -4.10274e-03, -4.10273e-03, -4.10273e-03, -4.10272e-03,
       -4.10269e-03, -4.10266e-03, -4.10260e-03, -4.10254e-03,
       -4.10247e-03, -4.10239e-03, -4.10232e-03, -4.10225e-03,
       -4.10217e-03, -4.10211e-03, -4.10204e-03, -4.10198e-03,
       -4.10192e-03, -4.10187e-03, -4.10183e-03, -4.10181e-03,
       -4.10181e-03, -4.10182e-03, -4.10186e-03, -4.10190e-03,
       -4.10195e-03, -4.10200e-03, -4.10205e-03, -4.10209e-03,
       -4.10213e-03, -4.10217e-03, -4.10220e-03, -4.10225e-03,
       -4.10229e-03, -4.10234e-03, -4.10239e-03, -4.10243e-03,
       -4.10245e-03, -4.10247e-03, -4.10248e-03, -4.10248e-03,
       -4.10246e-03, -4.10244e-03, -4.10242e-03, -4.10241e-03,
       -4.10239e-03, -4.10237e-03, -4.10235e-03, -4.10231e-03,
       -4.10226e-03, -4.10221e-03, -4.10215e-03, -4.10209e-03,
       -4.10202e-03, -4.10196e-03, -4.10191e-03, -4.10188e-03,
       -4.10186e-03, -4.10187e-03, -4.10190e-03, -4.10194e-03,
       -4.10200e-03, -4.10205e-03, -4.10211e-03, -4.10217e-03,
       -4.10221e-03, -4.10226e-03, -4.10229e-03, -4.10233e-03,
       -4.10235e-03, -4.10238e-03, -4.10240e-03, -4.10241e-03,
       -4.10243e-03, -4.10244e-03, -4.10245e-03, -4.10246e-03,
       -4.10247e-03, -4.10246e-03, -4.10245e-03, -4.10242e-03,
       -4.10236e-03, -4.10228e-03, -4.10219e-03, -4.10207e-03,
       -4.10195e-03, -4.10182e-03, -4.10169e-03, -4.10155e-03,
       -4.10143e-03, -4.10130e-03, -4.10120e-03, -4.10110e-03,
       -4.10104e-03, -4.10100e-03, -4.10099e-03, -4.10100e-03,
       -4.10102e-03, -4.10105e-03, -4.10107e-03, -4.10109e-03,
       -4.10112e-03, -4.10116e-03, -4.10120e-03, -4.10126e-03,
       -4.10133e-03, -4.10142e-03, -4.10151e-03, -4.10161e-03,
       -4.10170e-03, -4.10179e-03, -4.10186e-03, -4.10191e-03,
       -4.10193e-03, -4.10191e-03, -4.10188e-03, -4.10182e-03,
       -4.10175e-03, -4.10168e-03, -4.10161e-03, -4.10154e-03,
       -4.10148e-03, -4.10142e-03, -4.10137e-03, -4.10132e-03,
       -4.10129e-03, -4.10125e-03, -4.10122e-03, -4.10119e-03,
       -4.10117e-03, -4.10115e-03, -4.10114e-03, -4.10115e-03,
       -4.10116e-03, -4.10118e-03, -4.10120e-03, -4.10122e-03,
       -4.10124e-03, -4.10126e-03, -4.10128e-03, -4.10130e-03,
       -4.10133e-03, -4.10136e-03, -4.10140e-03, -4.10146e-03,
       -4.10152e-03, -4.10160e-03, -4.10166e-03, -4.10171e-03,
       -4.10174e-03, -4.10174e-03, -4.10172e-03, -4.10167e-03,
       -4.10160e-03, -4.10150e-03, -4.10137e-03, -4.10119e-03,
       -4.10095e-03, -4.10064e-03, -4.10022e-03, -4.09966e-03,
       -4.09893e-03, -4.09801e-03, -4.09688e-03, -4.09561e-03,
       -4.09500e-03, -4.09953e-03, -4.12944e-03, -4.26077e-03,
       -4.73576e-03, -6.21845e-03, -1.02896e-02, -2.02396e-02,
       -4.20853e-02, -8.55111e-02, -1.64223e-01, -2.95171e-01])
    
    nchnl=44
    topn=numpy.array(
       [0.005758, 0.005763, 0.005769, 0.005774, 0.005779, 0.005785,
        0.005791, 0.005799, 0.005808, 0.005818, 0.00583 , 0.005843,
        0.005858, 0.005876, 0.005896, 0.005919, 0.005946, 0.005976,
        0.006011, 0.006052, 0.006098, 0.006151, 0.006212, 0.006283,
        0.006363, 0.006456, 0.006563, 0.006685, 0.006826, 0.006987,
        0.007172, 0.007386, 0.007631, 0.007912, 0.008235, 0.008605,
        0.009032, 0.009521, 0.010081, 0.010727, 0.011469, 0.012321,
        0.0133  , 0.014425])
    tcls=numpy.array(
       [0.005763, 0.005769, 0.005774, 0.005779, 0.005785, 0.005791,
        0.005799, 0.005808, 0.005818, 0.00583 , 0.005843, 0.005858,
        0.005876, 0.005896, 0.005919, 0.005946, 0.005976, 0.006011,
        0.006052, 0.006098, 0.006151, 0.006212, 0.006283, 0.006363,
        0.006456, 0.006563, 0.006685, 0.006826, 0.006987, 0.007172,
        0.007386, 0.007631, 0.007912, 0.008235, 0.008605, 0.009032,
        0.009521, 0.010081, 0.010727, 0.011469, 0.012321, 0.0133  ,
        0.014425, 0.015717])
    
    
    return {"nsx": nsx,"swx": swx,"waveform": waveform},{"nchnl": nchnl,"topn": topn,"tcls": tcls}


# (topn + tcls) / 2
[gates,waveform] = default_gates_and_waveform()
system_spec.update(gates)
system_spec.update(waveform)

# ------- define true model
true_model = {
    "res": numpy.array([300, 1000]),
    "thk": numpy.array([20]),
    "peast": numpy.array([175]),
    "pnorth": numpy.array([100]),
    "ptop": numpy.array([30]),
    "pres": numpy.array([0.5]),
    "plngth1": numpy.array([100]),
    "plngth2": numpy.array([100]),
    "pwdth1": numpy.array([0.1]),
    "pwdth2": numpy.array([90]),
    "pdzm": numpy.array([90]),
    "pdip": numpy.array([60]),
}
