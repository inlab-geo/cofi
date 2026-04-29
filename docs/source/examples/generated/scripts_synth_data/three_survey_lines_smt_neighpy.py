"""
Invert three surveys line for a thin plate using the surrogate model
====================================================================

.. raw:: html

   <!-- Please leave the cell below as it is -->

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/airborne_em/airborne_em_three_lines_transmitters.ipynb
# 


######################################################################
# .. raw:: html
# 
#    <!-- Again, please don't touch the markdown cell above. We'll generate badge 
#         automatically from the above cell. -->
# 
# .. raw:: html
# 
#    <!-- This cell describes things related to environment setup, so please add more text 
#         if something special (not listed below) is needed to run this notebook -->
# 
# ..
# 
#    If you are running this notebook locally, make sure you’ve followed
#    `steps
#    here <https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally>`__
#    to set up the environment. (This
#    `environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/envs/environment.yml>`__
#    file specifies a list of packages required to run the notebooks)
# 


######################################################################
#    This notebook assumes that you have created a surrogate model by
#    executing the following two notebooks: - `Latin Hypercube
#    Sampling <./three_survey_lines_latin_hypercube_sampling.ipynb>`__ -
#    `Surrogate model
#    creation <./three_survey_lines_surrogate_model_creation.ipynb>`__
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi
# !pip install git+https://github.com/JuergHauser/PyP223.git
# !pip install smt
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/examples/vtem_max

######################################################################
#

# If this notebook is run locally PyP223 and smt need to be installed separately by uncommenting the following lines, 
# that is by removing the # and the white space between it and the exclamation mark.
# !pip install git+https://github.com/JuergHauser/PyP223.git
# !pip install smt

######################################################################
#

import pickle
import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cofi
import arviz
from vtem_max_forward_lib import (
    problem_setup, 
    system_spec, 
    survey_setup, 
    ForwardWrapper, 
    plot_predicted_profile, 
    plot_transient, 
    plot_plate_faces, 
    plot_plate_faces_single
)

numpy.random.seed(42)

######################################################################
#


######################################################################
# Background
# ----------
# 
# This example inverts three survey line of VTEM max data using the
# vertical component for a thin plate target. It thus becomes possible to
# invert for the easting,northing, depth of the plate reference point, the
# plate dip and plate azimuth. Solving the forward problem, that is
# calculating the objective function, usess the surrogate model that has
# been created by the `Kriging
# approach <./three_survey_lines_surrogate_model_creation.ipynb>`__
# applied to the `latin hypercube
# samples <three_survey_lines_latin_hypercube_sampling.ipynb>`__ of the
# objective function.
# 


######################################################################
# Problem definition
# ------------------
# 

tx_min = 115
tx_max = 281
tx_interval = 15
ty_min = 25
ty_max = 176
ty_interval = 75
tx_points = numpy.arange(tx_min, tx_max, tx_interval)
ty_points = numpy.arange(ty_min, ty_max, ty_interval)
n_transmitters = len(tx_points) * len(ty_points)
tx, ty = numpy.meshgrid(tx_points, ty_points)
tx = tx.flatten()
ty = ty.flatten()

######################################################################
#

fiducial_id = numpy.arange(len(tx))
line_id = numpy.zeros(len(tx), dtype=int)
line_id[ty==ty_points[0]] = 0
line_id[ty==ty_points[1]] = 1
line_id[ty==ty_points[2]] = 2

######################################################################
#

survey_setup = {
    "tx": tx,                                                   # transmitter easting/x-position
    "ty": ty,                                                   # transmitter northing/y-position
    "tz": numpy.array([50]*n_transmitters),                     # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90]*n_transmitters)),    # transmitter azimuth
    "tincl": numpy.deg2rad(numpy.array([6]*n_transmitters)),    # transmitter inclination
    "rx": tx,                                                   # receiver easting/x-position
    "ry": numpy.array([100]*n_transmitters),                    # receiver northing/y-position
    "rz": numpy.array([50]*n_transmitters),                     # receiver height/z-position
    "trdx": numpy.array([0]*n_transmitters),                    # transmitter receiver separation inline
    "trdy": numpy.array([0]*n_transmitters),                    # transmitter receiver separation crossline
    "trdz": numpy.array([0]*n_transmitters),                    # transmitter receiver separation vertical
    "fiducial_id": fiducial_id,                                 # unique id for each transmitter
    "line_id": line_id                  # id for each line
}

######################################################################
#

true_model = {
    "res": numpy.array([300, 1000]), 
    "thk": numpy.array([20]), 
    "peast": numpy.array([175]), 
    "pnorth": numpy.array([100]), 
    "ptop": numpy.array([30]), 
    "pres": numpy.array([0.1]), 
    "plngth1": numpy.array([100]), 
    "plngth2": numpy.array([100]), 
    "pwdth1": numpy.array([0.1]), 
    "pwdth2": numpy.array([90]), 
    "pdzm": numpy.array([75]),
    "pdip": numpy.array([60])
}

######################################################################
#

forward = ForwardWrapper(true_model, problem_setup, system_spec, survey_setup,
                         ["pdip","pdzm", "peast", "ptop", "pwdth2"], data_returned=["vertical"])

######################################################################
#

# check the order of parameters in a model vector
forward.params_to_invert

######################################################################
#

true_param_value = numpy.array([60.,65., 175., 30., 90.])

######################################################################
#


######################################################################
# Ensemble method using the surrogate model
# -----------------------------------------
# 

sm = None
filename = "kriging_surrogate_model.pkl"
with open(filename, "rb") as f:
   sm = pickle.load(f)

######################################################################
#


######################################################################
# **Initialise a model for inversion**
# 

init_param_value = numpy.array([45, 90, 160, 35, 80])
m_min = numpy.array([15, 35, 155, 30, 65])
m_max = numpy.array([75, 145, 185, 40, 115])

######################################################################
#


######################################################################
# **Define helper functions for CoFI**
# 

def my_objective(model):
    val=sm.predict_values(numpy.array([model]))[0][0]
    if val<1e-3:
        return 1e-3
    else:
        return val
        
def my_log_likelihood(model):
    return -0.5 * my_objective(model)

def my_log_prior(model):    # uniform distribution
    for i in range(len(model)):
        if model[i] < m_min[i] or model[i] > m_max[i]: return -numpy.inf
    return 0.0 # model lies within bounds -> return log(1)

######################################################################
#


######################################################################
# **Define CoFI problem**
# 

my_problem = cofi.BaseProblem()
my_problem.set_objective(my_objective)

######################################################################
#


######################################################################
# **Define CoFI options**
# 

inv_options = cofi.InversionOptions()
inv_options.set_tool("neighpy")
inv_options.suggest_solver_params()

######################################################################
#

inv_options.set_params(
    n_samples_per_iteration=100,
    n_initial_samples=10,
    n_resample=8000,
    n_iterations=100,
    bounds=numpy.array([m_min, m_max]).T,
    n_cells_to_resample=10,
    n_walkers=4
)
######## Run it
inv = cofi.Inversion(my_problem, inv_options)
inv_result = inv.run()

######## Check result
print(f"The inversion result from `neighpy`:")
inv_result.summary()

######################################################################
#


######################################################################
# Plotting
# --------
# 

arviz.style.use("default")
var_names = [
    "Dip (\u00b0)", 
    "Dip azimuth (\u00b0)", 
    "Easting (m)", 
    "Depth (m)", 
    "Width (m)"
]

var_lines=(
    ('Dip (\u00b0)', {}, 60),
        ('Dip azimuth (\u00b0)', {}, 65),
        ('Easting (m)', {}, 175),
        ('Depth (m)', {}, 30),
        ('Width (m)', {}, 90)
)
d = {k: v for k, v in zip(var_names, inv_result.appraisal_samples.T)}
az_idata = arviz.convert_to_inference_data(d)
arviz.plot_trace(az_idata.sel(draw=slice(2000,None)),lines=var_lines)
plt.tight_layout()

######################################################################
#

arviz.style.use("default")

_, axes = plt.subplots(5, 5, figsize=(12,12))
arviz.plot_pair(
    az_idata.sel(draw=slice(4000,None)), 
    marginals=True, 
    ax = axes
)
plt.tight_layout()

######################################################################
#

arviz.style.use("default")

_, axes = plt.subplots(2, 2)
axes[1,1].axis("off")
plot_plate_faces(
    "plate_true", forward, true_param_value, 
    axes[0,0], axes[0,1], axes[1,0], color="purple", label="True model"
)
plot_plate_faces(
    "plate_init", forward, init_param_value, 
    axes[0,0], axes[0,1], axes[1,0], color="green", label="Starting model"
)


plt.tight_layout()


ichain=0
idraw=2500
sample=numpy.zeros(5)

sample[0]=az_idata.posterior['Dip (\u00b0)'][ichain][idraw]
sample[1]=az_idata.posterior['Dip azimuth (\u00b0)'][ichain][idraw]
sample[2]=az_idata.posterior['Easting (m)'][ichain][idraw]
sample[3]=az_idata.posterior['Depth (m)'][ichain][idraw]
sample[4]=az_idata.posterior['Width (m)'][ichain][idraw]
plot_plate_faces(
    "plate_inverted", forward, sample, 
    axes[0,0], axes[0,1], axes[1,0], color="red", label="Posterior sample", linestyle="dotted"
)

point = Line2D([0], [0], label='Fiducial', marker='o', markersize=5, 
         markeredgecolor='orange', markerfacecolor='orange', linestyle='')

handles, labels = axes[1,0].get_legend_handles_labels()
handles.extend([point])

axes[1,0].legend(handles=handles,bbox_to_anchor=(1.04, 0), loc="lower left")


# plot 10 randomly selected samples of the posterior distirbution
for i in range(10):
    idraw=numpy.random.randint(2000,5000)
    sample[0]=az_idata.posterior['Dip (\u00b0)'][ichain][idraw]
    sample[1]=az_idata.posterior['Dip azimuth (\u00b0)'][ichain][idraw]
    sample[2]=az_idata.posterior['Easting (m)'][ichain][idraw]
    sample[3]=az_idata.posterior['Depth (m)'][ichain][idraw]
    sample[4]=az_idata.posterior['Width (m)'][ichain][idraw]
    plot_plate_faces(
    "plate_inverted", forward, sample, 
    axes[0,0], axes[0,1], axes[1,0], color="red", label="Posterior sample", linestyle="dotted"
)


######################################################################
#


######################################################################
# --------------
# 
# Watermark
# ---------
# 
# .. raw:: html
# 
#    <!-- Feel free to add more modules in the watermark_list below, if more packages are used -->
# 
# .. raw:: html
# 
#    <!-- Otherwise please leave the below code cell unchanged -->
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1