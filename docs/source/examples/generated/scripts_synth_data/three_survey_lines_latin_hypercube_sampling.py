"""
Latin hypercube sampling of the objective function in three survey line example
===============================================================================

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

import numpy
import smt
import smt.sampling_methods
import tqdm
from vtem_max_forward_lib import (
    problem_setup, 
    system_spec, 
    survey_setup, 
    ForwardWrapper
)

numpy.random.seed(42)

######################################################################
#


######################################################################
# Background
# ----------
# 
# The time required to solve the forward problem is what frequently
# dominates the time required to solve an inverse problem. An approximate
# mathematical model also known as a surrogate model may be constructed
# and used instead of the full forward problem with the advantage that
# evaluating the approximate model typically only takes a fraction of the
# time required to solve the full forward problem. The surrogate modelling
# toolbox (https://github.com/SMTorg/smt) is a Python library that
# provides a range of surrogate modelling methods.
# 
# https://github.com/SMTorg/smt/blob/master/tutorial/SMT_Tutorial.ipynb
# 
# Here we use the surrogate modelling toolbox to creata surrogate model
# for the objective function used in the `three survey line
# example <http://127.0.0.1:8888/notebooks/three_survey_lines.ipynb>`__.
# This notebook generates training and test/validation samples of the
# objective function using latin hypercube sampling. Compared to random
# sampling `latin hypercube
# sampling <https://en.wikipedia.org/wiki/Latin_hypercube_sampling>`__
# seeks to ensure that the set of random numbers is representative of the
# real variability. The training samples are used to create the surrogate
# model and the test samples are used to assess its predictive power.
# 
# For large numbers of samples it can be be convenient to convert the
# notebook into a script and run it from the comand line, using the
# following command to create the script.
# 
# ``jupyter nbconvert --to script three_survey_lines_latin_hypercube_sampling.ipynb``
# 

# set the number of training and test samples
ntrain=100
ntest=25

######################################################################
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
    "thk": numpy.array([25]), 
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

true_param_value = numpy.array([60,65, 175, 30, 90])
xtrue=true_param_value

######################################################################
#


######################################################################
# **Generate synthetic data**
# 

# The data 
absolute_noise= 0.05

# create data and ad a realisation of the noise
data_pred_true = forward(true_param_value)
data_obs = data_pred_true + numpy.random.randn(len(data_pred_true))*absolute_noise

# define data covariance matrix
sigma=absolute_noise
Cdinv=numpy.identity(len(data_obs))*(1.0/(sigma*sigma))

######################################################################
#


######################################################################
# Perform Latin Hypercube sampling
# --------------------------------
# 


######################################################################
# **Define objective function**
# 

def my_objective(model):
    dpred = forward(model)
    residual = dpred - data_obs
    return residual.T @ Cdinv @ residual

######################################################################
#

ndim=len(true_param_value)
xlimits=numpy.array([[10,80],[30,150],[150,190],[25,45],[60,120]])

######################################################################
#

sampling = smt.sampling_methods.LHS(xlimits=xlimits,seed=42)
xtrain=sampling(ntrain)
ytrain=[]
xtest=sampling(ntest)
ytest=[]


######################################################################
#

for x in tqdm.tqdm(xtrain):
    ytrain.append(my_objective(x))
for x in tqdm.tqdm(xtest):
    ytest.append(my_objective(x))

######################################################################
#

xtrain=numpy.array(xtrain)
ytrain=numpy.array(ytrain)

xtest=numpy.array(xtest)
ytest=numpy.array(ytest)

######################################################################
#

with open('three_survey_lines_lhs.npy', 'wb') as f:
    numpy.save(f,ndim)
    numpy.save(f,xlimits)
    numpy.save(f,xtrain)
    numpy.save(f,ytrain)
    numpy.save(f,xtest)
    numpy.save(f,ytest)

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# =========
# 
# .. raw:: html
# 
#    <!-- Feel free to add more modules in the watermark_list below, if more packages are used -->
# 
# .. raw:: html
# 
#    <!-- Otherwise please leave the below code cell unchanged -->
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib","smt"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#



######################################################################
#
# sphinx_gallery_thumbnail_number = -1