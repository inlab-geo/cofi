"""
Receiver Function
=================

.. raw:: html

   <!-- Please leave the cell below as it is -->

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/receiver_function/receiver_function.ipynb
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
# Receiver functions are a class of seismic data used to study
# discontinuities (layering) in the Earth’s crust. At each discontinuity,
# P-to-S conversions occur, introducing complexity in the waveform. By
# deconvolving horizontal- and vertical-channel waveforms from earthquakes
# at teleseismic distances, we can isolate information about these
# conversions, and hence learn about the crustal structure. This
# deconvolved signal is the receiver function, and has a highly non-linear
# dependence on the local crustal properties.
# 
# We refer you to the paper below for description of the algorithms and
# the forward kernel we use:
# 
# *Genetic algorithm inversion for receiver functions with application to
# crust and uppermost mantle structure beneath Eastern Australia*,
# Shibutani, T., Kennett, B. and Sambridge, M., Geophys. Res. Lett., 23 ,
# No. 4, 1829-1832, 1996.
# 
# In this notebook, we run inversion on a toy model with optimisation and
# parallel sampling.
# 


######################################################################
# 0. Import modules
# -----------------
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi geo-espresso

######################################################################
#

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import arviz
import emcee
import multiprocessing

import cofi
import espresso

######################################################################
#

# randomness is used to initialise emcee walkers starting points
np.random.seed(42)

######################################################################
#


######################################################################
# We are going to use the receiver function kernel wrapped in
# ```espresso`` <https://geo-espresso.readthedocs.io/en/latest/user_guide/contrib/generated/_receiver_function/index.html>`__,
# with calls to Fortran routines developed by Takuo Shibutani in the
# backend.
# 

my_receiver_function = espresso.ReceiverFunctionInversion(example_number=4)

######################################################################
#


######################################################################
# Consider a model setup of ``n`` layers described with 3 parameters for
# each layer. ``model`` is a NumPy array of dimension ``[nlayers,3]``. The
# values in ``model[:,0]`` give the depths of discontinuities in the
# model, while ``model[:,1]`` contains the S-wave speed above the
# interface. ``model[:,2]`` is the ratio of S-wave speed to P-wave speed.
# The maximum depth of discontinuity that can be considered is 60km.
# 
# In this example, we fix the ratio of S-wave speed to P-wave speed, and
# treat the interface depths and velocities of 5 layers as unknowns.
# 
# In order to better understand the complexity and non-linear nature of
# seismic receiver function inversion, we have included a few illustrative
# animations. These animations highlight the substantial influence of
# velocities and, more prominently, interface depths on the resulting
# receiver functions.
# 


######################################################################
# .. figure:: https://github.com/inlab-geo/cofi-examples/blob/main/examples/receiver_function/depth_layer3_anim4.gif?raw=true
#    :alt: depth_layer3_anim4.gif
# 
#    depth_layer3_anim4.gif
# 


######################################################################
# .. figure:: https://github.com/inlab-geo/cofi-examples/blob/main/examples/receiver_function/vel_layer3_anim4_400f.gif?raw=true
#    :alt: vel_layer3_anim4_400f.gif
# 
#    vel_layer3_anim4_400f.gif
# 


######################################################################
# This is a non-linear problem which can be highly sensitive to the
# starting model.
# 
# .. figure:: https://github.com/inlab-geo/cofi-examples/blob/main/examples/receiver_function/3Dsurf_x40y10_v55_l270_35.png?raw=true
#    :alt: 3Dsurf_x40y10_v55_l270_35
# 
#    3Dsurf_x40y10_v55_l270_35
# 
# Here we set a starting model that is reasonably close to the true model,
# so that the optimisation converges.
# 

null_model = my_receiver_function.starting_model 

print(f"Number of model parameters in this example: {null_model.size}")
my_receiver_function._model_setup(null_model)

######################################################################
#


######################################################################
# Let’s plot the starting Earth model.
# 

my_receiver_function.plot_model(null_model);

######################################################################
#


######################################################################
# Now we calculate the receiver function and plot it
# 

predicted_data = my_receiver_function.forward(null_model)
observed_data = my_receiver_function.data 
my_receiver_function.plot_data(
    data1=predicted_data, 
    data2=observed_data, 
    label="predicted_data", 
    label2="observed_data", 
)
plt.legend();

######################################################################
#


######################################################################
# 1. Solve with an optimiser
# --------------------------
# 


######################################################################
# 1.1 Define BaseProblem
# ~~~~~~~~~~~~~~~~~~~~~~
# 

my_problem = cofi.BaseProblem()

######################################################################
#


######################################################################
# In preparation for optimisation:
# 

def my_misfit(model, include_uncertainty=False):
    predicted_data = my_receiver_function.forward(model)
    misfit_val = my_receiver_function.misfit(predicted_data, observed_data)
    if math.isnan(misfit_val):
        return float("inf")
    return misfit_val

my_problem.set_objective(my_misfit)
my_problem.set_initial_model(null_model)

my_problem.summary()

######################################################################
#


######################################################################
# 1.2 Define InversionOptions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

my_options_optimiser = cofi.InversionOptions()
my_options_optimiser.set_tool("scipy.optimize.minimize")
my_options_optimiser.set_params(method="Nelder-Mead")   # Nelder-Mead or COBYLA

######################################################################
#


######################################################################
# 1.3 Define Inversion and run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

inv_optimiser = cofi.Inversion(my_problem, my_options_optimiser)
my_result_optimiser = inv_optimiser.run()
my_result_optimiser.summary()

######################################################################
#

print("Inversion result:    ", my_result_optimiser.model)
print("Reference good model:", my_receiver_function.good_model)

######################################################################
#


######################################################################
# 1.4 Plotting
# ~~~~~~~~~~~~
# 

predicted_data = my_receiver_function.forward(my_result_optimiser.model)
my_receiver_function.plot_data(
    data1=predicted_data, 
    data2=observed_data, 
    label="predicted_data", 
    label2="observed_data", 
)
plt.legend();

######################################################################
#


######################################################################
# 2. Solve with a sampler
# -----------------------
# 
# 2.1 Enrich BaseProblem
# ~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# In preparation for sampling:
# 

def my_log_likelihood(model):
    data1 = my_receiver_function.data
    data2 = my_receiver_function.forward(model)
    log_likelihood = my_receiver_function.log_likelihood(data1, data2)
    return log_likelihood

def my_log_prior(model):
    log_prior = my_receiver_function.log_prior(model)
    return log_prior

ndim = my_receiver_function.model_size

my_problem.set_model_shape(ndim)
my_problem.set_log_likelihood(my_log_likelihood)
my_problem.set_log_prior(my_log_prior)

my_problem.summary()

######################################################################
#

nwalkers = 12
nsteps = 25000
walkers_start = null_model + 1e-1 * np.random.randn(nwalkers, ndim)

######################################################################
#


######################################################################
# We can run ``emcee`` in parallel. Some additional preparation:
# 


######################################################################
# 2.2 Define InversionOptions, Inversion and run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

import warnings
warnings.filterwarnings("ignore")

my_options_sampler = cofi.InversionOptions()
my_options_sampler.set_tool("emcee")
my_options_sampler.set_params(
    nwalkers=nwalkers,
    nsteps=nsteps,
    initial_state=walkers_start,
    progress=True,
)
inv_sampler = cofi.Inversion(my_problem, my_options_sampler)
inv_result_sampler = inv_sampler.run()

######################################################################
#

inv_result_sampler.summary()

######################################################################
#


######################################################################
# 2.3 Plotting
# ~~~~~~~~~~~~
# 

var_names = (
    "depth1 (km)", 
    "velocity1 (km/s)", 
    "depth2 (km)", 
    "velocity2 (km/s)", 
    "depth3 (km)", 
    "velocity3 (km/s)", 
)
az_inf_data = inv_result_sampler.to_arviz(var_names=var_names)
az_inf_data

######################################################################
#

arviz.plot_trace(az_inf_data);
plt.tight_layout();

######################################################################
#


######################################################################
# The walkers start in small distributions around some chosen values and
# then they quickly wander and start exploring the full posterior
# distribution. In fact, after a relatively small number of steps, the
# samples seem pretty well “burnt-in”. That is a hard statement to make
# quantitatively, but we can look at an estimate of the integrated
# autocorrelation time (see Emcee’s package the -`Autocorrelation analysis
# & convergence
# tutorial <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`__
# for more details):
# 

tau = inv_result_sampler.sampler.get_autocorr_time()
print(f"autocorrelation time: {tau}")

######################################################################
#


######################################################################
# Let’s discard the initial 300 steps and make a corner plot:
# 

az_inf_data_after_300 = az_inf_data.sel(draw=slice(300,None))

arviz.plot_pair(
    az_inf_data_after_300, 
    marginals=True, 
)

print("Reference good model:", my_receiver_function.good_model)

######################################################################
#

true_model = my_receiver_function.good_model
mean_sample = np.array(az_inf_data["posterior"].mean().to_array())
median_sample = np.array(az_inf_data["posterior"].median().to_array())

print("Mean of samples:     ", mean_sample)
print("Reference good model:", true_model)

my_receiver_function.plot_model(true_model, mean_sample, "true_model", "mean_sample")
plt.legend();

######################################################################
#

mean_sample_predicted_data = my_receiver_function.forward(mean_sample)
my_receiver_function.plot_data(
    observed_data, 
    mean_sample_predicted_data,
    "observed_data",
    "mean_sample_predicted_data",
);
plt.legend();

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

watermark_list = ["cofi", "espresso", "numpy", "matplotlib", "emcee", "arviz"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1