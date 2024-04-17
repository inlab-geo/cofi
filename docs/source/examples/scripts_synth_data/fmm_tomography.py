"""
Seismic Wave Tomography via Fast Marching
=========================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/fmm_tomography/fmm_tomography.ipynb
# 


######################################################################
#    If you are running this notebook locally, make sure you’ve followed
#    `steps
#    here <https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally>`__
#    to set up the environment. (This
#    `environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/envs/environment.yml>`__
#    file specifies a list of packages required to run the notebooks)
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
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/examples/fmm_tomography

######################################################################
#

import numpy as np
import matplotlib.pyplot as plt
import pprint

import cofi
import espresso

######################################################################
#


######################################################################
# In this notebook, we use ``cofi`` to run a seismic wave tomography
# example, in which the forward calculation is based on the Fast Marching
# Fortran code by Nick Rawlinson. The Fast Marching code is wrapped in
# package ``espresso``.
# 
# Theoretical background
# ----------------------
# 

# display theory on travel time tomography
from IPython.display import display, Markdown

with open("../../theory/geo_travel_time_tomography.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#


######################################################################
# In this notebook we form and solve a travel time tomography problem
# using model damping and 2nd derivative smoothing. For forward modelling,
# a fast marching wave front tracker is used, utilizing the Fast Marching
# Fortran code within the package
# ```FMTOMO`` <http://iearth.edu.au/codes/FMTOMO/>`__ by Nick Rawlinson.
# 


######################################################################
# Problem setup
# ~~~~~~~~~~~~~
# 
# Before we starting working with ``cofi``, let’s get familiar with the
# problem itself.
# 
# Below is a plot of the true model and the paths generated from this
# model. As you can see, there are two anomalies, one with lower velocity
# (red, top left) and the other with higher velocity (blue, bottom right).
# 

fmm = espresso.FmmTomography()

fmm.plot_model(fmm.good_model, with_paths=True);

######################################################################
#

pprint.pprint(fmm.metadata)

######################################################################
#


######################################################################
# 1. Define the problem
# ---------------------
# 

# get problem information from  espresso FmmTomography
model_size = fmm.model_size         # number of model parameters
model_shape = fmm.model_shape       # 2D spatial grids
data_size = fmm.data_size           # number of data points
ref_start_slowness = fmm.starting_model

######################################################################
#

# define CoFI BaseProblem
fmm_problem = cofi.BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)

######################################################################
#

# add regularization: damping + smoothing
damping_factor = 50
smoothing_factor = 5e3
reg_damping = damping_factor * cofi.utils.QuadraticReg(
    model_shape=model_shape, 
    weighting_matrix="damping", 
    reference_model=ref_start_slowness
)
reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
    model_shape=model_shape,
    weighting_matrix="smoothing"
)
reg = reg_damping + reg_smoothing

######################################################################
#

def objective_func(slowness, reg, sigma, reduce_data=None):  # reduce_data=(idx_from, idx_to)
    if reduce_data is None: idx_from, idx_to = (0, fmm.data_size)
    else: idx_from, idx_to = reduce_data
    ttimes = fmm.forward(slowness)
    residual = fmm.data[idx_from:idx_to] - ttimes[idx_from:idx_to]
    data_misfit = residual.T @ residual / sigma**2
    model_reg = reg(slowness)
    return  data_misfit + model_reg

def gradient(slowness, reg, sigma, reduce_data=None):       # reduce_data=(idx_from, idx_to)
    if reduce_data is None: idx_from, idx_to = (0, fmm.data_size)
    else: idx_from, idx_to = reduce_data
    ttimes, A = fmm.forward(slowness, return_jacobian=True)
    ttimes = ttimes[idx_from:idx_to]
    A = A[idx_from:idx_to]
    data_misfit_grad = -2 * A.T @ (fmm.data[idx_from:idx_to] - ttimes) / sigma**2
    model_reg_grad = reg.gradient(slowness)
    return  data_misfit_grad + model_reg_grad

def hessian(slowness, reg, sigma, reduce_data=None):        # reduce_data=(idx_from, idx_to)
    if reduce_data is None: idx_from, idx_to = (0, fmm.data_size)
    else: idx_from, idx_to = reduce_data
    A = fmm.jacobian(slowness)[idx_from:idx_to]
    data_misfit_hess = 2 * A.T @ A / sigma**2 
    model_reg_hess = reg.hessian(slowness)
    return data_misfit_hess + model_reg_hess

######################################################################
#

sigma =  0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals

fmm_problem.set_objective(objective_func, args=[reg, sigma, None])
fmm_problem.set_gradient(gradient, args=[reg, sigma, None])
fmm_problem.set_hessian(hessian, args=[reg, sigma, None])

######################################################################
#


######################################################################
# Review what information is included in the ``BaseProblem`` object:
# 

fmm_problem.summary()

######################################################################
#


######################################################################
# 2. Define the inversion options
# -------------------------------
# 

my_options = cofi.InversionOptions()

# cofi's own simple newton's matrix-based optimization solver
my_options.set_tool("cofi.simple_newton")
my_options.set_params(num_iterations=5, step_length=1, verbose=True)

######################################################################
#


######################################################################
# Review what’s been defined for the inversion we are about to run:
# 

my_options.summary()

######################################################################
#


######################################################################
# 3. Start an inversion
# ---------------------
# 

inv = cofi.Inversion(fmm_problem, my_options)
inv_result = inv.run()
inv_result.summary()

######################################################################
#


######################################################################
# 4. Plotting
# -----------
# 

fmm.plot_model(inv_result.model);            # inverted model
fmm.plot_model(fmm.good_model);       # true model

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

watermark_list = ["cofi", "espresso", "numpy", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1