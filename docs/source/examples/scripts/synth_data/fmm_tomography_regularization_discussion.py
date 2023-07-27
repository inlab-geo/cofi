"""
Seismic Wave Tomography via Fast Marching - Regularization Demo
===============================================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/fmm_tomography/fmm_tomography.ipynb
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
# .. raw:: html
# 
#    <!-- TODO - background introduction for this problem. -->
# 
# In this notebook, we would like to demonstrate the capability of CoFI to
# easily switch between different types of regularizations.
# 
# We will use ``cofi`` to run a seismic wave tomography example, in which
# the forward calculation is based on the Fast Marching Fortran code by
# Nick Rawlinson. The Fast Marching code is wrapped in package
# ``espresso``.
# 
# We refer you to `fmm_tomography.ipynb <./fmm_tomography.ipynb>`__ for
# further theretical details.
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

import numpy as np
import matplotlib.pyplot as plt
import pprint

import cofi
import espresso

######################################################################
#


######################################################################
# Understanding the inference problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# 1. Problem setup and utilities
# ------------------------------
# 

# get problem information from  espresso FmmTomography
model_size = fmm.model_size         # number of model parameters
model_shape = fmm.model_shape       # 2D spatial grids
data_size = fmm.data_size           # number of data points
ref_start_slowness = fmm.starting_model

######################################################################
#

def objective_func(slowness, reg, sigma):
    ttimes = fmm.forward(slowness)
    residual = fmm.data - ttimes
    data_misfit = residual.T @ residual / sigma**2
    model_reg = reg(slowness)
    return  data_misfit + model_reg

def gradient(slowness, reg, sigma):
    ttimes, A = fmm.forward(slowness, return_jacobian=True)
    data_misfit_grad = -2 * A.T @ (fmm.data - ttimes) / sigma**2
    model_reg_grad = reg.gradient(slowness)
    return  data_misfit_grad + model_reg_grad

def hessian(slowness, reg, sigma):
    A = fmm.jacobian(slowness)
    data_misfit_hess = 2 * A.T @ A / sigma**2 
    model_reg_hess = reg.hessian(slowness)
    return data_misfit_hess + model_reg_hess

######################################################################
#


######################################################################
# 2. Invert with quadratic smoothing and damping regularization terms
# -------------------------------------------------------------------
# 
# 2.1 Define BaseProblem
# ~~~~~~~~~~~~~~~~~~~~~~
# 

# define CoFI BaseProblem
fmm_problem_quadratic_reg = cofi.BaseProblem()
fmm_problem_quadratic_reg.set_initial_model(ref_start_slowness)

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

sigma =  0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals

fmm_problem_quadratic_reg.set_objective(objective_func, args=[reg, sigma])
fmm_problem_quadratic_reg.set_gradient(gradient, args=[reg, sigma])
fmm_problem_quadratic_reg.set_hessian(hessian, args=[reg, sigma])

######################################################################
#


######################################################################
# 2.2 Define InversionOptions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

my_options = cofi.InversionOptions()

# cofi's own simple newton's matrix-based optimization solver
my_options.set_tool("cofi.simple_newton")
my_options.set_params(num_iterations=6, step_length=1, verbose=True)

######################################################################
#


######################################################################
# 2.3 Start an inversion
# ~~~~~~~~~~~~~~~~~~~~~~
# 

inv = cofi.Inversion(fmm_problem_quadratic_reg, my_options)
inv_result_quadratic_reg = inv.run()
inv_result_quadratic_reg.summary()

######################################################################
#


######################################################################
# 2.4 Plotting
# ~~~~~~~~~~~~
# 

fmm.plot_model(inv_result_quadratic_reg.model);            # inverted model
fmm.plot_model(fmm.good_model);       # true model

######################################################################
#


######################################################################
# --------------
# 
# 3. Invert with Gaussian prior as regularization term
# ----------------------------------------------------
# 
# Instead of using a smoothing and damping regularization, in this
# section, we use a model covariance matrix and prior model.
# 
# :math:`\chi_{P}^{2}=\left(\mathbf{y} -\mathbf{f}(\mathbf{m})\right)^T C_d^{-1} \left(\mathbf{y} -\mathbf{f}(\mathbf{m})\right) + \left( \mathbf{m} - \mathbf{m}_p \right)^T C_p^{-1} \left( \mathbf{m} - \mathbf{m}_p \right)`
# 
# :math:`\Delta \mathbf{m}= ({J}^T {C}_d^{-1} {J}+{C}_p^{-1})^{-1} ({J}^T{C}_d^{-1} (\mathbf{y}-\mathbf{f}(\mathbf{m}))+{C}_p^{-1}(\mathbf{m}_p-\mathbf{m}))`
# 
# We can use CoFI’s utility module to help us generate a the Gaussian
# prior term.
# 
# 3.1 Define BaseProblem
# ~~~~~~~~~~~~~~~~~~~~~~
# 

# define CoFI BaseProblem
fmm_problem_gaussian_prior = cofi.BaseProblem()
fmm_problem_gaussian_prior.set_initial_model(ref_start_slowness)

######################################################################
#

# add regularization: Gaussian prior
corrx = 3.0
corry = 3.0
sigma_slowness = 0.002
gaussian_prior = cofi.utils.GaussianPrior(
    model_covariance_inv=((corrx, corry), sigma_slowness),
    mean_model=ref_start_slowness.reshape(model_shape)
)

######################################################################
#

fmm_problem_gaussian_prior.set_objective(objective_func, args=[gaussian_prior, sigma])
fmm_problem_gaussian_prior.set_gradient(gradient, args=[gaussian_prior, sigma])
fmm_problem_gaussian_prior.set_hessian(hessian, args=[gaussian_prior, sigma])

######################################################################
#


######################################################################
# 3.2 Start an inversion
# ~~~~~~~~~~~~~~~~~~~~~~
# 

# reuse the previously defined InversionOptions object
inv = cofi.Inversion(fmm_problem_gaussian_prior, my_options)
inv_result_gaussian_prior = inv.run()
inv_result_gaussian_prior.summary()

######################################################################
#


######################################################################
# 3.3 Plotting
# ~~~~~~~~~~~~
# 

fmm.plot_model(inv_result_gaussian_prior.model);            # inverted model
fmm.plot_model(fmm.good_model);       # true model

######################################################################
#


######################################################################
# 4. Comparison and discussion
# ----------------------------
# 
# #TODO
# 

fmm.plot_model(inv_result_quadratic_reg.model);
fmm.plot_model(inv_result_gaussian_prior.model);

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