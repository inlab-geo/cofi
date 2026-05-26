"""
Seismic Travel time Tomography via Fast Marching - Demo on switching regularization and L-curve
===============================================================================================

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
# We will use ``cofi`` to run a seismic tomography example.
# 


######################################################################
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
# For forward modelling, a fast marching wave front tracker is used,
# utilizing the Fast Marching Fortran code within the package
# ```FMTOMO`` <http://iearth.edu.au/codes/FMTOMO/>`__ by Nick Rawlinson.
# The Fast Marching code is wrapped in package
# `pyfm2d <https://github.com/inlab-geo/pyfm2d>`__. Further details can be
# found in:
# 
# -  Rawlinson, N., de Kool, M. and Sambridge, M., 2006. Seismic wavefront
#    tracking in 3-D heterogeneous media: applications with multiple data
#    classes, Explor. Geophys., 37, 322-330.
# -  Rawlinson, N. and Urvoy, M., 2006. Simultaneous inversion of active
#    and passive source datasets for 3-D seismic structure with
#    application to Tasmania, Geophys. Res. Lett., 33 L24313,
#    10.1029/2006GL028105.
# -  de Kool, M., Rawlinson, N. and Sambridge, M. 2006. A practical grid
#    based method for tracking multiple refraction and reflection phases
#    in 3D heterogeneous media, Geophys. J. Int., 167, 253-270.
# -  Saygin, E. 2007. Seismic receiver and noise correlation based studies
#    in Australia, PhD thesis, Australian National University,
#    10.25911/5d7a2d1296f96.
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

# !pip install -U cofi pyfm2d

######################################################################
#

import numpy as np
import matplotlib.pyplot as plt
import pprint

import cofi
# NB You will need to separately install pyfm2d in your python environment with `pip install pyfm2d'
import pyfm2d as wt # import fmm package 

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

# read in problem data
loaded_dict = np.load('../../data/travel_time_tomography/nonlinear_tomo_example.npz')
nonlinear_tomo_example = dict(loaded_dict)
loaded_dict.close()

######################################################################
#

# set up problem
good_model = nonlinear_tomo_example["_mtrue"]
extent = nonlinear_tomo_example["extent"]
sources = nonlinear_tomo_example["sources"]
receivers = nonlinear_tomo_example["receivers"]
obstimes = nonlinear_tomo_example["_data"]
print(' New data set have:\n',len(receivers),' receivers\n',len(sources),' sources\n',len(obstimes),' travel times\n',
'Range of travel times: ',np.min(obstimes),'to',np.max(obstimes),'\n Mean travel time:',np.mean(obstimes))

######################################################################
#

# display true model and raypaths
options = wt.WaveTrackerOptions(paths=True,cartesian=True) # set wavetracker options
result = wt.calc_wavefronts(good_model,receivers,sources,extent=extent, options=options) # track wavefronts
wt.display_model(good_model,paths=result.paths,extent=extent,line=0.3,alpha=0.82)

######################################################################
#


######################################################################
# 1. Problem setup and utilities
# ------------------------------
# 

# get problem information 
model_size = good_model.size                           # number of model parameters
model_shape = good_model.shape                         # 2D spatial grid shape
data_size = data_size = len(obstimes)                  # number of data
ref_start_slowness = nonlinear_tomo_example["_sstart"] # use the starting guess supplied by the nonlinear example

######################################################################
#

def objective_func(slowness, reg, sigma, reduce_data=None):  # reduce_data=(idx_from, idx_to)
    if reduce_data is None: idx_from, idx_to = (0, data_size)
    else: idx_from, idx_to = reduce_data
    if(True):
        options = wt.WaveTrackerOptions(
            cartesian=True,
        )
        result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
        ttimes = result.ttimes
    residual = obstimes[idx_from:idx_to] - ttimes[idx_from:idx_to]
    data_misfit = residual.T @ residual / sigma**2
    model_reg = reg(slowness)
    return  data_misfit + model_reg

def gradient(slowness, reg, sigma, reduce_data=None):       # reduce_data=(idx_from, idx_to)
    if reduce_data is None: idx_from, idx_to = (0, data_size)
    else: idx_from, idx_to = reduce_data
    if(True):
        options = wt.WaveTrackerOptions(
                    paths=True,
                    frechet=True,
                    cartesian=True,
                    )
        result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
        ttimes = result.ttimes
        A = result.frechet.toarray()
    ttimes = ttimes[idx_from:idx_to]
    A = A[idx_from:idx_to]
    data_misfit_grad = -2 * A.T @ (obstimes[idx_from:idx_to] - ttimes) / sigma**2
    model_reg_grad = reg.gradient(slowness)
    return  data_misfit_grad + model_reg_grad

def hessian(slowness, reg, sigma, reduce_data=None):        # reduce_data=(idx_from, idx_to)
    if reduce_data is None: idx_from, idx_to = (0, data_size)
    else: idx_from, idx_to = reduce_data
    if(True):
        options = wt.WaveTrackerOptions(
                    paths=True,
                    frechet=True,
                    cartesian=True,
                    )
        result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options)
        A = result.frechet.toarray()
    A = A[idx_from:idx_to]
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
fmm_problem_quadratic_reg.set_initial_model(ref_start_slowness.flatten())

######################################################################
#

# add regularization: flattening + smoothing
smoothing_factor = 5e6
reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
    model_shape=model_shape,
    weighting_matrix="smoothing"
)
reg = reg_smoothing

######################################################################
#

sigma = 0.000008          # data standard deviation of noise
fmm_problem_quadratic_reg.set_objective(objective_func, args=[reg, sigma, None])
fmm_problem_quadratic_reg.set_gradient(gradient, args=[reg, sigma, None])
fmm_problem_quadratic_reg.set_hessian(hessian, args=[reg, sigma, None])

######################################################################
#


######################################################################
# 2.2 Define InversionOptions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

my_options = cofi.InversionOptions()

my_options.set_tool("cofi.simple_newton")
my_options.set_params(
    num_iterations=15, 
    step_length=1, 
    obj_tol=1e-16,
    verbose=True, 
    hessian_is_symmetric=True
)

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

vmodel_inverted = 1./inv_result_quadratic_reg.model.reshape(model_shape)
wt.display_model(vmodel_inverted,extent=extent) # inverted model
wt.display_model(good_model,extent=extent) # true model

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
fmm_problem_gaussian_prior.set_initial_model(ref_start_slowness.flatten())

######################################################################
#

# add regularization: Gaussian prior
corrx = 3.0
corry = 3.0
sigma_slowness = 0.5**2
sigma_slowness = 2.5E-6
gaussian_prior = 0.01 * cofi.utils.GaussianPrior(
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

vmodel_inverted = 1./inv_result_gaussian_prior.model.reshape(model_shape)
wt.display_model(vmodel_inverted,extent=extent) # inverted model
wt.display_model(good_model,extent=extent) # true model

######################################################################
#


######################################################################
# 4. L-curve
# ----------
# 
# Now we plot an L-curve for the smoothing regularization case.
# 

lambdas = np.logspace(-4, 4, 10)

my_lcurve_problems = []
for lamb in lambdas:
    my_reg = lamb * reg_smoothing
    my_problem = cofi.BaseProblem()
    my_problem.set_objective(objective_func, args=[my_reg, sigma])
    my_problem.set_gradient(gradient, args=[my_reg, sigma])
    my_problem.set_hessian(hessian, args=[my_reg, sigma])
    my_problem.set_initial_model(ref_start_slowness.flatten())
    my_lcurve_problems.append(my_problem)

my_options.set_params(verbose=False)

def my_callback(inv_result, i):
    m = inv_result.model
    slowness=m
    options = wt.WaveTrackerOptions(
            cartesian=True,
            )
    result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
    ttimes = result.ttimes
    res_norm = np.linalg.norm(ttimes - obstimes)/sigma**2
    reg_norm = np.sqrt(reg_smoothing(m))
    print(f"Finished inversion with lambda={lambdas[i]}: {res_norm}, {reg_norm}")
    return res_norm, reg_norm

my_inversion_pool = cofi.utils.InversionPool(
    my_lcurve_problems, 
    my_options, 
    my_callback, 
    False
)
all_res, all_cb_returns = my_inversion_pool.run()

l_curve_points = list(zip(*all_cb_returns))

######################################################################
#

# plot the L-curve
res_norm, reg_norm = l_curve_points
plt.plot(reg_norm, res_norm, '.-')
plt.xlabel(r'Norm of regularization term $||Wm||_2$')
plt.ylabel(r'Norm of residual $||g(m)-d||_2$')
for i in range(len(lambdas)):
    plt.annotate(f'{lambdas[i]:.1e}', (reg_norm[i], res_norm[i]), fontsize=8)

# plot the previously solved model
my_inverted_model = inv_result_quadratic_reg.model
my_reg_norm = np.sqrt(reg_smoothing(my_inverted_model))
slowness=my_inverted_model
options = wt.WaveTrackerOptions(cartesian=True)
result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
ttimes = result.ttimes
my_residual_norm = np.linalg.norm(ttimes - obstimes)/sigma**2
plt.plot(my_reg_norm, my_residual_norm, "x")
plt.annotate(f"{smoothing_factor:.1e}", (my_reg_norm, my_residual_norm), fontsize=8);

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

watermark_list = ["cofi", "numpy", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#



######################################################################
#



######################################################################
#
# sphinx_gallery_thumbnail_number = -1