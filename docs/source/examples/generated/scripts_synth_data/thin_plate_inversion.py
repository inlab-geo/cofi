"""
Thin plate inversion
====================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/airborne_em/airborne_em_single_transmitter.ipynb
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
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/examples/vtem_max

######################################################################
#

# If this notebook is run locally PyP223 needs to be installed separately by uncommenting the following line, 
# that is by removing the # and the white space between it and the exclamation mark.
# !pip install git+https://github.com/JuergHauser/PyP223.git

######################################################################
#

import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz
import cofi

from vtem_max_forward_lib import (
    problem_setup, 
    system_spec,
    survey_setup, 
    true_model, 
    ForwardWrapper, 
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
# When modelling the electromagnetic response of subvertical bodies such
# as a VMS deposit they can be approximated using a thin plate in the
# halfspace of a layered earth, that is the forward solver computes the 3D
# response of a thin plate (e.g. Prikhodko et al. 2019). Here we develop a
# thin plate inversion method using CoFI to solve the inverse problem and
# P223 (Raiche et. al., 2007) to solve the forward problem.
# 
# Model parametrisation
# ~~~~~~~~~~~~~~~~~~~~~
# 
# In the following we look at a thin plate with a conductance of
# :math:`2 \mathrm{S}` located in a halfspace with a resistivity of
# :math:`1000 \mathrm{\Omega m}` with a :math:`20 \mathrm{m}` thick
# regolith that has a resistivity of :math:`300 \mathrm{\Omega m}`.
# 
# .. figure::
#    https://raw.githubusercontent.com/inlab-geo/cofi-examples/main/examples/vtem_max/figures/wpar8wl.png
#    :alt: wpar8wl.png
# 
#    wpar8wl.png
# 
# The problem setup is imported from ``vtem_max_forward_lib.py`` but can
# be adjusted for other applications. The wrapper is created so that we
# can declare model parameters which are a subset of all the model
# parameters required by the forward solver. This allows to, for example,
# invert only for dip of the thin plate with all the other mode paremters
# assumed to be known. The thin plate is parameterised using the
# parametrisation introduced in (Hauser et. al. 2016). Compared to the
# commonly employed parametrisation with a plate reference point on the
# edge of the plate this parametrisation allows for a thin plate to grow
# and shrink around a plate refrerence point, without the need to move the
# reference point. This can be advantageous when there is for example a
# borehole intersecting a thin plate.
# 
# .. figure::
#    https://raw.githubusercontent.com/inlab-geo/cofi-examples/main/examples/vtem_max/figures/wpar7wl.png
#    :alt: wpar7wl.png
# 
#    wpar7wl.png
# 
# Forward solver
# ~~~~~~~~~~~~~~
# 
# The forward solver is LeroiAir (Raiche et. al, 2007) and the code has
# been reorganised so that the response measured by an AEM system is give
# by a function that can be called from Python. In LeroiAir plates are
# discretised into cells, with the accuracy of the forward solver a
# function of the chosen cell-size. The forward solver is kept in a
# seperate Python package that is available
# `here <https://github.com/JuergHauser/PyP223.git>`__
# 
# Jacobian via finite differencing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Parameter estimation methods frequently rely on the provision of a
# Jacobian for efficient optimisaiton. If an analytical Jacobian is not
# available it can be computed via finite differencing.
# 
# $ f’(x_0) = {f(x_0 +h)-f(x_0):raw-latex:`\over `h} $
# 
# Care must be taken when choosing the step size :math:`h` as a too small
# step size my result in a Jacobian that is affected by a limited accuracy
# of the forward solver and a too large step size :math:`h` might result
# in a Jacobian that is not representative of the derivatives at location
# :math:`x_0`. In the following we use a relative step size :math:`q` that
# is :math:`h=x0*(1.0+q)`. Further to this the gradient of the objective
# functions itself is affected by the noise on the data, thus for noisy
# data choosing a larger step size when computing the Jacobian can be
# advisable.
# 
# VTEM max Data
# ^^^^^^^^^^^^^
# 
# Airborne electromagnetic systems can be categorised into either
# helicopter or fixed wing systems. The examples in this directory use a
# VTEM max system which is a helicopter based system developed and
# operated by Geotech.
# 
# https://geotech.ca/services/electromagnetic/vtem-versatile-time-domain-electromagnetic-system/
# 
# Successful inversion also relies on the objective function being smooth
# and predictable. For the data being inverted here it is advantageous to
# convert measurements to scale logarithmically to obtain a smoother and
# more predictable objective function when compared with using the
# unscaled data. Similarly plate orientation angels are converted into
# radians.
# 
# Further reading
# '''''''''''''''
# 
# Hauser, J., Gunning, J., & Annetts, D. (2016). Probabilistic inversion
# of airborne electromagnetic data for basement conductors. Geophysics,
# 81(5), E389-E400.
# 
# Prikhodko, A., Morrison, E., Bagrianski, A., Kuzmin, P., Tishin, P., &
# Legault, J. (2010). Evolution of VTEM? technical solutions for effective
# exploration. ASEG Extended Abstracts, 2010(1), 1-4.
# 
# Raiche, A., Sugeng, F. and Wilson, G. (2007) Practical 3D EM inversion
# the P223F software suite, ASEG Extended Abstracts, 2007:1, 1-5
# 
# Wheelock, B., Constable, S., & Key, K. (2015). The advantages of
# logarithmically scaled data for electromagnetic inversion. Geophysical
# Journal International, 201(3), 1765–1780.
# https://doi.org/10.1093/GJI/GGV107
# 


######################################################################
# Problem definition
# ~~~~~~~~~~~~~~~~~~
# 

survey_setup = {
    "tx": numpy.array([205.]),                  # transmitter easting/x-position
    "ty": numpy.array([100.]),                  # transmitter northing/y-position
    "tz": numpy.array([50.]),                   # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90.])),  # transmitter azimuth
    "tincl": numpy.deg2rad(numpy.array([6.])),  # transmitter inclination
    "rx": numpy.array([205.]),                  # receiver easting/x-position
    "ry": numpy.array([100.]),                  # receiver northing/y-position
    "rz": numpy.array([50.]),                   # receiver height/z-position
    "trdx": numpy.array([0.]),                  # transmitter receiver separation inline
    "trdy": numpy.array([0.]),                  # transmitter receiver separation crossline
    "trdz": numpy.array([0.]),                  # transmitter receiver separation vertical
}

######################################################################
#

forward = ForwardWrapper(true_model, problem_setup, system_spec, survey_setup, ["pdip"])
true_param_value = numpy.array([60])

######################################################################
#


######################################################################
# True model
# ^^^^^^^^^^
# 

_, axes = plt.subplots(2, 2)
axes[1,1].axis("off")
plot_plate_faces(
    "plate_true", forward, true_param_value, 
    axes[0,0], axes[0,1], axes[1,0], color="purple", label="True model"
)
plt.tight_layout()
point = Line2D([0], [0], label='Fiducial', marker='o', markersize=5, 
         markeredgecolor='orange', markerfacecolor='orange', linestyle='')

handles, labels = axes[1,0].get_legend_handles_labels()
handles.extend([point])

axes[1,0].legend(handles=handles,bbox_to_anchor=(1.04, 0), loc="lower left")



######################################################################
#


######################################################################
# Generate synthetic data
# ^^^^^^^^^^^^^^^^^^^^^^^
# 
# We use a simplfied noise model that assumes an absolute noise that is a
# standard deviation of :math:`0.05` for the logarithms of the measured
# and observed data.
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
# Starting model
# ^^^^^^^^^^^^^^
# 
# Set an initial guess for the dip of the thin plate
# 

init_param_value = numpy.array([45])

######################################################################
#


######################################################################
# Define helper functions for CoFI
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

def my_objective(model):
    dpred = forward(model)
    residual = dpred - data_obs
    return residual.T @ Cdinv @ residual

def my_gradient(model):
    dpred = forward(model)
    jacobian = forward.jacobian(model, relative_step=0.1)
    residual = dpred - data_obs
    return jacobian.T @ Cdinv @ residual

def my_hessian(model):
    jacobian = forward.jacobian(model)
    return jacobian.T @ Cdinv @ jacobian



class PerIterationCallbackFunction:
    def __init__(self):
        self.x = None
        self.i = 0

    def __call__(self, xk):
        print(f"Iteration #{self.i+1}")
        print(f"  objective value: {my_problem.objective(xk)}")
        self.x = xk
        self.i += 1

######################################################################
#


######################################################################
# Sensitivity of the misfit function and usefulness of the gradient
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# When setting up a new inverse problem prior to any inversion it makes
# sense to verify that the misfit function is sensitve to the parameter of
# itnerest and that the gradient of the objective function points in the
# right direction.
# 

all_models = [numpy.array([pdip]) for pdip in range(40, 120, 5)]
all_misfits = []
all_gradients = []
for model in all_models:
    misfit = my_objective(model)
    gradient = my_gradient(model)
    all_misfits.append(misfit)
    all_gradients.append(gradient)
    print(f"pdip: {model}, data misfit: {misfit}, gradient: {gradient}")


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(all_models, all_misfits,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlabel("pdip")
ax1.set_ylabel("Data misfit",color=color)

ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.plot(all_models, all_gradients,color=color)
ax2.set_ylabel('Gradient', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

######################################################################
#


######################################################################
# Parameter estimation
# --------------------
# 
# First we solve the inverse problem using optimisation that is we seek to
# find the minimum of the objective function given as
# 
# .. math::
# 
# 
#    \chi^2 = (\mathbf{d} - \mathbf{f}(\mathbf{m}))^T\mathbf{C}_d^{-1}(\mathbf{d}-\mathbf{f}(\mathbf{m})),
# 
# with the full Newton step being
# 
# .. math::
# 
# 
#    \begin{equation} \Delta \mathbf{m}= (\underbrace{\mathbf{J}^T \mathbf{C}_d^{-1} \mathbf{J}}_{\mathbf{Hessian}})^{-1}
#    (\underbrace{ \mathbf{J}^T\mathbf{C}_d^{-1} 
#    (\mathbf{y}-\mathbf{f}(\mathbf{m}))}_\mathbf{Gradient}).
#    \end{equation} 
# 


######################################################################
# Define CoFI problem
# ^^^^^^^^^^^^^^^^^^^
# 

my_problem = cofi.BaseProblem()
my_problem.set_objective(my_objective)
my_problem.set_gradient(my_gradient)
my_problem.set_hessian(my_hessian)
my_problem.set_initial_model(init_param_value)

######################################################################
#


######################################################################
# Define CoFI options
# ^^^^^^^^^^^^^^^^^^^
# 

my_options = cofi.InversionOptions()
my_options.set_tool("scipy.optimize.minimize")
my_options.set_params(method="Newton-CG",callback=PerIterationCallbackFunction())

######################################################################
#


######################################################################
# CoFI inversion
# ^^^^^^^^^^^^^^
# 

my_inversion = cofi.Inversion(my_problem, my_options)
my_result = my_inversion.run()
print(my_result.model)

######################################################################
#


######################################################################
# Plotting
# ^^^^^^^^
# 


######################################################################
# Data
# ''''
# 

_, (ax1, ax2) = plt.subplots(1, 2)
plot_transient(true_param_value, forward, "Data from true model", ax1, ax2, color="purple")
plot_transient(init_param_value, forward, "Data from starting model", ax1, ax2, color="green", linestyle=":")
plot_transient(my_result.model, forward, "Data from MAP model", ax1, ax2, color="red", linestyle="-.")
ax1.legend(loc="upper center")
ax2.legend(loc="upper center")
ax1.set_title("vertical")
ax2.set_title("inline")
plt.tight_layout()

######################################################################
#


######################################################################
# Model
# '''''
# 

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
plot_plate_faces(
    "plate_inverted", forward, my_result.model, 
    axes[0,0], axes[0,1], axes[1,0], color="red", label="MAP solution", linestyle="dotted"
)
plt.tight_layout()
point = Line2D([0], [0], label='Fiducial', marker='o', markersize=5, 
         markeredgecolor='orange', markerfacecolor='orange', linestyle='')

handles, labels = axes[1,0].get_legend_handles_labels()
handles.extend([point])

axes[1,0].legend(handles=handles,bbox_to_anchor=(1.04, 0), loc="lower left")

######################################################################
#


######################################################################
# Ensemble method
# ---------------
# 
# Parameter estimation methods require an objective function while
# ensemble methods require a likelihood functions typically given in the
# form of a log likelidhood function. The objective function used for the
# parameter estimation consists of only a data misfit term and thus is
# closely related to the likelihood function, with the log likelihood
# being proportional to the value of the objective function multiplied by
# a factor of :math:`\frac{1}{2}`
# 
# .. math::
# 
# 
#    p({\mathbf d} | {\mathbf m}) \propto \exp \left\{- \frac{1}{2} ({\mathbf d}-{\mathbf f}({\mathbf m}))^T C_d^{-1} ({\mathbf d}-{\mathbf f}({\mathbf m})) \right\}
# 
# In the following we define a log likelihood function and log prior
# function. The prior distribution is a uniform distribution with an lower
# boundary of :math:`10 \degree` and an upper boundary of
# :math:`80 \degree`
# 

def my_log_likelihood(model):
    return -0.5 * my_objective(model)

def my_log_prior(m):    # uniform distribution
    for i in range(len(m)):
        if m[i] < m_min[i] or m[i] > m_max[i]: return -numpy.inf
    return 0.0 # model lies within bounds -> return log(1)

######################################################################
#

m_min=numpy.array([10])
m_max=numpy.array([80])

######################################################################
#


######################################################################
# Augment the CoFI problem
# ^^^^^^^^^^^^^^^^^^^^^^^^
# 
# To be able to use an ensemble method we need to augment our CoFI problem
# with a function providing the log of the prior probability and a second
# function that provides the log of the likelihood function.
# 

my_problem.set_log_prior(my_log_prior)
my_problem.set_log_likelihood(my_log_likelihood)
my_problem.set_model_shape(len(init_param_value))

######################################################################
#


######################################################################
# Define CoFI options
# ^^^^^^^^^^^^^^^^^^^
# 

nwalkers = 3
ndim = len(init_param_value)
nsteps = 500
walkers_start = init_param_value + 1 * numpy.random.randn(nwalkers, ndim)

######################################################################
#


######################################################################
# CoFI Inversion
# ^^^^^^^^^^^^^^
# 

inv_options = cofi.InversionOptions()
inv_options.set_tool("emcee")
inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps, initial_state=walkers_start, progress=True)

######## Run it
inv = cofi.Inversion(my_problem, inv_options)
inv_result = inv.run()

######## Check result
print(f"The inversion result from `emcee`:")
inv_result.summary()

######################################################################
#

sampler = inv_result.sampler
arviz.style.use("arviz-variat")
var_names = [
    "plate dip (°)", 
]
az_idata = inv_result.to_arviz(var_names=var_names)
true_values = [60]
pc = arviz.plot_trace_dist(
    az_idata.sel(draw=slice(100, None)),
    visuals={"xlabel_trace": False, "trace": {"color": "C0"}, "dist": {"color": "C0"}},
    figure_kwargs={"figsize": (12, 4), "constrained_layout": True},
)
var_list = list(az_idata.posterior.data_vars)
for i, vname in enumerate(var_list):
    ax_kde = pc.iget_target(i, 0)
    ax_trace = pc.iget_target(i, 1)
    ax_kde.set_title(vname)
    ax_trace.set_title(vname)
    ax_kde.axvline(true_values[i], color="green", linestyle="--", lw=1, alpha=0.5)
    ax_trace.axhline(true_values[i], color="green", linestyle="--", lw=1, alpha=0.5)
    ax_trace.margins(x=0)


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

watermark_list = ["cofi", "numpy", "scipy", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1