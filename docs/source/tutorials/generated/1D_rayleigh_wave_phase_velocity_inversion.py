"""
1D Rayleigh wave phase velocity inversion
=========================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/tutorials/rayleigh_wave_phase_velocity/1D_rayleigh_wave_phase_velocity_inversion.ipynb
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
# --------------
# 
# What we do in this notebook
# ---------------------------
# 
# Here we look at applying CoFI to the inversion of Rayleigh wave phase
# velocities for a 1D layered earth.
# 
# **Learning outcomes**
# 
# - A demonstration of CoFI’s ability to switch between parameter
#   estimation and ensemble methods.
# - A comparison between different McMC samplers that is fixed-d and
#   trans-d samplers
# - An application of CoFI to field data
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi git+https://github.com/inlab-geo/pysurf96.git
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/tutorials/rayleigh_wave_phase_velocity

######################################################################
#

# If this notebook is run locally pysurf96 needs to be installed separately by uncommenting the following line, 
# that is by removing the # and the white space between it and the exclamation mark.
# !pip install -U cofi git+https://github.com/inlab-geo/pysurf96.git

######################################################################
#

import numpy as np
import scipy
import matplotlib.pyplot as plt

from pysurf96 import surf96
import bayesbay
import cofi


######################################################################
#


######################################################################
# Problem description
# -------------------
# 
# Here we illustrate the range of inversion methods made avaialbe by CoFI.
# That is we first define a base problem and then explore the use of an
# iterative non linear apporach to find the MAP solution and then employ a
# range of Markov Chain Monte Carlo strategies to recover the posterior
# distribution. The forward problem is solved using pysurf 96
# (https://github.com/miili/pysurf96) and the field data example is taken
# from (https://www.eas.slu.edu/eqc/eqc_cps/TUTORIAL/STRUCT/index.html)
# and we will be inverting observed rayleigh wave phase velocities
# 


######################################################################
# **Inference problem**
# 

# display theory on the inference problem
from IPython.display import display, Markdown

with open("../../theory/geo_surface_wave_dispersion.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#


######################################################################
# **Solving methods**
# 

# display theory on the optimisation approach
with open("../../theory/inv_optimisation.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#

# display theory on the optimisation approach
with open("../../theory/inv_mcmc.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#


######################################################################
# **Further reading**
# 
# https://en.wikipedia.org/wiki/Surface_wave_inversion
# 


######################################################################
# Utilities
# ---------
# 


######################################################################
# 1D model paramterisation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 

# display theory on the 1D model parameterisation
with open("../../theory/misc_1d_model_parameterisation.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#

# layercake model utilities
def form_layercake_model(thicknesses, vs):
    model = np.zeros((len(vs)*2-1,))
    model[1::2] = thicknesses
    model[::2] = vs
    return model

def split_layercake_model(model):
    thicknesses = model[1::2]
    vs = model[::2]
    return thicknesses, vs

######################################################################
#

# voronoi model utilities
def form_voronoi_model(voronoi_sites, vs):
    return np.hstack((vs, voronoi_sites))

def split_voronoi_model(model):
    voronoi_sites = model[len(model)//2:]
    vs = model[:len(model)//2]
    return voronoi_sites, vs

######################################################################
#

def voronoi_to_layercake(voronoi_vector: np.ndarray) -> np.ndarray:
    n_layers = len(voronoi_vector) // 2
    velocities = voronoi_vector[:n_layers]
    voronoi_sites = voronoi_vector[n_layers:]
    depths = (voronoi_sites[:-1] + voronoi_sites[1:]) / 2
    thicknesses = depths - np.insert(depths[:-1], 0, 0)
    layercake_vector = np.zeros((2*n_layers-1,))
    layercake_vector[::2] = velocities
    layercake_vector[1::2] = thicknesses
    return layercake_vector

def layercake_to_voronoi(layercake_vector: np.ndarray, first_voronoi_site: float = 0.0) -> np.ndarray:
    n_layers = len(layercake_vector) // 2 + 1
    thicknesses = layercake_vector[1::2]
    velocities = layercake_vector[::2]
    depths = np.cumsum(thicknesses)
    voronoi_sites = np.zeros((n_layers,))
    for i in range(1,len(voronoi_sites)):
        voronoi_sites[i] = 2 * depths[i-1] - voronoi_sites[i-1]
    voronoi_vector = np.hstack((velocities, voronoi_sites))
    return voronoi_vector

######################################################################
#


######################################################################
# Forward solver
# ~~~~~~~~~~~~~~
# 

# display theory on the using the forward solver
with open("../../theory/geo_surface_wave_dispersion2.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#

# Constants
VP_VS = 1.77
RHO_VP_K = 0.32
RHO_VP_B = 0.77

######################################################################
#

# forward through pysurf96
def forward_sw(model, periods):
    thicknesses, vs = split_layercake_model(model)
    thicknesses = np.append(thicknesses, 10)
    vp = vs * VP_VS
    rho = RHO_VP_K * vp + RHO_VP_B
    return surf96(
        thicknesses,
        vp,
        vs,
        rho,
        periods,
        wave="rayleigh",
        mode=1,
        velocity="phase",
        flat_earth=False,
    )

# numerical jacobian
def jacobian_sw(model, periods, fwd=forward_sw, relative_step=0.01):
    jacobian = np.zeros((len(periods), len(model)))
    original_dpred = fwd(model, periods)
    for i in range(len(model)):
        perturbed_model = model.copy()
        step = relative_step * model[i]
        perturbed_model[i] += step
        perturbed_dpred = fwd(perturbed_model, periods)
        derivative = (perturbed_dpred - original_dpred) / abs(step)
        jacobian[:, i] = derivative
    return jacobian

######################################################################
#


######################################################################
# Visualisation
# -------------
# 
# For conveninece we also implement two functions to plot the data here
# the Rayleigh wave phase velocity and a model given in the layer based
# parametrisation.
# 

def plot_model(model, ax=None, title="model", **kwargs):
    # process data
    thicknesses = np.append(model[1::2], max(model[1::2]))
    velocities = model[::2]
    y = np.insert(np.cumsum(thicknesses), 0, 0)
    x = np.insert(velocities, 0, velocities[0])
    
    # plot depth profile
    if ax is None:
        _, ax = plt.subplots()
    plotting_style = {
        "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 0.5)),
        "alpha": 0.2,
        "color": kwargs.pop("color", kwargs.pop("c", "blue")),
    }
    plotting_style.update(kwargs)
    ax.step(x, y, where="post", **plotting_style)
    if ax.get_ylim()[0] < ax.get_ylim()[1]:
        ax.invert_yaxis()
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title)
    return ax

######################################################################
#

def plot_data(rayleigh_phase_velocities, periods, ax=None, scatter=False, 
              title="data", **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    plotting_style = {
        "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 1)),
        "alpha": 1,
        "color": kwargs.pop("color", kwargs.pop("c", "blue")),
    }
    plotting_style.update(**kwargs)
    if scatter:
        ax.scatter(periods, rayleigh_phase_velocities, **plotting_style)
    else:
        ax.plot(periods, rayleigh_phase_velocities, **plotting_style)
    ax.set_xlabel("Periods (s)")
    ax.set_ylabel("Rayleigh phase velocities (km/s)")
    ax.set_title(title)
    return ax

######################################################################
#

def plot_model_and_data(model, label_model, color_model, 
                        forward_func, periods, label_d_pred, color_d_pred, 
                        axes=None, light_thin=False):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2.5]})
    ax1, ax2 = axes
    if light_thin:
        plot_model(model, ax=ax1, color=color_model, alpha=0.2, lw=0.5, label=label_model)
        plot_data(forward_func(model, periods), periods, ax=ax2, color=color_d_pred, alpha=0.2, lw=0.5, label=label_d_pred)
    else:
        plot_model(model, ax=ax1, color=color_model, alpha=1, lw=1, label=label_model)
        plot_data(forward_func(model, periods), periods, ax=ax2, color=color_d_pred, label=label_d_pred)
    ax1.legend()
    ax2.legend()
    return ax1, ax2

######################################################################
#


######################################################################
# Synthetic example
# -----------------
# 
# Prior to inverting any field data it is good practice to test an
# inversion method using sythetic exmaples where we know the true model.
# It is also recommended to prior to this idnepently test any forward
# solver that is being used and verify the Jacobian, as problems related
# to the forward sovler are diffiuclt to identify and diagnose once they
# are integrated in an inversion methodology.
# 


######################################################################
# Generate synthetic data
# ~~~~~~~~~~~~~~~~~~~~~~~
# 

synth_d_periods = np.geomspace(3, 80, 20)

true_thicknesses = np.array([10, 10, 15, 20, 20, 20, 20, 20])
true_vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
true_model = form_layercake_model(true_thicknesses, true_vs)

######################################################################
#

noise_level = 0.02
d_true = forward_sw(true_model, synth_d_periods)
d_obs = d_true + np.random.normal(0, 0.01, len(d_true))

######################################################################
#

# plot true model and d_pred from true model
_, ax2 = plot_model_and_data(model=true_model, label_model="true model", color_model="orange",
                    forward_func=forward_sw, periods=synth_d_periods, 
                    label_d_pred="true data (noiseless)", color_d_pred="orange")

# plot d_obs
plot_data(d_obs, synth_d_periods, ax=ax2, scatter=True, color="red", s=20, label="observed data (noisy)")
ax2.legend();

######################################################################
#


######################################################################
# Optimisation
# ~~~~~~~~~~~~
# 


######################################################################
# **Prepare ``BaseProblem`` for optimisation**
# 

n_dims = 17

init_thicknesses = np.ones((n_dims//2,)) * 15
init_vs = np.ones((n_dims//2+1,)) * 4.0
init_model = form_layercake_model(init_thicknesses, init_vs)

######################################################################
#

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="starting model", color_model="purple",
                           forward_func=forward_sw, periods=synth_d_periods, 
                           label_d_pred="data predictions from starting model", color_d_pred="purple")

# plot the model and d_pred for true model
plot_model_and_data(model=true_model, label_model="true model", color_model="orange",
                    forward_func=forward_sw, periods=synth_d_periods, 
                    label_d_pred="true data (noiseless)", color_d_pred="orange", axes=axes)

# plot d_obs
plot_data(d_obs, synth_d_periods, ax=axes[1], scatter=True, color="red", s=20, label="d_obs")
axes[1].legend();

######################################################################
#

my_reg = cofi.utils.QuadraticReg(
    weighting_matrix="damping", 
    model_shape=(n_dims,), 
    reference_model=init_model
)

######################################################################
#

def my_objective(model, fwd, periods, d_obs, lamda=1.0):
    d_pred = fwd(model, periods)
    data_misfit = np.sum((d_obs - d_pred) ** 2)
    reg = my_reg(model)
    return data_misfit + lamda * reg

def my_objective_gradient(model, fwd, periods, d_obs, lamda=1.0):
    d_pred = fwd(model, periods)
    jac = jacobian_sw(model, periods, fwd)
    data_misfit_grad = -2 * jac.T @ (d_obs - d_pred)
    reg_grad = my_reg.gradient(model)
    return data_misfit_grad + lamda * reg_grad

def my_objective_hessian(model, fwd, periods, d_obs, lamda=1.0):
    jac = jacobian_sw(model, periods, fwd)
    data_misfit_hess = 2 * jac.T @ jac
    reg_hess = my_reg.hessian(model)
    return data_misfit_hess + lamda * reg_hess

######################################################################
#


######################################################################
# Optimisation with no damping
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

lamda = 0

kwargs = {
    "fwd": forward_sw, 
    "periods": synth_d_periods, 
    "d_obs": d_obs, 
    "lamda": lamda
}
sw_problem_no_reg = cofi.BaseProblem()
sw_problem_no_reg.set_objective(my_objective, kwargs=kwargs)
sw_problem_no_reg.set_gradient(my_objective_gradient, kwargs=kwargs)
sw_problem_no_reg.set_hessian(my_objective_hessian, kwargs=kwargs)
sw_problem_no_reg.set_initial_model(init_model)

######################################################################
#


######################################################################
# **Define ``InversionOptions``**
# 

inv_options_optimiser = cofi.InversionOptions()
inv_options_optimiser.set_tool("scipy.optimize.minimize")
inv_options_optimiser.set_params(method="trust-exact")

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inv_optimiser_no_reg = cofi.Inversion(sw_problem_no_reg, inv_options_optimiser)
inv_result_optimiser_no_reg = inv_optimiser_no_reg.run()

######################################################################
#


######################################################################
# **Plot results**
# 

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="starting model", color_model="black",
                           forward_func=forward_sw, periods=synth_d_periods, 
                           label_d_pred="d_pred from starting model", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=true_model, label_model="true model", color_model="orange",
                    forward_func=forward_sw, periods=synth_d_periods, 
                    label_d_pred="d_pred from true model", color_d_pred="orange", axes=axes)

# plot the model and d_pred for inverted model
plot_model_and_data(model=inv_result_optimiser_no_reg.model, label_model="inverted model", color_model="purple",
                    forward_func=forward_sw, periods=synth_d_periods,
                    label_d_pred="d_pred from inverted model", color_d_pred="purple", axes=axes);

# plot d_obs
plot_data(d_obs, synth_d_periods, ax=axes[1], scatter=True, color="red", s=20, label="d_obs")
axes[1].legend();

######################################################################
#


######################################################################
# Optimal damping
# ^^^^^^^^^^^^^^^
# 
# Obviously we get a very skewed 1D model out of an optimisation that
# solely tries to minimise the data misfit. We would like to add a damping
# term to our objective function, but we are not sure which factor suits
# the problem well.
# 
# In this situation, the ``InversionPool`` from CoFI can be handy.
# 

lambdas = np.logspace(-6, 6, 15)

my_lcurve_problems = []
for lamb in lambdas:
    my_problem = cofi.BaseProblem()
    kwargs = {
        "fwd": forward_sw, 
        "periods": synth_d_periods, 
        "d_obs": d_obs, 
        "lamda": lamb
    }
    my_problem.set_objective(my_objective, kwargs=kwargs)
    my_problem.set_gradient(my_objective_gradient, kwargs=kwargs)
    my_problem.set_hessian(my_objective_hessian, kwargs=kwargs)
    my_problem.set_initial_model(init_model)
    my_lcurve_problems.append(my_problem)

def my_callback(inv_result, i):
    m = inv_result.model
    res_norm = np.linalg.norm(forward_sw(m, synth_d_periods) - d_obs)
    reg_norm = np.sqrt(my_reg(m))
    print(f"Finished inversion with lambda={lambdas[i]}: {res_norm}, {reg_norm}")
    return res_norm, reg_norm

my_inversion_pool = cofi.utils.InversionPool(
    list_of_inv_problems=my_lcurve_problems,
    list_of_inv_options=inv_options_optimiser,
    callback=my_callback,
    parallel=False
)
all_res, all_cb_returns = my_inversion_pool.run()

l_curve_points = list(zip(*all_cb_returns))

######################################################################
#

# print all the lambdas
lambdas

######################################################################
#


######################################################################
# **Plot L-curve**
# 

# plot the L-curve
res_norm, reg_norm = l_curve_points
plt.plot(reg_norm, res_norm, '.-')
plt.xlabel(r'Norm of regularization term $||Wm||_2$')
plt.ylabel(r'Norm of residual $||g(m)-d||_2$')
for i in range(0, len(lambdas), 2):
    plt.annotate(f'{lambdas[i]:.1e}', (reg_norm[i], res_norm[i]), fontsize=8)

######################################################################
#


######################################################################
# Optimisation with damping
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# From the L-curve plot above, it seems that a damping factor of around
# 0.02 would be good.
# 

lamda = 0.02

kwargs = {
    "fwd": forward_sw, 
    "periods": synth_d_periods, 
    "d_obs": d_obs, 
    "lamda": lamda
}
sw_problem = cofi.BaseProblem()
sw_problem.set_objective(my_objective, kwargs=kwargs)
sw_problem.set_gradient(my_objective_gradient, kwargs=kwargs)
sw_problem.set_hessian(my_objective_hessian, kwargs=kwargs)
sw_problem.set_initial_model(init_model)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inv_optimiser = cofi.Inversion(sw_problem, inv_options_optimiser)
inv_result_optimiser = inv_optimiser.run()

######################################################################
#


######################################################################
# **Plot results**
# 

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="starting model", color_model="black",
                           forward_func=forward_sw, periods=synth_d_periods, 
                           label_d_pred="d_pred from starting model", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=true_model, label_model="true model", color_model="orange",
                    forward_func=forward_sw, periods=synth_d_periods, 
                    label_d_pred="d_pred from true model", color_d_pred="orange", axes=axes)

# plot the model and d_pred for damped solution, and d_obs
plot_model_and_data(model=inv_result_optimiser.model, label_model="damped solution", color_model="purple",
                    forward_func=forward_sw, periods=synth_d_periods,
                    label_d_pred="d_pred from damped solution", color_d_pred="purple", axes=axes);

# plot d_obs
plot_data(d_obs, synth_d_periods, ax=axes[1], scatter=True, color="red", s=20, label="d_obs")
axes[1].legend();

######################################################################
#


######################################################################
# Fixed-dimensional sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# **Prepare ``BaseProblem`` for fixed-dimensional sampling**
# 

thick_min = 5
thick_max = 30
vs_min = 2
vs_max = 5

def my_log_prior(model):
    thicknesses, vs = split_layercake_model(model)
    thicknesses_out_of_bounds = (thicknesses < thick_min) | (thicknesses > thick_max)
    vs_out_of_bounds = (vs < vs_min) | (vs > vs_max)
    if np.any(thicknesses_out_of_bounds) or np.any(vs_out_of_bounds):
        return float("-inf")
    log_prior = -np.log(thick_max - thick_min) * len(thicknesses) - np.log(vs_max - vs_min) * len(vs)
    return log_prior

######################################################################
#

Cdinv = np.eye(len(d_obs))/(noise_level**2)      # inverse data covariance matrix

def my_log_likelihood(model):
    try:
        d_pred = forward_sw(model, synth_d_periods)
    except:
        return float("-inf")
    residual = d_obs - d_pred
    return -0.5 * residual @ (Cdinv @ residual).T

######################################################################
#

n_walkers = 40

my_walkers_start = np.ones((n_walkers, n_dims)) * inv_result_optimiser.model
for i in range(n_walkers):
    my_walkers_start[i,:] += np.random.normal(0, 0.3, n_dims)

######################################################################
#

sw_problem.set_log_prior(my_log_prior)
sw_problem.set_log_likelihood(my_log_likelihood)

######################################################################
#


######################################################################
# **Define ``InversionOptions``**
# 

inv_options_fixed_d_sampling = cofi.InversionOptions()
inv_options_fixed_d_sampling.set_tool("emcee")
inv_options_fixed_d_sampling.set_params(
    nwalkers=n_walkers, 
    nsteps=2_000, 
    initial_state=my_walkers_start, 
    skip_initial_state_check=True, 
    progress=True
)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 


######################################################################
# We will disable the display of warnings temporarily due to the
# unavoidable existence of ``-inf`` in our prior.
# 
# https://github.com/dfm/emcee/issues/370#issuecomment-1013623444
# 

np.seterr(all="ignore");

######################################################################
#


######################################################################
# Sample the prior
# ^^^^^^^^^^^^^^^^
# 

prior_sampling_problem = cofi.BaseProblem()
prior_sampling_problem.set_log_posterior(my_log_prior)
prior_sampling_problem.set_model_shape(init_model.shape)
prior_sampler = cofi.Inversion(prior_sampling_problem, inv_options_fixed_d_sampling)
prior_results = prior_sampler.run()

######################################################################
#

import arviz as az

labels = ["v0", "t0", "v1", "t1", "v2", "t2", "v3", "t3", "v4", "t4", "v5", "t5", "v6", "t6", "v7", "t7", "v8"]

prior_results_sampler = prior_results.sampler
az_idata_prior = az.from_emcee(prior_results_sampler, var_names=labels)

axes = az.plot_trace(
    az_idata_prior, 
    backend_kwargs={"constrained_layout":True}, 
    figsize=(10,20),
)

for i, axes_pair in enumerate(axes):
    ax1 = axes_pair[0]
    ax2 = axes_pair[1]
    ax1.set_xlabel("parameter value")
    ax1.set_ylabel("density value")
    ax2.set_xlabel("number of iterations")
    ax2.set_ylabel("parameter value")

######################################################################
#


######################################################################
# Sample the posterior
# ^^^^^^^^^^^^^^^^^^^^
# 

inversion_fixed_d_sampler = cofi.Inversion(sw_problem, inv_options_fixed_d_sampling)
inv_result_fixed_d_sampler = inversion_fixed_d_sampler.run()

######################################################################
#

sampler = inv_result_fixed_d_sampler.sampler
az_idata = az.from_emcee(sampler, var_names=labels)

######################################################################
#

az_idata.get("posterior")

######################################################################
#

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="initial model for damped solution", color_model="black",
                           forward_func=forward_sw, periods=synth_d_periods, 
                           label_d_pred="d_pred from initial model for damped solution", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=true_model, label_model="true model", color_model="orange",
                    forward_func=forward_sw, periods=synth_d_periods, 
                    label_d_pred="d_pred from true model", color_d_pred="orange", axes=axes)

# plot the model and d_pred for damped solution, and d_obs
plot_model_and_data(model=inv_result_optimiser.model, label_model="damped solution", color_model="green",
                    forward_func=forward_sw, periods=synth_d_periods,
                    label_d_pred="d_pred from damped solution", color_d_pred="green", axes=axes);

# plot randomly selected samples and data predictions from samples
flat_samples = sampler.get_chain(discard=500, thin=500, flat=True)
rand_indices = np.random.randint(len(flat_samples), size=100)
for idx in rand_indices:
    sample = flat_samples[idx]
    label_model = "sample models" if idx == 0 else None
    label_d_pred = "d_pred from samples" if idx == 0 else None
    plot_model_and_data(model=sample, label_model=label_model, color_model="gray",
                        forward_func=forward_sw, periods=synth_d_periods,
                        label_d_pred=label_d_pred, color_d_pred="gray", axes=axes, light_thin=True)

# plot d_obs
plot_data(d_obs, synth_d_periods, ax=axes[1], scatter=True, color="red", s=20, label="d_obs")

axes[0].set_ylim(170)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46))
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46));

######################################################################
#

axes = az.plot_trace(
    az_idata, 
    backend_kwargs={"constrained_layout":True},
    figsize=(10,20)
)

for i, axes_pair in enumerate(axes):
    ax1 = axes_pair[0]
    ax2 = axes_pair[1]
    ax1.axvline(true_model[i], linestyle='dotted', color='red')
    ax1.set_xlabel("parameter value")
    ax1.set_ylabel("density value")
    ax2.set_xlabel("number of iterations")
    ax2.set_ylabel("parameter value")

######################################################################
#


######################################################################
# **More steps?**
# 
# Due to time restrictions, we have only run 2_000 steps above, which
# might be enough for illustration purpose and sanity check, but are not
# enough for an actual inversion.
# 
# On a seperate experiment, we ran 200_000 steps instead and produced the
# following samples plot.
# 
# .. figure::
#    https://raw.githubusercontent.com/inlab-geo/cofi-examples/main/tutorials/rayleigh_wave_phase_velocity/illustrations/emcee_200_000_iterations.png?raw=true
#    :alt: Fixed-dimensional sampling results with 200_000 steps
# 
#    Fixed-dimensional sampling results with 200_000 steps
# 


######################################################################
# Trans-dimensional sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# **Prepare utilities for trans-dimensional sampling**
# 

def forward_for_bayesbay(state):
    vs = state["voronoi"]["vs"]
    voronoi_sites = state["voronoi"]["discretization"]
    depths = (voronoi_sites[:-1] + voronoi_sites[1:]) / 2
    thicknesses = depths - np.insert(depths[:-1], 0, 0)
    model = form_layercake_model(thicknesses, vs)
    return forward_sw(model, synth_d_periods)

######################################################################
#

targets = [bayesbay.likelihood.Target("rayleigh", d_obs, covariance_mat_inv=1/noise_level**2)]
fwd_funcs = [forward_for_bayesbay]
my_log_likelihood = bayesbay.likelihood.LogLikelihood(targets, fwd_funcs) 

######################################################################
#

param_vs = bayesbay.prior.UniformPrior(
    name="vs", 
    vmin=[2.7, 3.2, 3.75], 
    vmax=[4, 4.75, 5], 
    position=[0, 40, 80], 
    perturb_std=0.15
)

def param_vs_initialize(param, positions): 
    vmin, vmax = param.get_vmin_vmax(positions)
    sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
    for i in range (len(sorted_vals)):
        val = sorted_vals[i]
        vmin_i = vmin if np.isscalar(vmin) else vmin[i]
        vmax_i = vmax if np.isscalar(vmax) else vmax[i]
        if val < vmin_i or val > vmax_i:
            if val > vmax_i: sorted_vals[i] = vmax_i
            if val < vmin_i: sorted_vals[i] = vmin_i
    return sorted_vals

param_vs.set_custom_initialize(param_vs_initialize)

######################################################################
#

parameterization = bayesbay.parameterization.Parameterization(
    bayesbay.discretization.Voronoi1D(
        name="voronoi", 
        vmin=0, 
        vmax=150, 
        perturb_std=10, 
        n_dimensions=None, 
        n_dimensions_min=4, 
        n_dimensions_max=15, 
        parameters=[param_vs], 
    )
)
my_perturbation_funcs=parameterization.perturbation_funcs

######################################################################
#

n_chains=12
walkers_start = []
for i in range(n_chains):
    walkers_start.append(parameterization.initialize())

######################################################################
#


######################################################################
# **Define ``InversionOptions``**
# 

inv_options_trans_d_sampling = cofi.InversionOptions()
inv_options_trans_d_sampling.set_tool("bayesbay")
inv_options_trans_d_sampling.set_params(
    walkers_starting_states=walkers_start,
    perturbation_funcs=my_perturbation_funcs,
    log_like_ratio_func=my_log_likelihood,
    n_chains=n_chains, 
    n_iterations=3_000, 
    burnin_iterations=1_000,
    verbose=False, 
    save_every=200, 
)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inversion_trans_d_sampler = cofi.Inversion(sw_problem, inv_options_trans_d_sampling)
inv_result_trans_d_sampler = inversion_trans_d_sampler.run()

######################################################################
#

inverted_models = inv_result_trans_d_sampler.models
samples = []
for v, vs in zip(inverted_models["voronoi.discretization"], inverted_models["voronoi.vs"]):
    sample = form_voronoi_model(v, vs)
    samples.append(voronoi_to_layercake(sample))

######################################################################
#

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="initial model for damped solution", color_model="black",
                           forward_func=forward_sw, periods=synth_d_periods, 
                           label_d_pred="d_pred from initial model for damped solution", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=true_model, label_model="true model", color_model="orange",
                    forward_func=forward_sw, periods=synth_d_periods, 
                    label_d_pred="d_pred from true model", color_d_pred="orange", axes=axes)

# plot the model and d_pred for damped solution, and d_obs
plot_model_and_data(model=inv_result_optimiser.model, label_model="damped solution", color_model="green",
                    forward_func=forward_sw, periods=synth_d_periods,
                    label_d_pred="d_pred from damped solution", color_d_pred="green", axes=axes);

# plot randomly selected samples and data predictions from samples
for i, sample in enumerate(samples):
    label_model = "sample models" if i == 0 else None
    label_d_pred = "d_pred from samples" if i == 0 else None
    plot_model_and_data(model=sample, label_model=label_model, color_model="gray",
                        forward_func=forward_sw, periods=synth_d_periods,
                        label_d_pred=label_d_pred, color_d_pred="gray", axes=axes, light_thin=True)

# plot d_obs
plot_data(d_obs, synth_d_periods, ax=axes[1], scatter=True, color="red", s=20, label="d_obs")

axes[0].set_ylim(170)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46))
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46));

######################################################################
#


######################################################################
# Field data example
# ------------------
# 


######################################################################
# Read data
# ~~~~~~~~~
# 


######################################################################
# **Rayleigh observations**
# 

file_surf_data = "../../data/sw_rf_joint/data/SURF/nnall.dsp"

with open(file_surf_data, "r") as file:
    lines = file.readlines()
    surf_data = []
    for line in lines:
        row = line.strip().split()
        if "C" in row:
            surf_data.append([float(e) for e in row[5:8]])

field_d = np.array(surf_data)
field_d_periods = field_d[:,0]
field_d_obs = field_d[:,1]

######################################################################
#

ax = plot_data(field_d_obs, field_d_periods, color="brown", s=5, scatter=True,
             label="d_obs")
ax.legend();

######################################################################
#


######################################################################
# **Reference good model**
# 

file_end_mod = "../../data/sw_rf_joint/data/SURF/end.mod"

with open(file_end_mod, "r") as file:
    lines = file.readlines()
    ref_good_model = []
    for line in lines[12:]:
        row = line.strip().split()
        ref_good_model.append([float(row[0]), float(row[2])])

ref_good_model = np.array(ref_good_model)
ref_good_model = form_layercake_model(ref_good_model[:-1,0], ref_good_model[:,1])

######################################################################
#

_, ax = plt.subplots(figsize=(4,6))
plot_model(ref_good_model, ax=ax, alpha=1);

######################################################################
#


######################################################################
# Modified forward utility
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 

def forward_sw_interp(model, periods=field_d_periods):
    pysurf_periods = np.logspace(
        np.log(np.min(periods)), 
        np.log(np.max(periods+1)), 
        60,
        base=np.e, 
    )
    pysurf_dpred = forward_sw(model, pysurf_periods)
    interp_func = scipy.interpolate.interp1d(pysurf_periods, 
                                             pysurf_dpred, 
                                             fill_value="extrapolate")
    dpred = interp_func(periods)
    return dpred

######################################################################
#


######################################################################
# Optimisation
# ~~~~~~~~~~~~
# 

n_dims = 29

init_thicknesses = np.ones((n_dims//2,)) * 10
init_vs = np.ones((n_dims//2+1,)) * 4.0
init_model = form_layercake_model(init_thicknesses, init_vs)

######################################################################
#

my_reg = cofi.utils.QuadraticReg(
    weighting_matrix="damping", 
    model_shape=(n_dims,), 
    reference_model=init_model
)

######################################################################
#


######################################################################
# Optimisation with no damping
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

lamda = 0

kwargs = {
    "fwd": forward_sw_interp,
    "periods": field_d_periods, 
    "d_obs": field_d_obs, 
    "lamda": lamda, 
}
sw_field_problem_no_reg = cofi.BaseProblem()
sw_field_problem_no_reg.set_objective(my_objective, kwargs=kwargs)
sw_field_problem_no_reg.set_gradient(my_objective_gradient, kwargs=kwargs)
sw_field_problem_no_reg.set_hessian(my_objective_hessian, kwargs=kwargs)
sw_field_problem_no_reg.set_initial_model(init_model)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inv_sw_field_problem_no_reg = cofi.Inversion(sw_field_problem_no_reg, inv_options_optimiser)
inv_result_sw_field_no_reg = inv_sw_field_problem_no_reg.run()

######################################################################
#


######################################################################
# **Plot results**
# 

field_d_periods_logspace = np.logspace(
    np.log(np.min(field_d_periods)), 
    np.log(np.max(field_d_periods+1)), 
    60, 
    base=np.e, 
)

######################################################################
#

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="starting model", color_model="black",
                           forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                           label_d_pred="d_pred from starting model", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=ref_good_model, label_model="reference good model", color_model="red",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                    label_d_pred="d_pred from reference good model", color_d_pred="red", axes=axes)

# plot the model and d_pred for inverted model, and d_obs
plot_model_and_data(model=inv_result_sw_field_no_reg.model, 
                    label_model="inverted model from field data", color_model="purple",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace,
                    label_d_pred="d_pred from inverted model", color_d_pred="purple", axes=axes)

# plot d_obs
plot_data(field_d_obs, field_d_periods, ax=axes[1], scatter=True, color="orange", s=8, label="d_obs")

axes[0].set_ylim(100, 0)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46));

######################################################################
#


######################################################################
# Optimal damping
# ^^^^^^^^^^^^^^^
# 
# Again, we would like to find a good regularisation factor.
# 

lambdas = np.logspace(-6, 6, 15)

my_lcurve_problems = []
for lamb in lambdas:
    my_problem = cofi.BaseProblem()
    kwargs = {
        "fwd": forward_sw_interp,
        "periods": field_d_periods, 
        "d_obs": field_d_obs, 
        "lamda": lamb, 
    }
    my_problem.set_objective(my_objective, kwargs=kwargs)
    my_problem.set_gradient(my_objective_gradient, kwargs=kwargs)
    my_problem.set_hessian(my_objective_hessian, kwargs=kwargs)
    my_problem.set_initial_model(init_model)
    my_lcurve_problems.append(my_problem)

def my_callback(inv_result, i):
    m = inv_result.model
    res_norm = np.linalg.norm(forward_sw_interp(m, field_d_periods) - field_d_obs)
    reg_norm = np.sqrt(my_reg(m))
    print(f"Finished inversion with lambda={lambdas[i]}: {res_norm}, {reg_norm}")
    return res_norm, reg_norm

my_inversion_pool = cofi.utils.InversionPool(
    list_of_inv_problems=my_lcurve_problems,
    list_of_inv_options=inv_options_optimiser,
    callback=my_callback,
    parallel=False
)
all_res, all_cb_returns = my_inversion_pool.run()

l_curve_points = list(zip(*all_cb_returns))

######################################################################
#

# print all the lambdas
lambdas

######################################################################
#

# plot the L-curve
res_norm, reg_norm = l_curve_points
plt.plot(reg_norm, res_norm, '.-')
plt.xlabel(r'Norm of regularization term $||Wm||_2$')
plt.ylabel(r'Norm of residual $||g(m)-d||_2$')
for i in range(0, len(lambdas), 2):
    plt.annotate(f'{lambdas[i]:.1e}', (reg_norm[i], res_norm[i]), fontsize=8)

######################################################################
#


######################################################################
# Optimisation with damping
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# 

lamda = 0.14

kwargs = {
    "fwd": forward_sw_interp,
    "periods": field_d_periods, 
    "d_obs": field_d_obs, 
    "lamda": lamda, 
}
sw_field_problem = cofi.BaseProblem()
sw_field_problem.set_objective(my_objective, kwargs=kwargs)
sw_field_problem.set_gradient(my_objective_gradient, kwargs=kwargs)
sw_field_problem.set_hessian(my_objective_hessian, kwargs=kwargs)
sw_field_problem.set_initial_model(init_model)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inv_sw_field_problem = cofi.Inversion(sw_field_problem, inv_options_optimiser)
inv_result_sw_field = inv_sw_field_problem.run()

######################################################################
#


######################################################################
# **Plot results**
# 

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="starting model", color_model="black",
                           forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                           label_d_pred="d_pred from starting model", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=ref_good_model, label_model="reference good model", color_model="red",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                    label_d_pred="d_pred from reference good model", color_d_pred="red", axes=axes)

# plot the model and d_pred for inverted model, and d_obs
plot_model_and_data(model=inv_result_sw_field.model, 
                    label_model="inverted model from field data", color_model="purple",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace,
                    label_d_pred="d_pred from inverted model", color_d_pred="purple", axes=axes)

# plot d_obs
plot_data(field_d_obs, field_d_periods, ax=axes[1], scatter=True, color="orange", s=8, label="d_obs")

axes[0].set_ylim(100, 0)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46));

######################################################################
#


######################################################################
# Fixed-dimensional sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We are going to use the same sets of log prior, and we will rewrite the
# log likelihood function to apply on the field data.
# 

thick_min = 3
thick_max = 10
vs_min = 2
vs_max = 5.5

def my_log_prior(model):
    thicknesses, vs = split_layercake_model(model)
    thicknesses_out_of_bounds = (thicknesses < thick_min) | (thicknesses > thick_max)
    vs_out_of_bounds = (vs < vs_min) | (vs > vs_max)
    if np.any(thicknesses_out_of_bounds) or np.any(vs_out_of_bounds):
        return float("-inf")
    log_prior = - np.log(thick_max - thick_min) * len(thicknesses) \
                - np.log(vs_max - vs_min) * len(vs)
    return log_prior

######################################################################
#

# estimate the data noise
d_pred_from_optimiser = forward_sw_interp(inv_result_sw_field.model, field_d_periods)
noise_level = np.std(field_d_obs - d_pred_from_optimiser)
Cdinv = np.eye(len(field_d_obs))/(noise_level**2)

print(f"Estimated noise level: {noise_level}")

######################################################################
#

def my_log_likelihood(model):
    try:
        d_pred = forward_sw_interp(model, field_d_periods)
    except:
        return float("-inf")
    residual = field_d_obs - d_pred
    return -0.5 * residual @ (Cdinv @ residual).T

######################################################################
#

n_walkers = 60

my_walkers_start = np.ones((n_walkers, n_dims)) * inv_result_sw_field.model
for i in range(n_walkers):
    my_walkers_start[i,:] += np.random.normal(0, 0.3, n_dims)

######################################################################
#

sw_field_problem.set_log_prior(my_log_prior)
sw_field_problem.set_log_likelihood(my_log_likelihood)

######################################################################
#


######################################################################
# **Define ``InversionOptions``**
# 

inv_options_fixed_d_sampling = cofi.InversionOptions()
inv_options_fixed_d_sampling.set_tool("emcee")
inv_options_fixed_d_sampling.set_params(
    nwalkers=n_walkers,
    nsteps=20_000,
    initial_state=my_walkers_start,
    skip_initial_state_check=True,
    progress=True
)

######################################################################
#


######################################################################
# Sample the posterior
# ^^^^^^^^^^^^^^^^^^^^
# 

inversion_fixed_d_sampler_field = cofi.Inversion(sw_field_problem,
                                                 inv_options_fixed_d_sampling)
inv_result_fixed_d_sampler_field = inversion_fixed_d_sampler_field.run()

######################################################################
#

sampler = inv_result_fixed_d_sampler.sampler
az_idata = az.from_emcee(sampler, var_names=labels)

######################################################################
#

az_idata.get("posterior")

######################################################################
#

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="starting model", color_model="black",
                           forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                           label_d_pred="d_pred from starting model", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=ref_good_model, label_model="reference good model", color_model="red",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                    label_d_pred="d_pred from reference good model", color_d_pred="red", axes=axes)

# plot the model and d_pred for inverted model, and d_obs
plot_model_and_data(model=inv_result_sw_field.model, 
                    label_model="inverted model from field data", color_model="green",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace,
                    label_d_pred="d_pred from inverted model", color_d_pred="green", axes=axes)

# plot randomly selected samples and data predictions from samples
flat_samples = sampler.get_chain(discard=1000, thin=300, flat=True)
rand_indices = np.random.randint(len(flat_samples), size=100)
for idx in rand_indices:
    sample = flat_samples[idx]
    label_model = "sample models" if idx == 0 else None
    label_d_pred = "d_pred from samples" if idx == 0 else None
    plot_model_and_data(model=sample, label_model=label_model, color_model="gray",
                        forward_func=forward_sw_interp, periods=field_d_periods_logspace,
                        label_d_pred=label_d_pred, color_d_pred="gray", axes=axes, light_thin=True)

# plot d_obs
plot_data(field_d_obs, field_d_periods, ax=axes[1], scatter=True, color="orange", s=8, label="d_obs")

axes[0].set_ylim(100, 0)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46));

######################################################################
#


######################################################################
# **More steps**
# 
# Similar to our earlier fixed-dimensional sampling run on the synthetic
# data, we are not sampling enough due to time limit.
# 
# On a seperate experiment, we ran 200_000 steps and produced the
# following samples plot.
# 
# .. figure::
#    https://raw.githubusercontent.com/inlab-geo/cofi-examples/main/tutorials/rayleigh_wave_phase_velocity/illustrations/emcee_200_000_iterations_field.png?raw=true
#    :alt: Fixed-dimensional sampling results on field data with 200_000
#    steps
# 
#    Fixed-dimensional sampling results on field data with 200_000 steps
# 


######################################################################
# Trans-dimensional sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

def forward_interp_for_bayesbay(state):
    vs = state["voronoi"]["vs"]
    voronoi_sites = state["voronoi"]["discretization"]
    depths = (voronoi_sites[:-1] + voronoi_sites[1:]) / 2
    thicknesses = depths - np.insert(depths[:-1], 0, 0)
    model = form_layercake_model(thicknesses, vs)
    return forward_sw_interp(model, field_d_periods)

######################################################################
#

targets = [bayesbay.likelihood.Target("rayleigh", field_d_obs, covariance_mat_inv=1/noise_level**2)]
fwd_funcs = [forward_interp_for_bayesbay]
my_log_likelihood = bayesbay.likelihood.LogLikelihood(targets, fwd_funcs) 

######################################################################
#

param_vs = bayesbay.prior.UniformPrior(
    name="vs", 
    vmin=[2.7, 3.2, 3.75], 
    vmax=[4, 4.75, 5], 
    position=[0, 40, 80], 
    perturb_std=0.15
)

def param_vs_initialize(param, positions): 
    vmin, vmax = param.get_vmin_vmax(positions)
    sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
    for i in range (len(sorted_vals)):
        val = sorted_vals[i]
        vmin_i = vmin if np.isscalar(vmin) else vmin[i]
        vmax_i = vmax if np.isscalar(vmax) else vmax[i]
        if val < vmin_i or val > vmax_i:
            if val > vmax_i: sorted_vals[i] = vmax_i
            if val < vmin_i: sorted_vals[i] = vmin_i
    return sorted_vals

param_vs.set_custom_initialize(param_vs_initialize)

######################################################################
#

parameterization = bayesbay.parameterization.Parameterization(
    bayesbay.discretization.Voronoi1D(
        name="voronoi", 
        vmin=0, 
        vmax=150, 
        perturb_std=10, 
        n_dimensions=None, 
        n_dimensions_min=4, 
        n_dimensions_max=20, 
        parameters=[param_vs], 
    )
)
my_perturbation_funcs = parameterization.perturbation_funcs

######################################################################
#

n_chains=12
walkers_start = []
for i in range(n_chains):
    walkers_start.append(parameterization.initialize())

######################################################################
#


######################################################################
# **Define ``InversionOptions``**
# 

inv_options_field_trans_d_sampling = cofi.InversionOptions()
inv_options_field_trans_d_sampling.set_tool("bayesbay")
inv_options_field_trans_d_sampling.set_params(
    walkers_starting_states=walkers_start,
    perturbation_funcs=my_perturbation_funcs,
    log_like_ratio_func=my_log_likelihood,
    n_chains=n_chains, 
    n_iterations=3_000, 
    burnin_iterations=1_000,
    verbose=False, 
    save_every=200, 
)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inversion_field_trans_d_sampler = cofi.Inversion(sw_field_problem, 
                                                 inv_options_field_trans_d_sampling)
inv_result_field_trans_d_sampler = inversion_field_trans_d_sampler.run()

######################################################################
#

inverted_models = inv_result_field_trans_d_sampler.models
samples = []
for v, vs in zip(inverted_models["voronoi.discretization"], inverted_models["voronoi.vs"]):
    sample = form_voronoi_model(v, vs)
    samples.append(voronoi_to_layercake(sample))

######################################################################
#

# plot the model and d_pred for starting model
axes = plot_model_and_data(model=init_model, label_model="starting model", color_model="black",
                           forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                           label_d_pred="d_pred from starting model", color_d_pred="black")

# plot the model and d_pred for true model
plot_model_and_data(model=ref_good_model, label_model="reference good model", color_model="red",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace, 
                    label_d_pred="d_pred from reference good model", color_d_pred="red", axes=axes)

# plot the model and d_pred for inverted model, and d_obs
plot_model_and_data(model=inv_result_sw_field.model, 
                    label_model="inverted model from field data", color_model="green",
                    forward_func=forward_sw_interp, periods=field_d_periods_logspace,
                    label_d_pred="d_pred from inverted model", color_d_pred="green", axes=axes)

# plot randomly selected samples and data predictions from samples
flat_samples = sampler.get_chain(discard=1000, thin=300, flat=True)
rand_indices = np.random.randint(len(flat_samples), size=100)
for i, sample in enumerate(samples):
    label_model = "sample models" if i == 0 else None
    label_d_pred = "d_pred from samples" if i == 0 else None
    plot_model_and_data(model=sample, label_model=label_model, color_model="gray",
                        forward_func=forward_sw_interp, periods=field_d_periods_logspace,
                        label_d_pred=label_d_pred, color_d_pred="gray", axes=axes, light_thin=True)

# plot d_obs
plot_data(field_d_obs, field_d_periods, ax=axes[1], scatter=True, color="orange", s=8, label="d_obs")

axes[0].set_ylim(100, 0)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.46));

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# ---------
# 

watermark_list = ["cofi", "numpy", "matplotlib", "scipy", "emcee", "bayesbay"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1