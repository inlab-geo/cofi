"""
Surface wave and receiver function - joint inversion (synthetic data)
=====================================================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/sw_rf_joint/sw_rf_joint_synthetic.ipynb
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
# Here we extend the example where CoFI has been used for the inversion of
# Rayleigh wave phase velocities for a 1D layered earth to a joint
# inversion where we also account for receiver functions.
# 
# **Learning outcomes**
# 
# -  A demonstration of CoFI’s ability to switch between parameter
#    estimation and ensemble methods.
# -  An application of CoFI for a joint inversion, here of Rayleigh wave
#    pahse velocity and receiver function data, to a synthetic dataset
# 


######################################################################
# Utilities preparation
# ---------------------
# 
# In this example, we look at a joint inversion problem of surface wave
# and receiver function.
# 
# We use ``pysurf96`` for computing the forward step of surface wave, and
# use ``espresso`` for receiver function calculations.
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi 
# !pip install git+https://github.com/miili/pysurf96.git
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/examples/sw_rf_joint

######################################################################
#

import glob
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pysurf96
import bayesbay
import cofi
import espresso

######################################################################
#

np.seterr(all="ignore");

######################################################################
#


######################################################################
# **Model vector**
# 

# display theory on the 1D model parameterisation
from IPython.display import display, Markdown

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
# **Interfacing to pysurf96**
# 

# display theory on the using the forward solver
with open("../../theory/geo_surface_wave_dispersion2.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#

VP_VS = 1.77
RHO_VP_K = 0.32
RHO_VP_B = 0.77
periods = np.geomspace(3, 80, 20)

######################################################################
#

# forward through pysurf96
def forward_sw(model, periods):
    thicknesses, vs = split_layercake_model(model)
    thicknesses = np.append(thicknesses, 10)
    vp = vs * VP_VS
    rho = RHO_VP_K * vp + RHO_VP_B
    return pysurf96.surf96(
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

######################################################################
#


######################################################################
# **Interfacing to Espresso**
# 

# obtain the receiver function library
rf_lib = espresso.ReceiverFunctionInversionKnt().rf

######################################################################
#

t_shift = 5
t_duration = 25
t_sampling_interval = 0.1
gauss = 1
ray_param_s_km = 0.07

######################################################################
#

# forward through rf in espresso
def forward_rf(
    model, 
    t_shift=t_shift, 
    t_duration=t_duration, 
    t_sampling_interval=t_sampling_interval, 
    gauss=gauss, 
    ray_param_s_km=ray_param_s_km,
    return_times=False
):
    h, vs = split_layercake_model(model)
    data = rf_lib.rf_calc(ps=0, thik=h, beta=vs, kapa=np.ones((len(vs),))*VP_VS, p=ray_param_s_km, 
                      duration=t_duration, dt=t_sampling_interval, shft=t_shift, gauss=gauss)
    if return_times:
        times = np.arange(len(data)) * t_sampling_interval - t_shift
        return data, times
    else:
        return data

######################################################################
#


######################################################################
# **Numerical Jacobian**
# 

def jacobian(model, data_length, fwd, fwd_kwargs, relative_step=0.01):
    jac = np.zeros((data_length, len(model)))
    original_dpred = fwd(model, **fwd_kwargs)
    for i in range(len(model)):
        perturbed_model = model.copy()
        step = relative_step * model[i]
        perturbed_model[i] += step
        perturbed_dpred = fwd(perturbed_model, **fwd_kwargs)
        derivative = (perturbed_dpred - original_dpred) / step
        jac[:, i] = derivative
    return jac

######################################################################
#


######################################################################
# **Plotting**
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

def plot_data(x, y, ax=None, scatter=False, xlabel=None, ylabel=None, 
              title="surface wave data", **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    plotting_style = {
        "linewidth": kwargs.pop("linewidth", kwargs.pop("lw", 1)),
        "alpha": 1,
        "color": kwargs.pop("color", kwargs.pop("c", "blue")),
    }
    plotting_style.update(**kwargs)
    if scatter:
        ax.scatter(x, y, **plotting_style)
    else:
        ax.plot(x, y, **plotting_style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

######################################################################
#

def plot_sw_data(rayleigh_phase_velocities, periods, ax=None, scatter=False, 
                 title="surface wave data", **kwargs):
    return plot_data(x=periods, 
                     y=rayleigh_phase_velocities, 
                     ax=ax, 
                     scatter=scatter, 
                     title=title, 
                     xlabel="Periods (s)", 
                     ylabel="Rayleigh phase velocities (km/s)", 
                     **kwargs)

def plot_rf_data(rf_data, rf_times, ax=None, scatter=False,
                 title="receiver function data", **kwargs):
    return plot_data(x=rf_times, 
                     y=rf_data, 
                     ax=ax, 
                     scatter=scatter, 
                     title=title, 
                     xlabel="Times (s)", 
                     ylabel="Receiver function amplitudes", 
                     **kwargs)

######################################################################
#


######################################################################
# Generate synthetic data
# -----------------------
# 

true_thickness = np.array([10, 10, 15, 20, 20, 20, 20, 20])
true_voronoi_positions = np.array([5, 15, 25, 45, 65, 85, 105, 125, 145])
true_vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
true_model = form_layercake_model(true_thickness, true_vs)

######################################################################
#

RAYLEIGH_STD = 0.02
RF_STD = 0.015

######################################################################
#

rayleigh = forward_sw(true_model, periods)
rayleigh_dobs = rayleigh + np.random.normal(0, RAYLEIGH_STD, rayleigh.size)

rf, rf_times = forward_rf(true_model, return_times=True)
rf_dobs = rf + np.random.normal(0, RF_STD, rf.size)

######################################################################
#

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 2.5, 2.5]})

# plot true model
plot_model(true_model, ax=ax1, alpha=1, color="r", label="true model")
ax1.grid()

# plot surface wave data
plot_sw_data(rayleigh, periods, ax=ax2, color="orange", label="true rayleigh data (noiseless)")
plot_sw_data(rayleigh_dobs, periods, ax=ax2, scatter=True, color="brown", s=4, 
          label="observed rayleigh data (noisy)")
ax2.grid()

# plot receiver function data
plot_rf_data(rf, rf_times, ax=ax3, color="lightblue", label="true receiver function data (noiseless)")
plot_rf_data(rf_dobs, rf_times, ax=ax3, scatter=True, color="darkblue", s=2, 
          label="observed receiver function data (noisy)")
ax3.grid()

ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))

plt.tight_layout()

######################################################################
#


######################################################################
# Optimisation
# ------------
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

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 2.5, 2.5]})

# plot true model
plot_model(init_model, ax=ax1, alpha=1, color="purple", label="starting model")
plot_model(true_model, ax=ax1, alpha=1, color="r", label="true model")
ax1.grid()

# plot surface wave data
plot_sw_data(forward_sw(init_model, periods), periods, ax=ax2, color="purple",
             label="predicted rayleigh data from starting model")
plot_sw_data(rayleigh_dobs, periods, ax=ax2, scatter=True, color="orange", s=4, 
          label="observed rayleigh data (noisy)")
ax2.grid()

# plot receiver function data
plot_rf_data(forward_rf(init_model), rf_times, ax=ax3, color="purple",
             label="predicted receiver function data from starting model")
plot_rf_data(rf_dobs, rf_times, ax=ax3, scatter=True, color="darkblue", s=2, 
          label="observed receiver function data (noisy)")
ax3.grid()

ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))

plt.tight_layout()

######################################################################
#

my_reg = cofi.utils.QuadraticReg(
    weighting_matrix="damping", 
    model_shape=(n_dims,),
    reference_model=init_model
)

######################################################################
#

def my_objective(model, fwd_funcs, d_obs_list, lamda=1.0):
    data_misfit = 0
    for (fwd, fwd_kwargs), d_obs in zip(fwd_funcs, d_obs_list):
        d_pred = fwd(model, **fwd_kwargs)
        data_misfit += np.sum((d_obs - d_pred) ** 2)
    reg = my_reg(model)
    return data_misfit + lamda * reg

def my_objective_gradient(model, fwd_funcs, d_obs_list, lamda=1.0):
    data_misfit_grad = 0
    for (fwd, fwd_kwargs), d_obs in zip(fwd_funcs, d_obs_list):
        d_pred = fwd(model, **fwd_kwargs)
        jac = jacobian(model, len(d_obs), fwd, fwd_kwargs)
        data_misfit_grad += -2 * jac.T @ (d_obs - d_pred)
    reg_grad = my_reg.gradient(model)
    return data_misfit_grad + lamda * reg_grad

def my_objective_hessian(model, fwd_funcs, d_obs_list, lamda=1.0):
    data_misfit_hess = 0
    for (fwd, fwd_kwargs), d_obs in zip(fwd_funcs, d_obs_list):
        jac = jacobian(model, len(d_obs), fwd, fwd_kwargs)
        data_misfit_hess += 2 * jac.T @ jac
    reg_hess = my_reg.hessian(model)
    return data_misfit_hess + lamda * reg_hess

######################################################################
#

fwd_funcs = [
    (forward_sw, {"periods": periods}), 
    (forward_rf, {
        "t_shift": t_shift, 
        "t_duration": t_duration, 
        "t_sampling_interval": t_sampling_interval, 
        "gauss": gauss, 
        "ray_param_s_km": ray_param_s_km
    })
]

d_obs_list = [rayleigh_dobs, rf_dobs]

######################################################################
#


######################################################################
# Optimisation with no damping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

lamda = 0

kwargs = {
    "fwd_funcs": fwd_funcs, 
    "d_obs_list": d_obs_list, 
    "lamda": lamda
}
joint_problem_no_reg = cofi.BaseProblem()
joint_problem_no_reg.set_objective(my_objective, kwargs=kwargs)
joint_problem_no_reg.set_gradient(my_objective_gradient, kwargs=kwargs)
joint_problem_no_reg.set_hessian(my_objective_hessian, kwargs=kwargs)
joint_problem_no_reg.set_initial_model(init_model)

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

inv_optimiser_no_reg = cofi.Inversion(joint_problem_no_reg, inv_options_optimiser)
inv_res_optimiser_no_reg = inv_optimiser_no_reg.run()

######################################################################
#


######################################################################
# **Plot results**
# 

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 2.5, 2.5]})

# plot true model
plot_model(inv_res_optimiser_no_reg.model, ax=ax1, alpha=1, color="purple", 
           label="inverted model")
plot_model(true_model, ax=ax1, alpha=1, color="r", label="true model")
ax1.grid()

# plot surface wave data
plot_sw_data(forward_sw(inv_res_optimiser_no_reg.model, periods), 
             periods, ax=ax2, color="purple",
             label="predicted rayleigh data from inverted model")
plot_sw_data(rayleigh_dobs, periods, ax=ax2, scatter=True, color="orange", s=4, 
          label="observed rayleigh data (noisy)")
ax2.grid()

# plot receiver function data
plot_rf_data(forward_rf(inv_res_optimiser_no_reg.model), rf_times, 
             ax=ax3, color="purple",
             label="predicted receiver function data from inverted model")
plot_rf_data(rf_dobs, rf_times, ax=ax3, scatter=True, color="darkblue", s=2, 
          label="observed receiver function data (noisy)")
ax3.grid()

ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))

plt.tight_layout()

######################################################################
#


######################################################################
# Optimal damping
# ~~~~~~~~~~~~~~~
# 
# Oh no! The inverted model doesn’t look good, and it even has negative
# thicknesses.
# 
# Maybe adding a damping term to our objective function would help… but
# how can we determine a good damping factor?
# 
# The ``InversionPool`` in CoFI can help you answer it.
# 

lambdas = np.logspace(-6, 6, 15)

my_lcurve_problems = []
for lamb in lambdas:
    my_problem = cofi.BaseProblem()
    kwargs = {
        "fwd_funcs": fwd_funcs, 
        "d_obs_list": d_obs_list, 
        "lamda": lamb
    }
    my_problem.set_objective(my_objective, kwargs=kwargs)
    my_problem.set_gradient(my_objective_gradient, kwargs=kwargs)
    my_problem.set_hessian(my_objective_hessian, kwargs=kwargs)
    my_problem.set_initial_model(init_model)
    my_lcurve_problems.append(my_problem)

def my_callback(inv_result, i):
    m = inv_result.model
    res_norm = 0
    for (fwd, fwd_kwargs), d_obs in zip(fwd_funcs, d_obs_list):
        d_pred = fwd(m, **fwd_kwargs)
        res_norm += np.sum((d_obs - d_pred) ** 2)
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# From the L-curve plot above, it seems that a damping factor of around
# 0.14 would be good.
# 

lamda = 0.14

kwargs = {
    "fwd_funcs": fwd_funcs, 
    "d_obs_list": d_obs_list, 
    "lamda": lamda
}
joint_problem = cofi.BaseProblem()
joint_problem.set_objective(my_objective, kwargs=kwargs)
joint_problem.set_gradient(my_objective_gradient, kwargs=kwargs)
joint_problem.set_hessian(my_objective_hessian, kwargs=kwargs)
joint_problem.set_initial_model(init_model)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inv_optimiser = cofi.Inversion(joint_problem, inv_options_optimiser)
inv_res_optimiser = inv_optimiser.run()

######################################################################
#


######################################################################
# **Plot results**
# 

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 2.5, 2.5]})

# plot true model
plot_model(inv_res_optimiser.model, ax=ax1, alpha=1, color="purple", 
           label="inverted model")
plot_model(true_model, ax=ax1, alpha=1, color="r", label="true model")
ax1.grid()

# plot surface wave data
plot_sw_data(forward_sw(inv_res_optimiser_no_reg.model, periods), 
             periods, ax=ax2, color="purple",
             label="predicted rayleigh data from inverted model")
plot_sw_data(rayleigh_dobs, periods, ax=ax2, scatter=True, color="orange", s=4, 
          label="observed rayleigh data (noisy)")
ax2.grid()

# plot receiver function data
plot_rf_data(forward_rf(inv_res_optimiser.model), rf_times, 
             ax=ax3, color="purple",
             label="predicted receiver function data from inverted model")
plot_rf_data(rf_dobs, rf_times, ax=ax3, scatter=True, color="darkblue", s=2, 
          label="observed receiver function data (noisy)")
ax3.grid()

ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))
ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5))

plt.tight_layout()

######################################################################
#


######################################################################
# Fixed-dimensional sampling
# --------------------------
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
    log_prior = - np.log(thick_max - thick_min) * len(thicknesses) \
                - np.log(vs_max - vs_min) * len(vs)
    return log_prior

######################################################################
#

# inverse data covariance matrix
Cdinv_rayleigh = np.eye(len(rayleigh_dobs)) / (RAYLEIGH_STD**2)
Cdinv_rf = np.eye(len(rf_dobs)) / (RF_STD**2)
Cdinv_list = [Cdinv_rayleigh, Cdinv_rf]

def my_log_likelihood(
    model, 
    fwd_funcs=fwd_funcs, 
    d_obs_list=d_obs_list, 
    Cdinv_list=Cdinv_list
):
    log_like_sum = 0
    for (fwd, fwd_kwargs), d_obs, Cdinv in zip(fwd_funcs, d_obs_list, Cdinv_list):
        try:
            d_pred = fwd(model, **fwd_kwargs)
        except:
            return float("-inf")
        residual = d_obs - d_pred
        log_like_sum += -0.5 * residual @ (Cdinv @ residual).T
    return log_like_sum

######################################################################
#

n_walkers = 40

my_walkers_start = np.ones((n_walkers, n_dims)) * inv_res_optimiser.model
for i in range(n_walkers):
    my_walkers_start[i,:] += np.random.normal(0, 0.5, n_dims)

######################################################################
#

joint_problem.set_log_prior(my_log_prior)
joint_problem.set_log_likelihood(my_log_likelihood)

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
# Sample the prior
# ~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~
# 

inversion_fixed_d_sampler = cofi.Inversion(joint_problem, inv_options_fixed_d_sampling)
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

flat_samples = sampler.get_chain(discard=500, thin=500, flat=True)
rand_indices = np.random.randint(len(flat_samples), size=100)

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 3, 3]})

ax1.set_ylim(170)

# plot samples and data predictions from samples
for idx in rand_indices:
    sample = flat_samples[idx]
    plot_model(sample, ax=ax1, alpha=0.2, lw=0.5, color="gray")
    plot_sw_data(forward_sw(sample, periods), periods, 
                 ax=ax2, alpha=0.2, lw=0.5, color="gray")
    plot_rf_data(forward_rf(sample), rf_times,
                 ax=ax3, alpha=0.2, lw=0.5, color="gray")

# add labels to samples
sample_0 = flat_samples[rand_indices[0]]
plot_model(sample_0, ax=ax1, alpha=0.5, lw=0.5, color="gray", label="samples")
plot_sw_data(forward_sw(sample_0, periods), periods, ax=ax2, 
             alpha=0.5, lw=0.5, color="gray", label="rayleigh_dpred from samples")
plot_rf_data(forward_rf(sample_0), rf_times, ax=ax3, 
             alpha=0.5, lw=0.5, color="gray", label="rf_dpred from samples")

# plot true model and data observations
plot_model(true_model, ax=ax1, alpha=1, color="r", label="true model")
plot_sw_data(rayleigh_dobs, periods, ax=ax2, scatter=True, color="r", s=4, 
          label="rayleigh_dobs")
plot_rf_data(rf_dobs, rf_times, ax=ax3, scatter=True, color="r", s=2, 
          label="rf_dobs")

# plot damped optimisation result
plot_model(inv_res_optimiser.model, ax=ax1, alpha=1, color="green", 
           label="damped solution")
plot_sw_data(forward_sw(inv_res_optimiser_no_reg.model, periods), 
             periods, ax=ax2, color="green",
             label="rayleigh_dpred from damped solution")
plot_rf_data(forward_rf(inv_res_optimiser.model), rf_times, 
             ax=ax3, color="green",
             label="rf_dpred from damped solution")

# plot initial model for dampied optimsiation
plot_model(init_model, ax=ax1, alpha=1, color="purple", 
           label="initial model for damped solution")
plot_sw_data(forward_sw(init_model, periods), periods, ax=ax2, color="purple",
             label="rayleigh_dpred from initial model for damped solution")
plot_rf_data(forward_rf(init_model), rf_times, ax=ax3, color="purple",
             label="rf_dpred from initial model for damped solution")

ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax1.grid()
ax2.grid()
ax3.grid()

plt.tight_layout()

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
# Trans-dimensional sampling
# --------------------------
# 


######################################################################
# **Prepare utilities for trans-d sampling**
# 

def fwd_rayleigh_for_bayesbay(state):
    vs = state["voronoi"]["vs"]
    voronoi_sites = state["voronoi"]["discretization"]
    depths = (voronoi_sites[:-1] + voronoi_sites[1:]) / 2
    thicknesses = depths - np.insert(depths[:-1], 0, 0)
    model = form_layercake_model(thicknesses, vs)
    return forward_sw(model, periods)

def fwd_rf_for_bayesbay(state):
    vs = state["voronoi"]["vs"]
    voronoi_sites = state["voronoi"]["discretization"]
    depths = (voronoi_sites[:-1] + voronoi_sites[1:]) / 2
    thicknesses = depths - np.insert(depths[:-1], 0, 0)
    model = form_layercake_model(thicknesses, vs)
    return forward_rf(model)

######################################################################
#

targets = [
    bayesbay.Target("rayleigh", rayleigh_dobs, covariance_mat_inv=1 / RAYLEIGH_STD**2),
    bayesbay.Target("rf", rf_dobs, covariance_mat_inv=1 / RF_STD**2),
]
forward_funcs = [fwd_rayleigh_for_bayesbay, fwd_rf_for_bayesbay]

my_log_likelihood = bayesbay.LogLikelihood(targets, forward_funcs)

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
my_perturbation_funcs = parameterization.perturbation_functions

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

inv_options_trans_d = cofi.InversionOptions()
inv_options_trans_d.set_tool("bayesbay")
inv_options_trans_d.set_params(
    walkers_starting_states=walkers_start,
    perturbation_funcs=my_perturbation_funcs,
    log_like_ratio_func=my_log_likelihood,
    n_chains=n_chains, 
    n_iterations=2_000, 
    burnin_iterations=1_000,
    verbose=False, 
    save_every=200, 
)

######################################################################
#


######################################################################
# **Define ``Inversion`` and run**
# 

inversion_trans_d_sampler = cofi.Inversion(joint_problem, inv_options_trans_d)
inv_res_trans_d_sampler = inversion_trans_d_sampler.run()

######################################################################
#

saved_states = inv_res_trans_d_sampler.models
samples_voronoi = saved_states["voronoi.discretization"]
samples_vs = saved_states["voronoi.vs"]
interp_depths = np.arange(150, dtype=float)

######################################################################
#

rand_indices = np.random.randint(len(samples_voronoi), size=100)

_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 3, 3]})

ax1.set_ylim(170)

# plot samples and data predictions from samples
for idx in rand_indices:
    sample_voronoi = form_voronoi_model(samples_voronoi[idx], samples_vs[idx])
    sample = voronoi_to_layercake(sample_voronoi)
    plot_model(sample, ax=ax1, alpha=0.2, lw=0.5, color="gray")
    plot_sw_data(forward_sw(sample, periods), periods, 
                 ax=ax2, alpha=0.2, lw=0.5, color="gray")
    plot_rf_data(forward_rf(sample), rf_times,
                 ax=ax3, alpha=0.2, lw=0.5, color="gray")

# add labels to samples
sample_0_voronoi = form_voronoi_model(samples_voronoi[0], samples_vs[0])
sample_0 = voronoi_to_layercake(sample_0_voronoi)
plot_model(sample_0, ax=ax1, alpha=0.2, lw=0.5, color="gray", label="samples")
plot_sw_data(forward_sw(sample_0, periods), periods, ax=ax2, 
             alpha=0.2, lw=0.5, color="gray", label="rayleigh_dpred from samples")
plot_rf_data(forward_rf(sample_0), rf_times, ax=ax3, 
             alpha=0.2, lw=0.5, color="gray", label="rf_dpred from samples")

# plot true model and data observations
plot_model(true_model, ax=ax1, alpha=1, color="r", label="true model")
plot_sw_data(rayleigh_dobs, periods, ax=ax2, scatter=True, color="r", s=4, 
          label="rayleigh_dobs")
plot_rf_data(rf_dobs, rf_times, ax=ax3, scatter=True, color="r", s=2, 
          label="rf_dobs")

# plot damped optimisation result
plot_model(inv_res_optimiser.model, ax=ax1, alpha=1, color="green", 
           label="damped solution")
plot_sw_data(forward_sw(inv_res_optimiser_no_reg.model, periods), 
             periods, ax=ax2, color="green",
             label="rayleigh_dpred from damped solution")
plot_rf_data(forward_rf(inv_res_optimiser.model), rf_times, 
             ax=ax3, color="green",
             label="rf_dpred from damped solution")

# plot initial model for dampied optimsiation
plot_model(init_model, ax=ax1, alpha=1, color="purple", 
           label="initial model for damped solution")
plot_sw_data(forward_sw(init_model, periods), periods, ax=ax2, color="purple",
             label="rayleigh_dpred from initial model for damped solution")
plot_rf_data(forward_rf(init_model), rf_times, ax=ax3, color="purple",
             label="rf_dpred from initial model for damped solution")

ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax1.grid()
ax2.grid()
ax3.grid()

plt.tight_layout()

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# ---------
# 

watermark_list = ["cofi", "espresso", "numpy", "matplotlib", "scipy", "bayesbay"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1