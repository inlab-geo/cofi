"""
Receiver Function Inversion with the Neighbourhood Algorithm
============================================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/receiver_function/receiver_function_neighpy.ipynb
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


######################################################################
# 1. Introduction 
# ----------------
# 
# 1.1 Receiver functions 
# ~~~~~~~~~~~~~~~~~~~~~~~
# 

# display theory on receiver functions
from IPython.display import display, Markdown

with open("../../theory/geo_receiver_function.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#


######################################################################
# 1.2 Neighbourhood Algorithm 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# display theory on the Neighbourhood Algorithm
from IPython.display import display, Markdown

with open("../../theory/neighbourhood_algorithm.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#


######################################################################
# 1.3 CoFI and the NA 
# ~~~~~~~~~~~~~~~~~~~~
# 
# The implementation of the NA that ``cofi`` wraps is called
# ```neighpy`` <https://github.com/inlab-geo/neighpy>`__. This
# implementation implements both phases of the NA as described in the
# original papers.
# 
# In this notebook we run the two stages of the NA **separately** using
# the ``neighpyI`` and ``neighpyII`` tools in CoFI:
# 
# -  **Stage I** (``neighpyI``) — Direct search. An ensemble optimiser
#    that minimises a user-defined **misfit** function to explore the
#    parameter space.
# -  **Stage II** (``neighpyII``) — Appraisal. An MCMC resampler that
#    resamples the Stage I ensemble according to a user-supplied **log
#    posterior probability density** (log-PPD). No additional forward
#    model evaluations are required.
# 
# The key insight is that the log-PPD used in Stage II does not have to
# equal the negative of the misfit used in Stage I. The relationship
# between the two is a modelling choice that depends on the assumed
# likelihood and prior. In this notebook we use
# :math:`\log p(\mathbf{m}|\mathbf{d}) = -\Phi(\mathbf{m},\mathbf{d})`,
# which corresponds to a Gaussian likelihood with a uniform prior, but we
# make this choice explicit so the user can modify it.
# 
# We apply the NA to two receiver function inversion problems of
# increasing complexity:
# 
# 1. **Problem 1** — Invert for S-wave velocities only (4 parameters,
#    layer depths fixed)
# 2. **Problem 2** — Invert for both layer depths and S-wave velocities (8
#    parameters)
# 
# ..
# 
#    **Note:** The hyperparameters in this notebook are intentionally set
#    low for a quick demonstration run. Results may not be fully
#    converged. To improve them, increase ``n_initial_samples``,
#    ``n_iterations``, and ``n_resample`` in the cells below.
# 


######################################################################
# --------------
# 


######################################################################
# 2. Import modules 
# ------------------
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi pyrf96

######################################################################
#

import math
import numpy as np
import matplotlib.pyplot as plt

from cofi import BaseProblem, InversionOptions, Inversion
import pyrf96

np.random.seed(42)

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 3. Define the problem 
# ----------------------
# 
# We use a 4-layer Earth model in pyrf96’s Voronoi nuclei format
# (``mtype=0``). Each row is ``[depth_km, Vs_km/s, Vp/Vs]``. The Vp/Vs
# ratios are held fixed throughout the inversion — only depths and/or
# velocities are inverted.
# 
# We generate synthetic observed data by computing the receiver function
# for the true model and adding correlated noise.
# 

# True Earth model: 4 layers [depth_km, Vs_km/s, Vp/Vs]
good_model = np.array([
    [ 1.0, 3.0, 1.7],
    [ 8.0, 3.2, 2.0],
    [20.0, 4.0, 1.7],
    [45.0, 4.2, 1.7],
])

# Fixed Vp/Vs ratios
vpvs = good_model[:, 2]

# Generate synthetic data
t, rfunc = pyrf96.rfcalc(good_model)                                  # noise-free
t2, rfunc_noise = pyrf96.rfcalc(good_model, sn=0.5, seed=12345678)   # noisy observed
observed_data = rfunc_noise

# Inverse data covariance matrix for correlated noise
Cdinv = pyrf96.InvDataCov(2.5, 0.01, len(rfunc))

######################################################################
#

def plot_model_staircase(model, ax, max_depth=60.0, **kwargs):
    """Plot a pyrf96 Voronoi nuclei model as a Vs-depth staircase."""
    order = np.argsort(model[:, 0])
    depths = model[order, 0]
    vs = model[order, 1]
    interfaces = 0.5 * (depths[:-1] + depths[1:])
    plot_depths = np.concatenate([[0.0], np.repeat(interfaces, 2), [max_depth]])
    plot_vs = np.repeat(vs, 2)
    ax.plot(plot_vs, plot_depths, **kwargs)

######################################################################
#

# Plot the true model and observed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Velocity-depth profile
plot_model_staircase(good_model, ax1, color="red", lw=2, label="True model")
ax1.set_xlim(2.5, 5.0)
ax1.set_ylim(60, 0)
ax1.set_xlabel("Vs (km/s)")
ax1.set_ylabel("Depth (km)")
ax1.set_title("True Earth model")
ax1.legend()

# Receiver functions
ax2.plot(t2, rfunc_noise, color="k", label="Observed (noisy)")
ax2.plot(t, rfunc, color="r", ls="--", label="True (noise-free)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.set_title("Synthetic receiver function")
ax2.legend()

plt.tight_layout()

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 4. Problem 1: Invert for velocities only 
# -----------------------------------------
# 
# In this first problem we fix the layer depths at their true values and
# invert only for the 4 S-wave velocities. This is a simpler 4-dimensional
# problem.
# 

# Conversion functions for velocity-only inversion
fixed_depths = good_model[:, 0]  # [1, 8, 20, 45] — held fixed

def get_vel_params(fullmodel):
    """Extract velocity parameters from full model."""
    return fullmodel[:, 1].copy()

def make_model_vel(vel_params):
    """Reconstruct full model from velocity parameters (fixed depths and Vp/Vs)."""
    return np.column_stack([fixed_depths, vel_params, vpvs])

# Misfit function for velocity-only inversion (used by Stage I)
def misfit_vel(vel_params):
    model = make_model_vel(vel_params)
    t_pred, predicted_data = pyrf96.rfcalc(model)
    res = observed_data - predicted_data
    misfit_val = np.dot(res, np.transpose(np.dot(Cdinv, res))) / 2.0
    if math.isnan(misfit_val):
        return float("inf")
    return misfit_val

# Log posterior function (used by Stage II)
def log_posterior_vel(vel_params):
    """Log posterior = -misfit (uniform prior, Gaussian likelihood)."""
    return -misfit_vel(vel_params)

# CoFI BaseProblem
inv_problem_vel = BaseProblem()
inv_problem_vel.name = "Receiver Function - Velocity only"
inv_problem_vel.set_objective(misfit_vel)
inv_problem_vel.summary()

######################################################################
#


######################################################################
# 4.1 Stage I: Direct Search 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Stage I of the Neighbourhood Algorithm is a derivative-free ensemble
# optimiser. It minimises the misfit function defined above by iteratively
# sampling new models in the neighbourhood of the best-fitting models
# found so far. The ``neighpyI`` tool in CoFI runs only this direct search
# phase.
# 

# NA-I hyperparameters for 4-D problem
bounds_vel = [(2.0, 4.5)] * 4

inv_options_vel = InversionOptions()
inv_options_vel.set_tool("neighpyI")
inv_options_vel.set_params(
    bounds=bounds_vel,
    n_initial_samples=2000,
    n_samples_per_iteration=50,
    n_cells_to_resample=10,
    n_iterations=100,
)

inv_vel = Inversion(inv_problem_vel, inv_options_vel)
inv_result_vel = inv_vel.run()
inv_result_vel.summary()

######################################################################
#

best_vel = inv_result_vel.model
ds_samples_vel = inv_result_vel.samples
ds_objectives_vel = inv_result_vel.objectives

print("Best model (velocities):")
print(make_model_vel(best_vel))
print("\nTrue model:")
print(good_model)

######################################################################
#


######################################################################
# Convergence
# ^^^^^^^^^^^
# 

best_i_vel = np.argmin(ds_objectives_vel)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ds_objectives_vel, marker=".", linestyle="", markersize=2, color="black")
ax.scatter(best_i_vel, ds_objectives_vel[best_i_vel], c="g", s=30, zorder=10,
           label="Best model")
ax.axvline(2000, c="grey", ls="--", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("Sample number")
ax.set_ylabel("Misfit")
ax.set_title("Problem 1: NA-I convergence (velocity-only)")
ax.text(0.15, 0.95, "Initial random search", transform=ax.transAxes,
        ha="center", va="top")
ax.text(0.65, 0.95, "Neighbourhood search", transform=ax.transAxes,
        ha="center", va="top")
ax.legend()

######################################################################
#


######################################################################
# NA-I model ensemble
# ^^^^^^^^^^^^^^^^^^^
# 

fig, ax = plt.subplots(figsize=(5, 8))

# Plot initial random samples as faint ensemble
for i in range(min(2000, len(ds_samples_vel))):
    m = make_model_vel(ds_samples_vel[i])
    plot_model_staircase(m, ax, color="k", alpha=0.01, lw=0.5)

plot_model_staircase(good_model, ax, color="b", lw=2, label="True model")
plot_model_staircase(make_model_vel(best_vel), ax, color="g", lw=2, ls="--",
                     label="Best model (NA-I)")
ax.set_xlim(2.5, 5.0)
ax.set_ylim(60, 0)
ax.set_xlabel("Vs (km/s)")
ax.set_ylabel("Depth (km)")
ax.set_title("Problem 1: NA-I model ensemble")
ax.legend(loc="lower left")

######################################################################
#


######################################################################
# NA-I predicted data
# ^^^^^^^^^^^^^^^^^^^
# 

fig, ax = plt.subplots(figsize=(10, 4))

# Plot predicted RFs from initial samples
for i in range(min(2000, len(ds_samples_vel))):
    m = make_model_vel(ds_samples_vel[i])
    try:
        t_pred, rf_pred = pyrf96.rfcalc(m)
        ax.plot(t_pred, rf_pred, color="k", alpha=0.01, lw=0.5)
    except Exception:
        pass

ax.plot(t2, observed_data, color="k", lw=1.5, label="Observed", zorder=10)
t_best, rf_best = pyrf96.rfcalc(make_model_vel(best_vel))
ax.plot(t_best, rf_best, color="g", ls="--", lw=1.5, label="Best model (NA-I)",
        zorder=11)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Problem 1: Predicted receiver functions")
ax.legend()

######################################################################
#


######################################################################
# 4.2 Stage II: Appraisal 
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Stage II of the NA resamples the Voronoi cells constructed by Stage I
# according to a **log posterior probability density** (log-PPD).
# Crucially, this stage requires no additional forward model evaluations —
# it only uses the geometry of the Stage I ensemble and the log-PPD values
# assigned to each sample.
# 
# The user must provide ``log_ppd`` values for each sample in the Stage I
# ensemble. The relationship between the log-PPD and the misfit is a
# **modelling choice**:
# 
# .. math:: \log p(\mathbf{m}|\mathbf{d}) = -\Phi(\mathbf{m},\mathbf{d})
# 
# corresponds to a Gaussian likelihood with a uniform prior. More
# generally, one could incorporate a non-uniform prior or a different
# likelihood function, which would change the ``log_ppd`` values without
# requiring Stage I to be re-run.
# 

# Compute log-PPD for each Stage I sample.
# Since log_posterior_vel(s) = -misfit_vel(s), this is equivalent to:
#   log_ppd_vel = np.array([log_posterior_vel(s) for s in ds_samples_vel])
# but using the already-computed misfit values avoids redundant forward evaluations.
log_ppd_vel = -ds_objectives_vel

######################################################################
#

# NA-II hyperparameters
inv_options_vel_app = InversionOptions()
inv_options_vel_app.set_tool("neighpyII")
inv_options_vel_app.set_params(
    bounds=bounds_vel,
    initial_ensemble=ds_samples_vel,
    log_ppd=log_ppd_vel,
    n_resample=20000,
    n_walkers=10,
)

inv_vel_app = Inversion(inv_problem_vel, inv_options_vel_app)
inv_result_vel_app = inv_vel_app.run()
inv_result_vel_app.summary()

######################################################################
#

appraisal_samples_vel = inv_result_vel_app.new_samples
appraisal_mean_vel = appraisal_samples_vel.mean(axis=0)

fig, ax = plt.subplots(figsize=(5, 8))
for i in range(0, len(appraisal_samples_vel), 10):
    m = make_model_vel(appraisal_samples_vel[i])
    plot_model_staircase(m, ax, color="k", alpha=0.01, lw=0.5)

plot_model_staircase(good_model, ax, color="b", lw=2, label="True model")
plot_model_staircase(make_model_vel(best_vel), ax, color="g", lw=2, ls="--",
                     label="Best model (NA-I)")
plot_model_staircase(make_model_vel(appraisal_mean_vel), ax, color="r", lw=2,
                     ls="--", label="Mean model (NA-II)")
ax.set_xlim(2.5, 5.0)
ax.set_ylim(60, 0)
ax.set_xlabel("Vs (km/s)")
ax.set_ylabel("Depth (km)")
ax.set_title("Problem 1: NA-II appraisal ensemble")
ax.legend(loc="lower left")

######################################################################
#

# 4x4 corner plot for velocity-only problem
true_vel = get_vel_params(good_model)
n_params_vel = 4
var_names_vel = [f"Vs{i+1}" for i in range(n_params_vel)]

fig, axes = plt.subplots(n_params_vel, n_params_vel, figsize=(8, 8),
                         tight_layout=True,
                         gridspec_kw={"hspace": 0, "wspace": 0})
for i in range(n_params_vel):
    for j in range(n_params_vel):
        axes[i, j].set_xlim(bounds_vel[j])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if i == j:
            axes[i, j].hist(appraisal_samples_vel[:, j], bins=100,
                            histtype="step", color="k")
            axes[i, j].axvline(best_vel[j], c="g", ls="--")
            axes[i, j].axvline(true_vel[j], c="b", ls="--")
            axes[i, j].axvline(appraisal_mean_vel[j], c="r", ls="--")
        elif j < i:
            axes[i, j].scatter(appraisal_samples_vel[:, j],
                               appraisal_samples_vel[:, i],
                               s=1, c="k", alpha=0.01)
            axes[i, j].scatter(best_vel[j], best_vel[i], c="g", s=10, zorder=10)
            axes[i, j].scatter(true_vel[j], true_vel[i], c="b", s=10, zorder=10)
            axes[i, j].scatter(appraisal_mean_vel[j], appraisal_mean_vel[i],
                               c="r", s=10, zorder=10)
            axes[i, j].set_ylim(bounds_vel[i])
        else:
            axes[i, j].axis("off")
        if i == n_params_vel - 1 and j <= i:
            axes[i, j].set_xlabel(var_names_vel[j])
            axes[i, j].set_xticks(np.linspace(*bounds_vel[j], 3))
        if j == 0 and i > 0:
            axes[i, j].set_ylabel(var_names_vel[i])

fig.suptitle("Problem 1: Posterior corner plot (velocity-only)")

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 5. Problem 2: Invert for depths and velocities 
# -----------------------------------------------
# 
# Now we invert for both layer depths and S-wave velocities
# simultaneously. This is a harder 8-dimensional problem with interleaved
# parameters: ``[d1, vs1, d2, vs2, d3, vs3, d4, vs4]``.
# 

# Conversion functions for depth+velocity inversion
def get_inversion_parameters(fullmodel):
    """Extract [depth, Vs] pairs as flat 8-vector."""
    return fullmodel[:, :2].flatten()

def get_model_parameters(invmodel):
    """Reconstruct full [4,3] model from 8-vector, restoring fixed Vp/Vs."""
    return np.append(invmodel.reshape(len(vpvs), -1), vpvs[:, None], axis=1)

# Misfit function for depth+velocity inversion (used by Stage I)
def misfit_dv(imodel):
    model = get_model_parameters(imodel)
    t_pred, predicted_data = pyrf96.rfcalc(model)
    res = observed_data - predicted_data
    misfit_val = np.dot(res, np.transpose(np.dot(Cdinv, res))) / 2.0
    if math.isnan(misfit_val):
        return float("inf")
    return misfit_val

# Log posterior function (used by Stage II)
def log_posterior_dv(imodel):
    """Log posterior = -misfit (uniform prior, Gaussian likelihood)."""
    return -misfit_dv(imodel)

# CoFI BaseProblem
inv_problem_dv = BaseProblem()
inv_problem_dv.name = "Receiver Function - Depth + Velocity"
inv_problem_dv.set_objective(misfit_dv)
inv_problem_dv.summary()

######################################################################
#


######################################################################
# 5.1 Stage I: Direct Search 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# NA-I hyperparameters for 8-D problem (larger than Problem 1)
bounds_dv = [(0.0, 60.0), (2.0, 4.5)] * 4

inv_options_dv = InversionOptions()
inv_options_dv.set_tool("neighpyI")
inv_options_dv.set_params(
    bounds=bounds_dv,
    n_initial_samples=2000,
    n_samples_per_iteration=100,
    n_cells_to_resample=20,
    n_iterations=50,
)

inv_dv = Inversion(inv_problem_dv, inv_options_dv)
inv_result_dv = inv_dv.run()
inv_result_dv.summary()

######################################################################
#

best_dv = inv_result_dv.model
ds_samples_dv = inv_result_dv.samples
ds_objectives_dv = inv_result_dv.objectives

print("Best model (depth + velocity):")
print(get_model_parameters(best_dv))
print("\nTrue model:")
print(good_model)

######################################################################
#


######################################################################
# Convergence
# ^^^^^^^^^^^
# 

best_i_dv = np.argmin(ds_objectives_dv)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ds_objectives_dv, marker=".", linestyle="", markersize=2, color="black")
ax.scatter(best_i_dv, ds_objectives_dv[best_i_dv], c="g", s=30, zorder=10,
           label="Best model")
ax.axvline(2000, c="grey", ls="--", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("Sample number")
ax.set_ylabel("Misfit")
ax.set_title("Problem 2: NA-I convergence (depth + velocity)")
ax.text(0.15, 0.95, "Initial random search", transform=ax.transAxes,
        ha="center", va="top")
ax.text(0.65, 0.95, "Neighbourhood search", transform=ax.transAxes,
        ha="center", va="top")
ax.legend()

######################################################################
#


######################################################################
# 5.2 Stage II: Appraisal 
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 

# Compute log-PPD for each Stage I sample (same choice as Problem 1)
log_ppd_dv = -ds_objectives_dv

######################################################################
#

# NA-II hyperparameters
inv_options_dv_app = InversionOptions()
inv_options_dv_app.set_tool("neighpyII")
inv_options_dv_app.set_params(
    bounds=bounds_dv,
    initial_ensemble=ds_samples_dv,
    log_ppd=log_ppd_dv,
    n_resample=20000,
    n_walkers=10,
)

inv_dv_app = Inversion(inv_problem_dv, inv_options_dv_app)
inv_result_dv_app = inv_dv_app.run()
inv_result_dv_app.summary()

######################################################################
#

appraisal_samples_dv = inv_result_dv_app.new_samples
appraisal_mean_dv = appraisal_samples_dv.mean(axis=0)

fig, ax = plt.subplots(figsize=(5, 8))
for i in range(0, len(appraisal_samples_dv), 10):
    m = get_model_parameters(appraisal_samples_dv[i])
    plot_model_staircase(m, ax, color="k", alpha=0.01, lw=0.5)

plot_model_staircase(good_model, ax, color="b", lw=2, label="True model")
plot_model_staircase(get_model_parameters(best_dv), ax, color="g", lw=2,
                     ls="--", label="Best model (NA-I)")
plot_model_staircase(get_model_parameters(appraisal_mean_dv), ax, color="r",
                     lw=2, ls="--", label="Mean model (NA-II)")
ax.set_xlim(2.5, 5.0)
ax.set_ylim(60, 0)
ax.set_xlabel("Vs (km/s)")
ax.set_ylabel("Depth (km)")
ax.set_title("Problem 2: NA-II appraisal ensemble")
ax.legend(loc="lower left")

######################################################################
#


######################################################################
# Corner plot
# ^^^^^^^^^^^
# 

# 8x8 corner plot for depth+velocity problem
true_dv = get_inversion_parameters(good_model)
n_params_dv = 8
var_names_dv = ["d1", "Vs1", "d2", "Vs2", "d3", "Vs3", "d4", "Vs4"]

fig, axes = plt.subplots(n_params_dv, n_params_dv, figsize=(12, 12),
                         tight_layout=True,
                         gridspec_kw={"hspace": 0, "wspace": 0})
for i in range(n_params_dv):
    for j in range(n_params_dv):
        axes[i, j].set_xlim(bounds_dv[j])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if i == j:
            axes[i, j].hist(appraisal_samples_dv[::10, j], bins=100,
                            histtype="step", color="k")
            axes[i, j].axvline(best_dv[j], c="g", ls="--")
            axes[i, j].axvline(true_dv[j], c="b", ls="--")
            axes[i, j].axvline(appraisal_mean_dv[j], c="r", ls="--")
        elif j < i:
            axes[i, j].scatter(appraisal_samples_dv[::10, j],
                               appraisal_samples_dv[::10, i],
                               s=1, c="k", alpha=0.01)
            axes[i, j].scatter(best_dv[j], best_dv[i], c="g", s=10, zorder=10)
            axes[i, j].scatter(true_dv[j], true_dv[i], c="b", s=10, zorder=10)
            axes[i, j].scatter(appraisal_mean_dv[j], appraisal_mean_dv[i],
                               c="r", s=10, zorder=10)
            axes[i, j].set_ylim(bounds_dv[i])
        else:
            axes[i, j].axis("off")

for ax_, xlabel in zip(axes[-1, :], var_names_dv):
    ax_.set_xlabel(xlabel)
for ax_, ylabel in zip(axes[:, 0], var_names_dv):
    if ylabel != var_names_dv[0]:
        ax_.set_ylabel(ylabel)

fig.suptitle("Problem 2: Posterior corner plot (depth + velocity)")

######################################################################
#


######################################################################
# Predicted data ensemble
# ^^^^^^^^^^^^^^^^^^^^^^^
# 

fig, ax = plt.subplots(figsize=(10, 4))

for i in range(0, len(appraisal_samples_dv), 15):
    m = get_model_parameters(appraisal_samples_dv[i])
    try:
        t_pred, rf_pred = pyrf96.rfcalc(m)
        ax.plot(t_pred, rf_pred, color="k", alpha=0.01, lw=0.5)
    except Exception:
        pass

ax.plot(t2, observed_data, color="k", lw=1.5, label="Observed", zorder=10)
t_best, rf_best = pyrf96.rfcalc(get_model_parameters(best_dv))
ax.plot(t_best, rf_best, color="g", ls="--", lw=1.5, label="Best model (NA-I)",
        zorder=11)
t_mean, rf_mean = pyrf96.rfcalc(get_model_parameters(appraisal_mean_dv))
ax.plot(t_mean, rf_mean, color="r", ls="--", lw=1.5, label="Mean model (NA-II)",
        zorder=11)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Problem 2: Predicted receiver functions from NA-II ensemble")
ax.legend()

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# Watermark
# ---------
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#



######################################################################
#
# sphinx_gallery_thumbnail_number = -1