"""
Finding Neptune using Uranus
============================

"""


######################################################################
# Note : - For better understanding of this notebook, refer to the
# `md_file <https://github.com/inlab-geo/cofi-examples/blob/main/theory/finding_neptune_bayesian.md>`__
# designed specifically for this notebook and better insights into the
# theory.
# 
# -  The import methods and functions from
#    `neptune_bayesian_methods <https://github.com/inlab-geo/cofi-examples/blob/main/theory/finding_neptune_deterministic.md>`__,
#    `neptune_deterministic_methods <https://github.com/inlab-geo/cofi-examples/blob/main/theory/finding_neptune_deterministic.md>`__
#    and
#    `setup_inversion <https://github.com/inlab-geo/cofi-examples/blob/main/theory/finding_neptune_deterministic.md>`__
#    are used to set up the simulation and perform the necessary
#    calculations.
# 

# This notebook requires the following libraries to run, in order to install them uncomment the lines below
# %pip install cofi
# %pip install numba
# %pip install tqdm
# %pip install matplotlib
# %pip install astroquery


######################################################################
#


######################################################################
# 1. Introduction
# ---------------
# 


######################################################################
# -  The following Notebook is based on the historical problem on how
#    Neptune was found by Johann Galle using mathematical predictions made
#    independently by two astronomers:
# 
#    -  Urbain Le Verrier (France)
# 
#    -  John Couch Adams (England)
# 
#    Through this Notebook we wish to demostrate how ``CoFI`` can be used
#    to solve this problem via deterministic inversion. For more details
#    on this problem, see the following
#    `thesis <www.diva-portal.org/smash/get/diva2:1218549/FULLTEXT01.pdf>`__
# 
# -  In the following notebook we discuss the problem of finding Neptune’s
#    mass, its velocity components and its position coordinates in the
#    year 1775, by modeling the trajectory of Uranus with and without the
#    influence of Neptune.
# 
# -  We define $ g(m) $, our forward model, as vector-valued function that
#    predicts the position coordinates of Uranus at each observation time
#    :math:`t_j`, as a function of Neptune’s parameters :math:`m`:
# 
#    | 
# 
#      .. math::
# 
# 
#            g(m) =
#           \begin{bmatrix}
#           \hat x_1(m) \\
#           \vdots \\
#           \hat x_N(m) \\
#           \hat y_1(m) \\
#           \vdots \\
#           \hat y_N(m) \\
#           \hat z_1(m) \\
#           \vdots \\
#           \hat z_N(m)
#           \end{bmatrix}
#           \in \mathbb{R}^{3M \times 1}
#           
# 
#      where $ N $ is the number of data points, and $
#      :raw-latex:`\hat `x_j(m),  :raw-latex:`\hat `y_j(m),
#       :raw-latex:`\hat `z_j(m) $ are the coordinates of Uranus at data
#      point $ j  =  1,  2,  ….  N$ as a function of Neptune’s parameters
#      $ m $,
#    | where :math:`m = (m_M, m_x, m_y, m_z, {m_{v_x}}, m_{v_y}, m_{v_z})`
#      is the set of parameters describing Neptune’s mass (:math:`m_M`),
#      its position coordinates :math:`(m_x, m_y, m_z)` and its velocity
#      components :math:`(m_{v_x}, m_{v_y}, m_{v_z})`
# 
#    | and :math:`d` as the data vector of positions of Uranus at
#      different time steps:
#    | 
# 
#      .. math::
# 
# 
#           d =
#           \begin{bmatrix}
#           x_1 \\
#           \vdots \\
#           x_N \\
#           y_1 \\
#           \vdots \\
#           y_N \\
#           z_1 \\
#           \vdots \\
#           z_N
#           \end{bmatrix}
#           \in \mathbb{R}^{3M \times 1}
#           
# 
#      where $ N $ is the number of data points, and $
#      :raw-latex:`\hat `x_j,  :raw-latex:`\hat `y_j,
#       :raw-latex:`\hat `z_j $ are the true coordinates of Uranus at data
#      point $ j  =  1,  2,  ….  N$.
# 
# -  hence our problem formulation changes to :
# 
#    .. math::
# 
# 
#        \underset{m}{\min}   || g(m) - {d} ||_{2}^2 
#         
# 


######################################################################
# 2. Problem Setting
# ------------------
# 


######################################################################
# -  We use **Newton’s Law of gravitation** to compute the gravitational
#    force acting on a planet due to other celestial bodies.
# 
# -  This formulation uses Newton’s Law of Universal Gravitation to model
#    the **net gravitational influence** from multiple bodies on a single
#    target planet.
# 


######################################################################
# -  Let :math:`r` be the vector containing the positions of all planets
#    and :math:`a` be the vector containing the accelerations of all
#    planets.
# 
#    The velocity and acceleration vectors are defined as:
# 
#    .. math::
# 
# 
#         \dot{\mathbf{r}} = \mathbf{v}, \quad \ddot{\mathbf{r}} = \mathbf{a}
#         
# 
#    where
# 
#    .. math::
# 
# 
#         \mathbf{r} = 
#         \begin{bmatrix}
#         \mathbf{r}_1 \\
#         \vdots \\
#         \mathbf{r}_9
#         \end{bmatrix}, \quad
#         \mathbf{a} = 
#         \begin{bmatrix}
#         \mathbf{a}_1 \\
#         \vdots \\
#         \mathbf{a}_9
#         \end{bmatrix}, \quad
#         \mathbf{r}, \mathbf{a} \in \mathbb{R}^{27 \times 1}
#         
# 
# -  This results in the system of differential equations:
# 
#    .. math::
# 
# 
#         \frac{d}{dt}
#         \begin{bmatrix}
#         \mathbf{r}(t) \\
#         \mathbf{v}(t)
#         \end{bmatrix}
#         = 
#         \begin{bmatrix}
#         \mathbf{v}(t) \\
#         \mathbf{a}(t)
#         \end{bmatrix}
#         
# 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
import copy
from numba import njit, jit
import warnings
import multiprocessing as mp
import time
import arviz
from astroquery.jplhorizons import Horizons
warnings.filterwarnings('ignore')

from cofi import BaseProblem, InversionOptions, Inversion
from neptune_mcmc_helpers import (
    analysis_samples,
    load_mcmc_results,
    plot_pair_kde_legacy_style,
    run_cofi_emcee,
    safe_autocorr_time,
    save_mcmc_results,
    scaled_reference_values,
    selected_trajectory_samples,
)

np.random.seed(42)


######################################################################
#

# Long-run control
# Keep RUN_LONG_MCMC = False for quick validation. Set it to True when you want
# to regenerate converged MCMC outputs and save them beside this notebook.
from pathlib import Path

RUN_LONG_MCMC = True
USE_SAVED_LONG_RUN_RESULTS = False
SAVE_LONG_RUN_RESULTS = False

VALIDATION_STEPS = 40
SYNTHETIC_LONG_STEPS = 10_000
REAL_LONG_STEPS = 50_000
LONG_RUN_RESULTS_FILE = Path("neptune_long_run_results.npz")


######################################################################
#


######################################################################
# -  We solve our ODEs with the **Runge-Kutta 4 (RK4)** method, which is
#    an explicit and iterative method, well-suited for initial value
#    problems.
# 


######################################################################
# -  In the following cell, we import - ``acceleration`` and ``rk4_step``
#    to serve as functions for our forward model.
# 


######################################################################
# -  We now demostrate our forward model, using the above defined
#    functions, in the ``run_simulation`` method defined below, which
#    helps us run a simulation of solar system.
# 
# -  Throughout this notebook for the purpose of our inversion, we are
#    going to define mass in terms of solar masses, positions coordinates
#    in **Astonomical Units (AU)** and velocities for planets in
#    **Au/day**
# 

from neptune_deterministic_methods import acceleration, rk4_step, run_simulation
trajectories = run_simulation(T = 100, dt = 1, plot_only=['Uranus', 'Neptune', 'Saturn', 'Jupiter', 'Mars', 'Earth', 'Venus', 'Mercury'])


######################################################################
#


######################################################################
# 2.1 Bayesian Inversion via the Stretch Move
# -------------------------------------------
# 
# In this notebook, we use ``CoFI`` to perform Bayesian inversion.
# Specifically, we use the **stretch move** ensemble sampler introduced by
# goodman & Weare (2010), which is well-suited for exploring complex,
# high-dimensional posterior distributions.
# 
# MCMC is a method to sample from the **posterior distribution** of model
# parameters :math:`\mathbf{m}` given observed data :math:`\mathbf{d}`.
# According to Bayes’ theorem:
# 
# .. math::
# 
# 
#    \tag{1}
#    p(\mathbf{m} \mid \mathbf{d}) \propto p(\mathbf{d} \mid \mathbf{m}) \, p(\mathbf{m}),
# 
# where:
# 
# -  :math:`p(\mathbf{m} \mid \mathbf{d})` is the **posterior**
#    probability of the model,
# -  :math:`p(\mathbf{d} \mid \mathbf{m})` is the **likelihood**, and
# -  :math:`p(\mathbf{m})` is the **prior**.
# 
# The likelihood measures how well a model explains the data.
# 
# In the stretch move algorithm, a population of walkers explores the
# posterior distribution simultaneously. Each walker proposes new
# positions based on the positions of other walkers, allowing for more
# efficient exploration of complex posteriors.
# 
# The sampling proceeds in the following steps:
# 
# 1. Initialize a set of walkers in parameter space.
# 2. At each iteration, propose new positions for each walker using the
#    stretch move.
# 3. Accept or reject the proposed positions based on the Metropolis
#    criterion.
# 4. After a **burn-in** period, the samples from all walkers are
#    collected to approximate the posterior.
# 
# This method yields not only the most likely model but also the full
# distribution over model space, capturing uncertainty in the inferred
# parameters.
# 


######################################################################
# 3. Inversion on Synthetic Data
# ------------------------------
# 
# -  We will first demonstrate bayesian inversion using ``CoFI`` on
#    synthetic data.
# 
# -  The synthetic observations are generated by integrating our
#    gravitational forward model with a fourth-order Runge-Kutta (``RK4``)
#    solver to simulate Uranus’s trajectory under the influence of
#    Neptune.
# 


######################################################################
# -  We simulate observational noise by sampling from zero-mean Gaussian
#    distributions with specified variances for each coordinate:
# 
# .. math::
# 
# 
#    x_\text{obs} = x + \epsilon_x, \quad y_\text{obs} = y + \epsilon_y, \quad z_\text{obs} = z + \epsilon_z
# 
# -  where
# 
# -  
# 
#    .. math::
# 
# 
#         \epsilon_x \sim \mathcal{N}(0, \sigma_x^2), \quad 
#         \epsilon_y \sim \mathcal{N}(0, \sigma_y^2), \quad 
#         \epsilon_z \sim \mathcal{N}(0, \sigma_z^2)
# 
# -  with noise levels set as
# 
# -  
# 
#    .. math::
# 
# 
#       \sigma_x = \sigma_y = 10^{-3}, \quad \sigma_z = 10^{-5}
# 


######################################################################
# -  The function below generates the synthetic data with the specified
#    noise levels.
# 


######################################################################
# 3.1 Generating synthetic data
# -----------------------------
# 

from neptune_deterministic_methods import generate_synthetic_data

T = 190 # time for which we want to generate synthetic data
z_scale_factor = 1
dt = 1

U_true = generate_synthetic_data(T = T, 
                                 dt = dt, 
                                 z_scaling = False, 
                                 add_noise = True, 
                                 noise_level = np.array([0.001, 0.001, 0.00001]))


######################################################################
#


######################################################################
# -  The cell below sets up the starting model for our walkers/chains in
#    MCMC and some pre-defined scales to be used for scaling while running
#    our inversion, to ensure that all parameters are roughly on the same
#    scale or at least near to each other.
# 
# -  For our bayesian inversion we first scale all the parameters and the
#    convert our mass to log mass. This ensures that all our parameters
#    are in the same scale.
# 

# True/reference parameters for Neptune [mass, x, y, z, vx, vy, vz]
from setup_inversion import get_inversion_indices, set_true_m, get_param_bounds, get_param_scales, set_initial_conditions, get_starting_points, PARAM_NAMES     

m_0 = set_true_m()
initial_conditions = set_initial_conditions()

PARAM_BOUNDS = get_param_bounds()
PARAM_SCALES = get_param_scales()
INVERT_INDICES = get_inversion_indices()
STARTING_POINTS = get_starting_points()


######################################################################
#

from setup_inversion import scale_param, unscale_param, validate_config

validate_config()


######################################################################
#

names = list(initial_conditions.keys())
n_bodies = len(names)
uranus_idx = names.index("Uranus")


######################################################################
#

from neptune_bayesian_methods import predict_U


######################################################################
#

m_start_scaled = scale_param(m_0)

if 0 in INVERT_INDICES:
    m_start_scaled[0] = np.log10(m_start_scaled[0])  # Convert mass to log scale

if len(INVERT_INDICES) == 1:
    m_start_scaled = m_start_scaled.item() if hasattr(m_start_scaled, 'item') else m_start_scaled

print(f"\nStarting points (unscaled): {STARTING_POINTS}")
print(f"Starting points (scaled): {m_start_scaled}")

print("\nTesting forward function...")
try:
    pred_test = predict_U(m_start_scaled, T = T, dt = dt, z_scale_factor = z_scale_factor)
    residual_test = pred_test - U_true
    print(f"Initial residual norm: {np.linalg.norm(residual_test):.6f}")
    print(f"Residual by component:")
    print(f"  X component: {np.linalg.norm(residual_test[:T]):.6f}")
    print(f"  Y component: {np.linalg.norm(residual_test[T:2*T]):.6f}")
    print(f"  Z component: {np.linalg.norm(residual_test[2*T:]):.6f}")

    # Match the archived notebook's pre-MCMC state exactly. In the archive,
    # predict_U/build_neptune_vector mutated m_start_scaled[0] in place from
    # log10(scaled_mass) back to scaled_mass during this forward-function test.
    # The current predictor copies inputs, so we reproduce that side effect
    # explicitly before constructing the emcee walkers.
    if 0 in INVERT_INDICES:
        m_start_scaled[0] = 10**m_start_scaled[0]
    print(f"Archive-compatible m_start_scaled before MCMC: {m_start_scaled}")
except Exception as e:
    print(f"Forward function test failed: {e}")
    import traceback
    traceback.print_exc()


######################################################################
#


######################################################################
# 3.2 Running the Inversion on Synthetic Data
# -------------------------------------------
# 
# The Bayesian setup below keeps the original likelihood and prior
# formulation: a bounded uniform prior on the scaled parameters and a
# diagonal Gaussian data-error model, implemented as
# ``-0.5 * sum((d - g(m))**2 * inv_variances)``. The notebook defaults to
# a short validation run. Set ``RUN_LONG_MCMC = True`` in the
# configuration cell to run the original 10,000-step synthetic-data chain
# and save converged upload outputs into ``neptune_long_run_results.npz``.
# 


######################################################################
# -  Defined below are our ``log prior`` and ``log likelihood`` functions.
# 
# -  We set a uniform prior on log of our mass and all the other
#    parameters (Neptune’s position coordinates and velocity components).
# 

param_bounds_lower = np.array([bound[0] for bound in PARAM_BOUNDS])
param_bounds_upper = np.array([bound[1] for bound in PARAM_BOUNDS])

bounds_lower_scaled = scale_param(param_bounds_lower)
bounds_upper_scaled = scale_param(param_bounds_upper)

bounds_lower_scaled[0] = np.log10(bounds_lower_scaled[0])  # Convert mass back to log scale
bounds_upper_scaled[0] = np.log10(bounds_upper_scaled[0])  # Convert mass back to log scale

sigma_x = 8e-3          # Standard deviation for X position in uranus orbit data, taken one order higher than the noise added
sigma_y = 8e-3          # Standard deviation for Y position in uranus orbit data, taken one order higher than the noise added
sigma_z = 1e-4           # Standard deviation for Z position in uranus orbit data, taken two orders higher than the noise added

diagonal_values = [1/sigma_x**2]*T + [1/sigma_y**2]*T + [1/sigma_z**2]*T
inv_variances = np.array(diagonal_values)


######################################################################
#

def my_log_likelihood(m_scaled : np.ndarray, U_true : np.ndarray) -> float:
    
    """
    Log-likelihood function for Bayesian inversion

    Parameters
    ----------
    m_scaled : np.array
        Scaled parameters being inverted.
    U_true : np.array
        True Uranus positions in the format [x, y, z] scaled by z_scale_factor.
    Returns
    -------
    log_likelihood : float
        The log-likelihood value, or -inf if the prediction fails or residuals are not finite.
    """
    m_scaled = np.atleast_1d(m_scaled).astype(np.float64)

    try:
        y_hat = predict_U(m_scaled, T = T, dt = dt, z_scale_factor = z_scale_factor)
        if not np.isfinite(y_hat).all():
            return -np.inf
        residuals = U_true - y_hat
        if residuals.shape != U_true.shape:
            return -np.inf
        return -0.5 * np.sum(residuals**2 * inv_variances)
    
    except (ValueError, ArithmeticError):
        return -np.inf
    except Exception as e:
        print(f"Error in log-likelihood calculation: {e}")
        return -np.inf

def my_log_prior(m_scaled : np.ndarray) -> float:
    """
    Log-prior function for Bayesian inversion

    Parameters
    ----------
    m_scaled : np.ndarray
        Scaled parameters being inverted.

    Returns
    -------
    float
        The log prior value, or -inf if the parameters are out of bounds.
    """
    m_scaled = np.atleast_1d(m_scaled)
    if np.any(m_scaled < bounds_lower_scaled) or np.any(m_scaled > bounds_upper_scaled):
        return -np.inf
    return 0


######################################################################
#

n_walkers = 20
n_dim = len(INVERT_INDICES)
nsteps = SYNTHETIC_LONG_STEPS if RUN_LONG_MCMC else VALIDATION_STEPS
use_pool = True
show_progress = RUN_LONG_MCMC
run_mode = "long" if RUN_LONG_MCMC else "validation"

print(f"Optimized setup: {len(inv_variances)} inverse variance values")
print(f"Running {run_mode} MCMC chain: {nsteps} steps, {n_walkers} walkers")
print("Set RUN_LONG_MCMC = True in the configuration cell to run the original 10,000-step synthetic chain.")

print("Parameter bounds (unscaled):")
print("  Lower:", param_bounds_lower)
print("  Upper:", param_bounds_upper)
print("Scaled bounds:")
print("  Lower:", bounds_lower_scaled)
print("  Upper:", bounds_upper_scaled)

print(f"Number of dimensions: {n_dim}")

# Reset the MCMC starting model to the scaled/log parameterization.
# The archive-compatibility forward test above intentionally mutates
# m_start_scaled[0] from log10(scaled_mass) back to scaled_mass. That
# state is useful for apples-to-apples archive checks, but it is wrong
# for tight walker initialization because it clips mass to the upper bound.
m_start_scaled = scale_param(m_0).astype(float)
if 0 in INVERT_INDICES:
    m_start_scaled[0] = np.log10(m_start_scaled[0])
print("MCMC start (scaled/log):", m_start_scaled)

# starting_lower_bounds = m_start_scaled - 10
# starting_upper_bounds = m_start_scaled + 10

# walkers_start = np.random.uniform(
#     starting_lower_bounds,
#     starting_upper_bounds,
#     size=(n_walkers, n_dim)
# )

rng = np.random.default_rng(42)

walker_scales = 0.075 * (bounds_upper_scaled - bounds_lower_scaled)

# Keep mass a bit tighter if desired
walker_scales[0] = 0.075

walkers_start = m_start_scaled + rng.normal(
    scale=walker_scales,
    size=(n_walkers, n_dim),
)

walkers_start = np.clip(walkers_start, bounds_lower_scaled, bounds_upper_scaled)

print("Initial state shape:", walkers_start.shape)
print(
    "Initial walker mass min/max/std:",
    f"{walkers_start[:, 0].min():.6g}",
    f"{walkers_start[:, 0].max():.6g}",
    f"{walkers_start[:, 0].std():.6g}",
)

for i, w in enumerate(walkers_start):
    ll = my_log_likelihood(w, U_true)
    if not np.isfinite(ll):
        print(f"Invalid log-likelihood at walker {i}: {w}, ll = {ll}")
        
for i, w in enumerate(walkers_start):
    walkers_start[i] = np.clip(w, bounds_lower_scaled, bounds_upper_scaled)
    if not np.array_equal(w, walkers_start[i]):
        print(f"Walker {i} clipped from {w} to {walkers_start[i]}")

print("All walkers are now within bounds.")

for i, w in enumerate(walkers_start):
    if not np.isfinite(my_log_prior(w)):
        print(f"ERROR: Walker {i} still outside prior bounds: {w}")
    else:
        ll = my_log_likelihood(w, U_true)
        if not np.isfinite(ll):
            print(f"WARNING: Walker {i} has invalid log-likelihood: {ll}")
            
inv_problem = BaseProblem()
inv_problem.name = "Neptune Orbit Determination - Config Driven"
inv_problem.set_data(U_true)
inv_problem.set_forward(predict_U)
inv_problem.set_initial_model(np.atleast_1d(m_start_scaled))
inv_problem.set_log_prior(my_log_prior)
inv_problem.set_log_likelihood(my_log_likelihood, args=(U_true,))
inv_problem.set_model_shape(n_dim)

import emcee
stretch_move = emcee.moves.StretchMove(a=1.2)

inv_result, elapsed = run_cofi_emcee(
    inv_problem,
    walkers_start,
    n_walkers,
    nsteps,
    InversionOptions,
    Inversion,
    progress=show_progress,
    use_pool=use_pool,
    n_threads=10,
    moves=stretch_move,
    skip_initial_state_check=True,
)

print(f"OPTIMIZED Inversion completed in {elapsed:.2f} seconds using `emcee`.")
print("The inversion result from `emcee`:")
inv_result.summary()


######################################################################
#


######################################################################
# 3.3 Getting the acceptance rates and auto-correlation times
# -----------------------------------------------------------
# 

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

param_names = PARAM_NAMES
param_labels = ["m" if idx == 0 else PARAM_NAMES[idx] for idx in INVERT_INDICES]

sampler = inv_result.sampler
reference_values_scaled = scaled_reference_values(scale_param, m_0)

n_trajectory_samples = 10_000
discard_burn = 0
thin_interval = 10

saved_result = None
if (not RUN_LONG_MCMC) and USE_SAVED_LONG_RUN_RESULTS:
    saved_result = load_mcmc_results(LONG_RUN_RESULTS_FILE, "synthetic")

if saved_result is not None:
    analysis_chain = saved_result["chain"]
    flat_samples = analysis_chain[discard_burn::thin_interval].reshape(-1, analysis_chain.shape[-1])
    acceptance_rates = saved_result["acceptance_fraction"]
    autocorr_times = saved_result["autocorr_times"]
    sample_source = f"saved long-run result ({LONG_RUN_RESULTS_FILE})"
    print(f"Loaded synthetic long-run results from {LONG_RUN_RESULTS_FILE}")
else:
    analysis_chain, flat_samples, sample_source = analysis_samples(
        sampler,
        reference_values_scaled,
        bounds_lower_scaled,
        bounds_upper_scaled,
        thin=thin_interval,
        burn=discard_burn,
        seed=42,
    )
    acceptance_rates = sampler.acceptance_fraction
    autocorr_times = safe_autocorr_time(sampler, fallback_chain=analysis_chain, discard=300)

selected_samples = selected_trajectory_samples(flat_samples, n_trajectory_samples, seed=42)

mean_acceptance = np.mean(acceptance_rates)
max_autocorr = np.max(autocorr_times)
mean_autocorr = np.mean(autocorr_times)

if RUN_LONG_MCMC and SAVE_LONG_RUN_RESULTS:
    save_mcmc_results(
        LONG_RUN_RESULTS_FILE,
        "synthetic",
        analysis_chain,
        flat_samples,
        acceptance_rates,
        autocorr_times,
        nsteps,
        n_walkers,
        param_labels,
    )
    print(f"Saved synthetic long-run results to {LONG_RUN_RESULTS_FILE}")

print(f"Posterior samples for plots: {sample_source} ({len(flat_samples):,} samples)")
print(f"Mean acceptance rate: {mean_acceptance:.3f}")
print("Acceptance rate per walker:")
for i, acc in enumerate(acceptance_rates):
    print(f"  Walker {i:2d}: {acc:.3f}")

print("\nAcceptance rate statistics:")
print(f"  Range: {np.min(acceptance_rates):.3f} - {np.max(acceptance_rates):.3f}")
print(f"  Std dev: {np.std(acceptance_rates):.3f}")

print("Autocorrelation times by parameter:")
for i, (label, tau) in enumerate(zip(param_labels, autocorr_times)):
    print(f"  {label:>3s}: {tau:6.1f} steps")

print("\nAutocorrelation statistics:")
print(f"  Mean τ: {mean_autocorr:.1f} steps")
print(f"  Max τ:  {max_autocorr:.1f} steps")


######################################################################
#


######################################################################
# 3.4 Plotting the Posteriors
# ---------------------------
# 


######################################################################
# -  We now plot the posteriors and along with that, we alsomark the true
#    parameter values of Neptune using a black ‘X’ mark.
# 
# -  Note that all our parameters are scaled.
# 

reference_values_scaled = scaled_reference_values(scale_param, m_0)
param_labels = ["m" if idx == 0 else PARAM_NAMES[idx] for idx in INVERT_INDICES]
ref_vals = [reference_values_scaled[i] for i in range(n_dim)]

print("Creating corner plot with KDE contours and reference values...")

fig = plot_pair_kde_legacy_style(
    flat_samples,
    param_labels,
    ref_vals,
    figsize=(16, 14),
    textsize=10,
    hdi_probs=(0.3, 0.6, 0.9),
)

plt.suptitle(f'MCMC Parameter Posterior Distributions\n'
            f'(N_eff = {len(flat_samples):,}, burn-in = {discard_burn}, thin = {thin_interval})', 
            fontsize=14, y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()


######################################################################
#


######################################################################
# 3.5 Plotting the Chains
# -----------------------
# 

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

chain_full = analysis_chain
labels = ["m" if idx == 0 else PARAM_NAMES[idx] for idx in INVERT_INDICES]

for i in range(n_dim):
    ax = axes[i]
    
    for walker in range(min(14, chain_full.shape[1])):
        ax.plot(chain_full[:, walker, i], alpha=0.6, linewidth=0.5)
    
    ax.axhline(reference_values_scaled[i], color='black', linestyle='--', alpha=0.8, 
            linewidth=2, label='Reference Value' if i == 0 else '')    
    ax.set_xlabel('Step')
    ax.set_ylabel(f'{labels[i]}')
    ax.set_title(f'Trace: {labels[i]}')
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend()

for i in range(n_dim, len(axes)):
    axes[i].remove()

plt.suptitle('MCMC Trace Plots (All Walkers)', fontsize=16)
plt.tight_layout()
plt.show()


######################################################################
#


######################################################################
# 4. Real Data Inversion
# ----------------------
# 
# -  We then apply our deterministic inversion via CoFI on actual
#    observational data obtained from `NASA JPL
#    Horizons <https://ssd.jpl.nasa.gov/horizons/app.html#/>`__.
# 
# -  The data consists of geometric Cartesian position and velocity
#    vectors of **Uranus**, relative to the **Solar System Barycenter**,
#    with the following settings:
# 
#    -  **Target body**: Uranus (799)
#    -  **Center body**: Solar System Barycenter (0)
#    -  **Reference frame**: Ecliptic of J2000.0
#    -  **Time span**: A.D. 1775-Jan-01 to 2125-Jan-02
#    -  **Step size**: 1 calendar year
#    -  **Output format**: Cartesian position and velocity (AU, AU/day)
#    -  **Output type**: GEOMETRIC states
#    -  **Calendar mode**: Mixed Julian/Gregorian
#    -  **Ephemeris source**: ``ura183_merged`` (Uranus), ``DE441`` (Solar
#       System)
# 
# This dataset provides real-world observations to test the robustness of
# our inversion pipeline.
# 

#uncomment to install the astroquery package
# !pip install astroquery


######################################################################
#


######################################################################
# 4.1 Getting the actual data from NASA
# -------------------------------------
# 

from neptune_deterministic_methods import get_actual_data
U_true = get_actual_data(z_scaling = False, T = 190)


######################################################################
#


######################################################################
# 4.2 Setting the Inversion and Running it
# ----------------------------------------
# 
# The real-data section uses the same Bayesian objective as the synthetic
# section, but calibrates the likelihood with the effective per-coordinate
# residual scales obtained from the deterministic Morozov analysis. The
# MCMC walkers are initialized near that deterministic real-data solution
# rather than across the whole prior box.
# 

param_bounds_lower = np.array([bound[0] for bound in PARAM_BOUNDS])
param_bounds_upper = np.array([bound[1] for bound in PARAM_BOUNDS])

bounds_lower_scaled = scale_param(param_bounds_lower)
bounds_upper_scaled = scale_param(param_bounds_upper)

bounds_lower_scaled[0] = np.log10(bounds_lower_scaled[0])  # Convert mass back to log scale
bounds_upper_scaled[0] = np.log10(bounds_upper_scaled[0])  # Convert mass back to log scale

# Effective real-data residual scales from the deterministic Morozov analysis.
# These include observational noise plus model mismatch for this forward model.
REAL_DETERMINISTIC_SIGMAS = np.array([7.8277e-02, 7.4224e-02, 9.7833e-04])
# sigma_x, sigma_y, sigma_z = REAL_DETERMINISTIC_SIGMAS
sigma_x, sigma_y, sigma_z = np.array([5e-1, 5e-1, 5e-3])

diagonal_values = [1/sigma_x**2]*T + [1/sigma_y**2]*T + [1/sigma_z**2]*T
inv_variances = np.array(diagonal_values)

# Deterministic real-data solution from the Morozov run; used as the MCMC centre.
REAL_DETERMINISTIC_M_UNSCALED = np.array([
    5.129472e-05,
    -2.991824e+01,
    3.145458e+00,
    1.050744e-01,
    -2.804390e-04,
    -3.109318e-03,
    4.187314e-05,
])

real_mcmc_center_scaled = scale_param(REAL_DETERMINISTIC_M_UNSCALED).astype(float)
real_mcmc_center_scaled[0] = np.log10(real_mcmc_center_scaled[0])
# real_mcmc_center_scaled = scale_param(m_0).astype(float)
# real_mcmc_center_scaled[0] = np.log10(real_mcmc_center_scaled[0])

real_center_pred = predict_U(real_mcmc_center_scaled, T=T, dt=dt, z_scale_factor=z_scale_factor)
real_center_residual = U_true - real_center_pred
real_center_reduced_chi2 = np.sum(real_center_residual**2 * inv_variances) / len(inv_variances)

print("Real-data effective sigmas:")
print(f"  sigma_x = {sigma_x:.4e}")
print(f"  sigma_y = {sigma_y:.4e}")
print(f"  sigma_z = {sigma_z:.4e}")
print("Deterministic real-data MCMC centre (scaled/log):", real_mcmc_center_scaled)
print("Residual RMS at deterministic centre:")
print(f"  X: {np.sqrt(np.mean(real_center_residual[:T]**2)):.4e}")
print(f"  Y: {np.sqrt(np.mean(real_center_residual[T:2*T]**2)):.4e}")
print(f"  Z: {np.sqrt(np.mean(real_center_residual[2*T:]**2)):.4e}")
print(f"Reduced chi^2 at deterministic centre: {real_center_reduced_chi2:.4f}")


######################################################################
#

n_walkers = 28
n_dim = len(INVERT_INDICES)
nsteps = REAL_LONG_STEPS if RUN_LONG_MCMC else VALIDATION_STEPS
use_pool = True
show_progress = RUN_LONG_MCMC
run_mode = "long" if RUN_LONG_MCMC else "validation"

print(f"Optimized setup: {len(inv_variances)} inverse variance values")
print(f"Running {run_mode} MCMC chain: {nsteps} steps, {n_walkers} walkers")
print("Set RUN_LONG_MCMC = True in the configuration cell to run the calibrated 80,000-step real-data chain.")

print("Parameter bounds (unscaled):")
print("  Lower:", param_bounds_lower)
print("  Upper:", param_bounds_upper)
print("Scaled bounds:")
print("  Lower:", bounds_lower_scaled)
print("  Upper:", bounds_upper_scaled)

print(f"Number of dimensions: {n_dim}")

rng = np.random.default_rng(84)

walker_scales = 0.03 * (bounds_upper_scaled - bounds_lower_scaled)
walker_scales[0] = 0.05

walkers_start = real_mcmc_center_scaled + rng.normal(
    scale=walker_scales,
    size=(n_walkers, n_dim),
)

walkers_start = np.clip(walkers_start, bounds_lower_scaled, bounds_upper_scaled)

print("Initial state shape:", walkers_start.shape)
print(
    "Initial walker mass min/max/std:",
    f"{walkers_start[:, 0].min():.6g}",
    f"{walkers_start[:, 0].max():.6g}",
    f"{walkers_start[:, 0].std():.6g}",
)

for i, w in enumerate(walkers_start):
    ll = my_log_likelihood(w, U_true)
    if not np.isfinite(ll):
        print(f"Invalid log-likelihood at walker {i}: {w}, ll = {ll}")
        
for i, w in enumerate(walkers_start):
    walkers_start[i] = np.clip(w, bounds_lower_scaled, bounds_upper_scaled)
    if not np.array_equal(w, walkers_start[i]):
        print(f"Walker {i} clipped from {w} to {walkers_start[i]}")

print("All walkers are now within bounds.")

def my_log_prior(m_scaled):
    # The synthetic example can use a broad uniform prior because the data are
    # generated by this same simplified forward model. The real Horizons data
    # include full-Solar-System and ephemeris effects that this reduced model
    # cannot represent exactly. With only a uniform box prior, the sampler can
    # find unphysical Neptune parameters that overfit those model discrepancies.
    # This Gaussian prior is the Bayesian analogue of the deterministic
    # regularization term: it encodes external knowledge that Neptune's state is
    # near the MAP value determined in the deterministic inversion notebook
    # (finding_neptune_via_deterministic_inv) while still allowing uncertainty around it.
    if np.any(m_scaled < bounds_lower_scaled) or np.any(m_scaled > bounds_upper_scaled):
        return -np.inf

    prior_sigma_full = np.array([
        0.2,  # log mass, atry 0.1
        0.5,  # x
        0.5,  # y
        0.20,  # z 
        0.1,  # vx
        0.1,  # vy
        0.20,  # vz 
    ])

    prior_sigma = prior_sigma_full[INVERT_INDICES]

    dm = m_scaled - real_mcmc_center_scaled
    return -0.5 * np.sum((dm / prior_sigma)**2)


for i, w in enumerate(walkers_start):
    if not np.isfinite(my_log_prior(w)):
        print(f"ERROR: Walker {i} still outside prior bounds: {w}")
    else:
        ll = my_log_likelihood(w, U_true)
        if not np.isfinite(ll):
            print(f"WARNING: Walker {i} has invalid log-likelihood: {ll}")
            
inv_problem = BaseProblem()
inv_problem.name = "Neptune Orbit Determination - Config Driven"
inv_problem.set_data(U_true)
inv_problem.set_forward(predict_U)
inv_problem.set_initial_model(np.atleast_1d(real_mcmc_center_scaled))
inv_problem.set_log_prior(my_log_prior)
inv_problem.set_log_likelihood(my_log_likelihood, args=(U_true,))
inv_problem.set_model_shape(n_dim)

import emcee
stretch_move = emcee.moves.StretchMove(a=1.2)

inv_result, elapsed = run_cofi_emcee(
    inv_problem,
    walkers_start,
    n_walkers,
    nsteps,
    InversionOptions,
    Inversion,
    progress=show_progress,
    use_pool=use_pool,
    n_threads=14,
    moves=stretch_move,
    skip_initial_state_check=True,
)

print(f"OPTIMIZED Inversion completed in {elapsed:.2f} seconds using `emcee`.")
print("The inversion result from `emcee`:")
inv_result.summary()


######################################################################
#


######################################################################
# 4.3 Getting the acceptance rates and auto-correlation times
# -----------------------------------------------------------
# 

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

param_names = PARAM_NAMES
param_labels = ["m" if idx == 0 else PARAM_NAMES[idx] for idx in INVERT_INDICES]

sampler = inv_result.sampler
reference_values_scaled = scaled_reference_values(scale_param, m_0)

n_trajectory_samples = 10_000
discard_burn = 0
thin_interval = 1

saved_result = None
if (not RUN_LONG_MCMC) and USE_SAVED_LONG_RUN_RESULTS:
    saved_result = load_mcmc_results(LONG_RUN_RESULTS_FILE, "real")

if saved_result is not None:
    analysis_chain = saved_result["chain"]
    flat_samples = analysis_chain[discard_burn::thin_interval].reshape(-1, analysis_chain.shape[-1])
    acceptance_rates = saved_result["acceptance_fraction"]
    autocorr_times = saved_result["autocorr_times"]
    sample_source = f"saved long-run result ({LONG_RUN_RESULTS_FILE})"
    print(f"Loaded real-data long-run results from {LONG_RUN_RESULTS_FILE}")
else:
    analysis_chain, flat_samples, sample_source = analysis_samples(
        sampler,
        reference_values_scaled,
        bounds_lower_scaled,
        bounds_upper_scaled,
        thin=thin_interval,
        burn=discard_burn,
        seed=84,
    )
    acceptance_rates = sampler.acceptance_fraction
    autocorr_times = safe_autocorr_time(sampler, fallback_chain=analysis_chain, discard=300)

selected_samples = selected_trajectory_samples(flat_samples, n_trajectory_samples, seed=84)

mean_acceptance = np.mean(acceptance_rates)
max_autocorr = np.max(autocorr_times)
mean_autocorr = np.mean(autocorr_times)

if RUN_LONG_MCMC and SAVE_LONG_RUN_RESULTS:
    save_mcmc_results(
        LONG_RUN_RESULTS_FILE,
        "real",
        analysis_chain,
        flat_samples,
        acceptance_rates,
        autocorr_times,
        nsteps,
        n_walkers,
        param_labels,
    )
    print(f"Saved real-data long-run results to {LONG_RUN_RESULTS_FILE}")

print(f"Posterior samples for plots: {sample_source} ({len(flat_samples):,} samples)")
print(f"Mean acceptance rate: {mean_acceptance:.3f}")
print("Acceptance rate per walker:")
for i, acc in enumerate(acceptance_rates):
    print(f"  Walker {i:2d}: {acc:.3f}")

print("\nAcceptance rate statistics:")
print(f"  Range: {np.min(acceptance_rates):.3f} - {np.max(acceptance_rates):.3f}")
print(f"  Std dev: {np.std(acceptance_rates):.3f}")

print("Autocorrelation times by parameter:")
for i, (label, tau) in enumerate(zip(param_labels, autocorr_times)):
    print(f"  {label:>3s}: {tau:6.1f} steps")

print("\nAutocorrelation statistics:")
print(f"  Mean τ: {mean_autocorr:.1f} steps")
print(f"  Max τ:  {max_autocorr:.1f} steps")


######################################################################
#


######################################################################
# 4.4 Plotting the Posteriors
# ---------------------------
# 


######################################################################
# -  We now plot the posteriors and along with that, we alsomark the true
#    parameter values of Neptune using a black ‘X’ mark.
# 
# -  Note that all our parameters are scaled.
# 

reference_values_scaled = scaled_reference_values(scale_param, m_0)
param_labels = ["m" if idx == 0 else PARAM_NAMES[idx] for idx in INVERT_INDICES]
ref_vals = [reference_values_scaled[i] for i in range(n_dim)]

print("Creating corner plot with KDE contours and reference values...")

fig = plot_pair_kde_legacy_style(
    flat_samples,
    param_labels,
    ref_vals,
    figsize=(16, 14),
    textsize=10,
    hdi_probs=(0.3, 0.6, 0.9),
)

plt.suptitle(f'MCMC Parameter Posterior Distributions\n'
            f'(N_eff = {len(flat_samples):,}, burn-in = {discard_burn}, thin = {thin_interval})', 
            fontsize=14, y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()


######################################################################
#


######################################################################
# 4.5 Plotting the Chains
# -----------------------
# 

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

chain_full = analysis_chain
labels = ["m" if idx == 0 else PARAM_NAMES[idx] for idx in INVERT_INDICES]

for i in range(n_dim):
    ax = axes[i]
    
    for walker in range(min(14, chain_full.shape[1])):
        ax.plot(chain_full[:, walker, i], alpha=0.6, linewidth=0.5)
    
    ax.axhline(reference_values_scaled[i], color='black', linestyle='--', alpha=0.8, 
            linewidth=2, label='Reference Value' if i == 0 else '')    
    ax.set_xlabel('Step')
    ax.set_ylabel(f'{labels[i]}')
    ax.set_title(f'Trace: {labels[i]}')
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend()

for i in range(n_dim, len(axes)):
    axes[i].remove()

plt.suptitle('MCMC Trace Plots (All Walkers)', fontsize=16)
plt.tight_layout()
plt.show()


######################################################################
#

true_scaled = scale_param(m_0).astype(float)
true_scaled[0] = np.log10(true_scaled[0])

det_scaled = real_mcmc_center_scaled

flat = sampler.get_chain(discard=10000, thin=100, flat=True)
logp = sampler.get_log_prob(discard=10000, thin=100, flat=True)

map_scaled = flat[np.argmax(logp)]

print("log posterior at true:", my_log_prior(true_scaled) + my_log_likelihood(true_scaled, U_true))
print("log posterior at deterministic centre:", my_log_prior(det_scaled) + my_log_likelihood(det_scaled, U_true))
print("log posterior at chain MAP:", my_log_prior(map_scaled) + my_log_likelihood(map_scaled, U_true))

print("true:", true_scaled)
print("det :", det_scaled)
print("MAP :", map_scaled)


######################################################################
#


######################################################################
# 5. Watermark
# ============
# 
# -  For version of libraries used.
# 

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import numba
import arviz
import astroquery
import cofi

def print_versions():
    print("Library Versions:")
    print(f"{'numpy':<15}: {np.__version__}")
    print(f"{'matplotlib':<15}: {plt.matplotlib.__version__}")
    print(f"{'tqdm':<15}: {tqdm.__version__}")
    print(f"{'numba':<15}: {numba.__version__}")
    print(f"{'arviz':<15}: {arviz.__version__}")
    print(f"{'astroquery':<15}: {astroquery.__version__}")
    print(f"{'cofi':<15}: {cofi.__version__}")

print_versions()


######################################################################
#
# sphinx_gallery_thumbnail_number = -1