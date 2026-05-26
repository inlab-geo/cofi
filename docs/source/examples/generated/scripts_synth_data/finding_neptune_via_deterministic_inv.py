"""
Finding Neptune using Uranus
============================

"""


######################################################################
# Note : - For better understanding of ths notebook, refer to the
# `md_file <https://github.com/inlab-geo/cofi-examples/blob/main/theory/finding_neptune_deterministic.md>`__
# designed specifically for this notebook and better insights into the
# theory.
# 
# -  The import methods and functions from
#    `neptune_deterministic_methods <https://github.com/inlab-geo/cofi-examples/blob/main/examples/Finding_Neptune_Inversions/neptune_deterministic_methods.py>`__
#    and
#    `setup_inversion <https://github.com/inlab-geo/cofi-examples/blob/main/examples/Finding_Neptune_Inversions/setup_inversion.py>`__
#    are used to set up the simulation and perform the necessary
#    calculations.
# 

# This notebook requires the following libraries to run. In order to install them uncomment the lines below
# %pip install cofi
# %pip install numba
# %pip install tqdm
# %pip install matplotlib
# %pip install astroquery

#also make sure you have the neptune_deterministic_methods.py file in the same directory as this notebook.

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
# -  This formulation uses Newton’s Law of Universal Gravitation to model
#    the **net gravitational influence** from multiple bodies on a single
#    target planet.
# 


######################################################################
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
warnings.filterwarnings('ignore')

from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

######################################################################
#


######################################################################
# -  We solve our ODEs with the **Runge-Kutta 4 (RK4)** method, which is
#    an explicit and iterative method, well-suited for initial value
#    problems.
# 


######################################################################
# We solve the system above with a fused ``@njit`` Runge-Kutta 4 (RK4)
# integrator (``integrate_all_bodies`` in
# ``neptune_deterministic_methods.py``). The function ``run_simulation``
# is a thin wrapper around this integrator that returns the per-body
# trajectories, with optional plotting.
# 


######################################################################
# We demonstrate the forward model by integrating the inner solar system
# over a 100-year window and plotting each planet’s orbit (xy plane,
# heliocentric).
# 

from neptune_deterministic_methods import run_simulation

# `run_simulation` builds initial-condition arrays, runs the fused @njit
# integrator `integrate_all_bodies`, and returns per-body trajectories.
trajectories = run_simulation(
    T=100, dt=1,
    plot_only=['Uranus', 'Neptune', 'Saturn', 'Jupiter',
               'Mars', 'Earth', 'Venus', 'Mercury'],
)


######################################################################
#


######################################################################
# 3. Synthetic Data and Inversion Setup
# -------------------------------------
# 
# We generate noisy synthetic observations of Uranus by perturbing the
# truth Neptune state and running the forward model, then set up the
# scaling and bounds used by the inversion. The inversion itself is
# performed in Section 4 with the χ² Morozov method on both synthetic and
# (in 4.2) real NASA Horizons data.
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

# True/reference parameters for Neptune [mass, x, y, z, vx, vy, vz]
from setup_inversion import get_inversion_indices, set_true_m, unscale_param, get_param_bounds, get_starting_points, get_param_scales, validate_config, set_initial_conditions     

m_0 = set_true_m()
initial_conditions = set_initial_conditions()
PARAM_BOUNDS = get_param_bounds()
PARAM_SCALES = get_param_scales()

######################################################################
#


######################################################################
# -  Cell below is used for validating our setup and setting up the
#    scaling and unscaling functions, along with a
#    ``build_neptune_vector`` method that helps us build the scaled
#    version of our model depending on what we are inverting for. For
#    example - if it’s just the mass then all other parameters, velocities
#    and positions, would be derived from the true values.
# 

validate_config()

######################################################################
#

names = list(initial_conditions.keys())
n_bodies = len(names)
uranus_idx = names.index("Uranus")

######################################################################
#


######################################################################
# -  The cell below imports ``predict_U``, our full forward model.
#    Internally it calls the fused ``@njit`` integrator
#    ``integrate_uranus`` (the Uranus-only sibling of
#    ``integrate_all_bodies`` used in Section 2).
# 

from setup_inversion import scale_param, get_starting_points
from neptune_deterministic_methods import predict_U, residual, jacobian

INVERT_INDICES = get_inversion_indices()
STARTING_POINTS = get_starting_points()
z_scale_factor = 1

m_start_scaled = scale_param(np.array(STARTING_POINTS))
if len(INVERT_INDICES) == 1:
    Nmstart_scaled = m_start_scaled.item() if hasattr(m_start_scaled, 'item') else m_start_scaled

print(f"\nStarting points (unscaled): {STARTING_POINTS}")
print(f"Starting points (scaled): {m_start_scaled}")

print("\nTesting forward function...")
try:
    pred_test = predict_U(m_start_scaled, T=T, dt=dt, z_scale_factor=z_scale_factor)
    residual_test = residual(m_start_scaled, U_true, T=T, dt=dt)
    print(f"Initial residual norm: {np.linalg.norm(residual_test):.6f}")
    print(f"Residual by component:")
    print(f"  X component: {np.linalg.norm(residual_test[:T]):.6f}")
    print(f"  Y component: {np.linalg.norm(residual_test[T:2*T]):.6f}")
    print(f"  Z component: {np.linalg.norm(residual_test[2*T:]):.6f}")
except Exception as e:
    print(f"Forward function test failed: {e}")
    import traceback
    traceback.print_exc()



######################################################################
#


######################################################################
# 4. Regularisation by χ² Discrepancy (Morozov’s principle)
# ---------------------------------------------------------
# 
# The L-curve method in Section 3.3 picks the regularisation strength
# :math:`\alpha` from a “knee” in a trade-off curve. A more principled
# alternative is **Morozov’s discrepancy principle**, which uses the noise
# level of the data directly.
# 
# Given per-component noise standard deviations :math:`\sigma_c` (for
# components :math:`c \in \{x,y,z\}`), the chi-squared statistic at a
# candidate solution :math:`m` is
# 
# .. math:: \chi^2(m) \; = \; \sum_{c \in \{x,y,z\}} \sum_{t=1}^{T} \frac{\big(d_{c,t} - g_{c,t}(m)\big)^2}{\sigma_c^2}
# 
# For a model that fits the data to within the noise, the expected value
# is :math:`\mathbb{E}[\chi^2] \approx N`, where :math:`N` is the total
# number of data points (here :math:`N = 3T = 570`). The reduced
# chi-squared :math:`\chi^2/N` is then :math:`\approx 1`.
# 
# **Morozov’s pick**: choose the regularisation parameter :math:`\alpha`
# such that :math:`\chi^2(m_\alpha)/N` is closest to 1 — fitting to the
# noise floor, neither under- nor over-fitting.
# 
# The inversion itself uses the per-parameter-weighted Tikhonov
# regularisation we built in Section 3.3 (``residual_weighted`` +
# ``jacobian_weighted``). What changes here is only the **selection rule**
# for :math:`\alpha`.
# 
# For *synthetic* data we know the injected noise levels exactly, so the
# chi-squared criterion is directly applicable. For *real* data the noise
# is unknown and needs to be estimated — that’s deferred to a follow-up
# section.
# 

import cofi
from neptune_deterministic_methods import residual_weighted, jacobian_weighted, get_actual_data, plot_uranus_orbits, plot_neptune_orbits
import time


def run_chi2_morozov_sweep(U_data, alphas, m_start_scaled,
                           sigmas_per_coord, bounds_lower_scaled, bounds_upper_scaled,
                           T=190, dt=1, verbose=True):
    '''Sweep alpha via cofi.utils.InversionPool, compute chi^2 at each
    converged solution against the supplied per-coordinate sigmas.

    Per-alpha inversion uses the cofi pattern (BaseProblem + InversionOptions
    + scipy.optimize.least_squares tool), matching the L-curve sweep in
    Section 3.3.

    Parameters
    ----------
    U_data : np.ndarray, shape (3*T,)
        Concatenated [x, y, z] Uranus observations.
    alphas : iterable of float
        Regularisation strengths to sweep.
    m_start_scaled : np.ndarray
        Starting model in scaled coordinates; also the centre of the Tikhonov term.
    sigmas_per_coord : tuple (sigma_x, sigma_y, sigma_z)
        Standard deviation of noise on each coordinate, in AU.
    bounds_lower_scaled, bounds_upper_scaled : np.ndarray
        Box bounds in scaled coordinates.
    T, dt : int, float
        Forward-model arguments.

    Returns
    -------
    results : list of dict
        One entry per alpha with keys: alpha, chi2, chi2_per_N, residual_norm,
        m_scaled, m_unscaled.
    idx_best : int
        Index of the entry with chi^2/N closest to 1 (the Morozov pick).
    N_data : int
        Number of data points (= 3*T).
    '''
    sigma_x, sigma_y, sigma_z = sigmas_per_coord
    inv_var = np.concatenate([
        np.full(T, 1.0 / sigma_x**2),
        np.full(T, 1.0 / sigma_y**2),
        np.full(T, 1.0 / sigma_z**2),
    ])
    N_data = inv_var.size
    reg_weight = 1.0 / np.abs(m_start_scaled)

    # Build one BaseProblem per alpha (cofi pattern, same as Section 3.3 L-curve)
    problems = []
    for alpha in alphas:
        inv_problem_alpha = BaseProblem()
        inv_problem_alpha.name = f"Neptune Morozov chi^2 sweep alpha={alpha}"
        inv_problem_alpha.set_data(U_data)
        inv_problem_alpha.set_forward(predict_U, args=[T, dt])
        inv_problem_alpha.set_initial_model(np.atleast_1d(m_start_scaled))
        inv_problem_alpha.set_residual(
            residual_weighted,
            args=(U_data, m_start_scaled, reg_weight, alpha, T, dt),
        )
        inv_problem_alpha.set_jacobian(
            jacobian_weighted,
            args=(U_data, m_start_scaled, reg_weight, alpha, T, dt),
        )
        problems.append(inv_problem_alpha)

    inv_options_sweep = InversionOptions()
    inv_options_sweep.set_tool("scipy.optimize.least_squares")
    inv_options_sweep.set_params(
        bounds=(bounds_lower_scaled, bounds_upper_scaled),
        method="trf",
        max_nfev=100,
        ftol=1e-12,
        xtol=1e-12,
        verbose=0,
    )

    # Run all alpha-inversions through cofi.utils.InversionPool
    t0 = time.time()
    pool = cofi.utils.InversionPool(
        list_of_inv_problems=problems,
        list_of_inv_options=inv_options_sweep,
        parallel=False,
    )
    all_inv_results, _ = pool.run()
    pool_time = time.time() - t0

    # Compute chi^2 for each converged solution
    results = []
    for alpha, inv_result in zip(alphas, all_inv_results):
        m_alpha = np.atleast_1d(inv_result.model)
        final_pred = predict_U(m_alpha, T=T, dt=dt)
        residuals = U_data - final_pred
        chi2 = float(np.sum(residuals**2 * inv_var))
        results.append({
            'alpha': alpha,
            'chi2': chi2,
            'chi2_per_N': chi2 / N_data,
            'residual_norm': float(np.linalg.norm(residuals)),
            'm_scaled': m_alpha.copy(),
            'm_unscaled': unscale_param(m_alpha),
        })

    if verbose:
        print(f"Per-coordinate sigmas (AU): x = {sigma_x:.1e}, y = {sigma_y:.1e}, z = {sigma_z:.1e}")
        print(f"N = {N_data}  (target: chi^2 = N, equivalently chi^2/N = 1)")
        print(f"cofi.utils.InversionPool ran {len(alphas)} inversions in {pool_time:.1f}s\n")
        print(f"{'alpha':>10}  {'chi^2':>12}  {'chi^2/N':>10}  {'||resid||':>12}")
        print('-' * 54)
        for r in results:
            print(f"{r['alpha']:>10.4e}  {r['chi2']:>12.4e}  {r['chi2_per_N']:>10.4f}  {r['residual_norm']:>12.6f}")

    diffs = np.abs(np.array([r['chi2_per_N'] for r in results]) - 1.0)
    idx_best = int(np.argmin(diffs))
    return results, idx_best, N_data


def iterate_chi2_morozov(U_data, m_start_scaled, alphas,
                         bounds_lower_scaled, bounds_upper_scaled,
                         T=190, dt=1, max_iter=5, tol=0.0005, verbose=True):
    '''Iterative bootstrap-sigma chi^2 Morozov for unknown-noise problems.

    1. Estimate per-coordinate sigma from residuals at m_start_scaled.
    2. Run run_chi2_morozov_sweep with these sigmas; pick alpha minimising
       |chi^2/N - 1|.
    3. Re-estimate sigma from residuals at the picked alpha's m.
    4. Iterate until max relative sigma change < tol.

    Returns a dict with keys: alpha (final picked), m_scaled, m_unscaled,
    chi2_per_N, sigmas, history (list of per-iteration entries).
    '''
    def _estimate_sigmas(r):
        return (float(np.std(r[:T])), float(np.std(r[T:2*T])), float(np.std(r[2*T:])))

    r0 = U_data - predict_U(m_start_scaled.copy(), T=T, dt=dt)
    sigma_x, sigma_y, sigma_z = _estimate_sigmas(r0)
    if verbose:
        print(f"Initial residual at m_start: ||r0|| = {np.linalg.norm(r0):.6f}")
        print(f"Initial sigmas: x={sigma_x:.4e}, y={sigma_y:.4e}, z={sigma_z:.4e}")

    history = []
    for it in range(max_iter):
        sigmas = (sigma_x, sigma_y, sigma_z)
        if verbose:
            print(f"\n--- Outer iteration {it + 1}/{max_iter} ---")
        results, idx_best, N_data = run_chi2_morozov_sweep(
            U_data, alphas, m_start_scaled, sigmas,
            bounds_lower_scaled, bounds_upper_scaled,
            T=T, dt=dt, verbose=verbose,
        )
        best = results[idx_best]
        r = U_data - predict_U(best['m_scaled'], T=T, dt=dt)
        sigma_x_new, sigma_y_new, sigma_z_new = _estimate_sigmas(r)
        rel_change = max(
            abs(sigma_x_new - sigma_x) / max(sigma_x, 1e-30),
            abs(sigma_y_new - sigma_y) / max(sigma_y, 1e-30),
            abs(sigma_z_new - sigma_z) / max(sigma_z, 1e-30),
        )
        history.append({
            'iter': it + 1, 'sigmas_used': sigmas,
            'sigmas_new': (sigma_x_new, sigma_y_new, sigma_z_new),
            'alpha_star': best['alpha'], 'chi2_per_N': best['chi2_per_N'],
            'best': best, 'rel_change': rel_change,
        })
        if verbose:
            print(f"  Pick: alpha={best['alpha']:.4e}, chi^2/N={best['chi2_per_N']:.4f}")
            print(f"  Re-estimated sigmas: x={sigma_x_new:.4e}, y={sigma_y_new:.4e}, z={sigma_z_new:.4e}")
            print(f"  Max relative sigma change: {rel_change:.4f}  (tol={tol})")
        if rel_change < tol:
            if verbose:
                print(f"  CONVERGED after {it + 1} outer iteration(s)")
            sigma_x, sigma_y, sigma_z = sigma_x_new, sigma_y_new, sigma_z_new
            break
        sigma_x, sigma_y, sigma_z = sigma_x_new, sigma_y_new, sigma_z_new

    final = history[-1]
    return {
        'alpha': final['alpha_star'],
        'm_scaled': final['best']['m_scaled'],
        'm_unscaled': final['best']['m_unscaled'],
        'chi2_per_N': final['chi2_per_N'],
        'sigmas': (sigma_x, sigma_y, sigma_z),
        'history': history,
    }


######################################################################
#


######################################################################
# 4.1 Synthetic data — proper χ² discrepancy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The synthetic dataset was created by ``generate_synthetic_data`` with
# Gaussian noise of standard deviation
# :math:`(\sigma_x, \sigma_y, \sigma_z) = (10^{-3}, 10^{-3}, 10^{-5})` AU
# on each coordinate. We use these *exact* injected values to build the
# chi-squared statistic. Sweeping :math:`\alpha` over a wide log-spaced
# range, we report :math:`\chi^2`, :math:`\chi^2/N` and residual norm at
# each, then pick the :math:`\alpha` with :math:`\chi^2/N` closest to 1.
# 

# Generate the synthetic dataset with the exact RNG state used by the inversion
# so that the chi^2 statistic uses the data we actually inverted.
np.random.seed(42)

T_synth, dt_synth = 190, 1
sigmas_synth = (1e-3, 1e-3, 1e-5)
U_synth = generate_synthetic_data(
    T=T_synth, dt=dt_synth, z_scaling=False, add_noise=True,
    noise_level=np.array(sigmas_synth),
)

# Setup: starting model and bounds in scaled coordinates
m_start_synth = scale_param(np.array(STARTING_POINTS))
PB = get_param_bounds()
bls_synth = scale_param(np.array([b[0] for b in PB]))
bus_synth = scale_param(np.array([b[1] for b in PB]))

# Sweep alpha and find the Morozov pick (chi^2/N closest to 1)
alphas_synth = np.logspace(-6, 2, 13)
sweep_synth, idx_synth, N_synth = run_chi2_morozov_sweep(
    U_synth, alphas_synth,
    m_start_synth, sigmas_synth,
    bls_synth, bus_synth,
    T=T_synth, dt=dt_synth,
)
best_synth = sweep_synth[idx_synth]

print()
print(f"Morozov pick: alpha = {best_synth['alpha']:.4e}")
print(f"  chi^2     = {best_synth['chi2']:.4f}     (target: N = {N_synth})")
print(f"  chi^2/N   = {best_synth['chi2_per_N']:.4f}     (target: 1.0)")
print(f"  ||resid|| = {best_synth['residual_norm']:.6f} AU")

# Compare recovered parameters with truth
m_0 = set_true_m()
param_names = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
print("\nInverted parameters (synthetic, chi^2 Morozov):")
for i, idx in enumerate(get_inversion_indices()):
    true_val = m_0[idx]
    val = best_synth['m_unscaled'][i]
    err = 100 * abs(val - true_val) / max(abs(true_val), 1e-30)
    print(f"  {param_names[idx]:>4s}: {val:>14.6e}   (true: {true_val:>14.6e},   err: {err:>8.2f}%)")

# Plot Uranus and Neptune orbits at the picked alpha
predicted_uranus = predict_U(best_synth['m_scaled'], T=T_synth, dt=dt_synth)
plot_uranus_orbits(predicted_uranus, U_synth, T_synth)
plot_neptune_orbits(best_synth['m_unscaled'], initial_conditions, T=T_synth, dt=dt_synth)


######################################################################
#


######################################################################
# 4.2 Real data — iterative bootstrap σ + χ² discrepancy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For real Uranus data the per-coordinate noise standard deviations
# :math:`(\sigma_x, \sigma_y, \sigma_z)` are not known. We estimate them
# by an **iterative bootstrap**:
# 
# 1. Initial :math:`\hat\sigma_c` ← per-coordinate std of the residual at
#    the starting model.
# 2. Run the χ² sweep with these :math:`\hat\sigma`, pick
#    :math:`\alpha^\star` via :math:`\chi^2/N \approx 1`.
# 3. Re-estimate :math:`\hat\sigma_c` from residuals at the converged
#    :math:`m_{\alpha^\star}`.
# 4. Iterate until :math:`\hat\sigma` stops changing (relative change
#    below tolerance).
# 
# In practice, for this problem the starting model is already close to
# truth (a known-good seed model), so the residuals at the start yield an
# excellent first estimate of :math:`\hat\sigma`, and the iteration
# usually converges in one or two outer steps.
# 
# The χ²/N at the final picked :math:`\alpha` should be ≈ 1 by
# construction — that’s the Morozov criterion telling us we’ve fit the
# data to within the estimated noise floor, no more.
# 

# Iterative chi^2 Morozov on real data
T_real, dt_real = 190, 1
U_real = get_actual_data(z_scaling=False, T=T_real)

m_start_real = scale_param(np.array(STARTING_POINTS))
PB = get_param_bounds()
bls_real = scale_param(np.array([b[0] for b in PB]))
bus_real = scale_param(np.array([b[1] for b in PB]))

# Wide log-spaced sweep over alpha
alphas_real = np.logspace(-4, 4, 9)

result_real = iterate_chi2_morozov(
    U_real, m_start_real, alphas_real,
    bls_real, bus_real,
    T=T_real, dt=dt_real,
    max_iter=5, tol=0.0005,
)

print()
print(f"Final picked alpha = {result_real['alpha']:.4e}")
print(f"Final chi^2/N      = {result_real['chi2_per_N']:.4f}")
print(f"Final sigmas       = (x:{result_real['sigmas'][0]:.4e}, y:{result_real['sigmas'][1]:.4e}, z:{result_real['sigmas'][2]:.4e})")
print(f"Outer iterations   = {len(result_real['history'])}")

# Recovered parameters vs truth
m_0 = set_true_m()
param_names = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
print("\nInverted parameters (real data, iterative chi^2 Morozov):")
for i, idx in enumerate(get_inversion_indices()):
    true_val = m_0[idx]
    val = result_real['m_unscaled'][i]
    err = 100 * abs(val - true_val) / max(abs(true_val), 1e-30)
    print(f"  {param_names[idx]:>4s}: {val:>14.6e}   (true: {true_val:>14.6e},   err: {err:>8.2f}%)")

# Plot Uranus + Neptune trajectories at the final picked alpha
predicted_uranus_real = predict_U(result_real['m_scaled'], T=T_real, dt=dt_real)
plot_uranus_orbits(predicted_uranus_real, U_real, T_real)
plot_neptune_orbits(result_real['m_unscaled'], initial_conditions, T=T_real, dt=dt_real)


######################################################################
#


######################################################################
# 5. Watermark
# ============
# 
# -  For version of libraries used.
# 

watermark_list = ["numba", "cofi", "tqdm", "numpy", "matplotlib", "astroquery"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))


######################################################################
#
# sphinx_gallery_thumbnail_number = -1