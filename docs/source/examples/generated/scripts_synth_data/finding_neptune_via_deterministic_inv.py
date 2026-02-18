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
# - The import methods and functions from
#   `neptune_deterministic_methods <https://github.com/inlab-geo/cofi-examples/blob/main/examples/Finding_Neptune_Inversions/neptune_deterministic_methods.py>`__
#   and
#   `setup_inversion <https://github.com/inlab-geo/cofi-examples/blob/main/examples/Finding_Neptune_Inversions/setup_inversion.py>`__
#   are used to set up the simulation and perform the necessary
#   calculations.
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
# - The following Notebook is based on the historical problem on how
#   Neptune was found by Johann Galle using mathematical predictions made
#   independently by two astronomers:
# 
#   - Urbain Le Verrier (France)
# 
#   - John Couch Adams (England)
# 
#   Through this Notebook we wish to demostrate how ``CoFI`` can be used
#   to solve this problem via deterministic inversion. For more details on
#   this problem, see the following
#   `thesis <www.diva-portal.org/smash/get/diva2:1218549/FULLTEXT01.pdf>`__
# 
# - In the following notebook we discuss the problem of finding Neptune’s
#   mass, its velocity components and its position coordinates in the year
#   1775, by modeling the trajectory of Uranus with and without the
#   influence of Neptune.
# 
# - We define $ g(m) $, our forward model, as vector-valued function that
#   predicts the position coordinates of Uranus at each observation time
#   :math:`t_j`, as a function of Neptune’s parameters :math:`m`:
# 
#   | 
# 
#     .. math::
# 
# 
#           g(m) =
#          \begin{bmatrix}
#          \hat x_1(m) \\
#          \vdots \\
#          \hat x_N(m) \\
#          \hat y_1(m) \\
#          \vdots \\
#          \hat y_N(m) \\
#          \hat z_1(m) \\
#          \vdots \\
#          \hat z_N(m)
#          \end{bmatrix}
#          \in \mathbb{R}^{3M \times 1}
#          
# 
#     where $ N $ is the number of data points, and $
#     :raw-latex:`\hat `x_j(m),  :raw-latex:`\hat `y_j(m),
#      :raw-latex:`\hat `z_j(m) $ are the coordinates of Uranus at data
#     point $ j  =  1,  2,  ….  N$ as a function of Neptune’s parameters $
#     m $,
#   | where :math:`m = (m_M, m_x, m_y, m_z, {m_{v_x}}, m_{v_y}, m_{v_z})`
#     is the set of parameters describing Neptune’s mass (:math:`m_M`),
#     its position coordinates :math:`(m_x, m_y, m_z)` and its velocity
#     components :math:`(m_{v_x}, m_{v_y}, m_{v_z})`
# 
#   | and :math:`d` as the data vector of positions of Uranus at different
#     time steps:
#   | 
# 
#     .. math::
# 
# 
#          d =
#          \begin{bmatrix}
#          x_1 \\
#          \vdots \\
#          x_N \\
#          y_1 \\
#          \vdots \\
#          y_N \\
#          z_1 \\
#          \vdots \\
#          z_N
#          \end{bmatrix}
#          \in \mathbb{R}^{3M \times 1}
#          
# 
#     where $ N $ is the number of data points, and $
#     :raw-latex:`\hat `x_j,  :raw-latex:`\hat `y_j,
#      :raw-latex:`\hat `z_j $ are the true coordinates of Uranus at data
#     point $ j  =  1,  2,  ….  N$.
# 
# - hence our problem formulation changes to :
# 
#   .. math::
# 
# 
#       \underset{m}{\min}   || g(m) - {d} ||_{2}^2 
#        
# 


######################################################################
# 2. Problem Setting
# ------------------
# 


######################################################################
# - This formulation uses Newton’s Law of Universal Gravitation to model
#   the **net gravitational influence** from multiple bodies on a single
#   target planet.
# 


######################################################################
# - This results in the system of differential equations:
# 
#   .. math::
# 
# 
#        \frac{d}{dt}
#        \begin{bmatrix}
#        \mathbf{r}(t) \\
#        \mathbf{v}(t)
#        \end{bmatrix}
#        = 
#        \begin{bmatrix}
#        \mathbf{v}(t) \\
#        \mathbf{a}(t)
#        \end{bmatrix}
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
# - We solve our ODEs with the **Runge-Kutta 4 (RK4)** method, which is an
#   explicit and iterative method, well-suited for initial value problems.
# 


######################################################################
# - In the following cell, we import - ``acceleration`` and ``rk4_step``
#   to serve as functions for our forward model.
# 


######################################################################
# - We now demostrate our forward model, using the above defined
#   functions, in the ``run_simulation`` method, which helps us run a
#   simulation of solar system.
# 
# - Throughout this notebook for the purpose of our inversion, we are
#   going to define mass in terms of solar masses, positions coordinates
#   in **Astonomical Units (AU)** and velocities for planets in **Au/day**
# 

from neptune_deterministic_methods import acceleration, rk4_step, run_simulation
trajectories = run_simulation(T = 100, dt = 1, plot_only=['Uranus', 'Neptune', 'Saturn', 'Jupiter', 'Mars', 'Earth', 'Venus', 'Mercury'])

######################################################################
#


######################################################################
# 3. Inversion on Synthetic Data
# ------------------------------
# 
# - We will first demonstrate the use of our deterministic inversion using
#   ``CoFI`` on synthetic data. We are going to use
#   ``levenberg-marqudt method`` to solve this deterministic inversion
#   problem.
# - The synthetic observations are generated by integrating our
#   gravitational forward model with a fourth-order Runge-Kutta (``RK4``)
#   solver to simulate Uranus’s trajectory under the influence of Neptune.
# 


######################################################################
# - We simulate observational noise by sampling from zero-mean Gaussian
#   distributions with specified variances for each coordinate:
# 
# .. math::
# 
# 
#    x_\text{obs} = x + \epsilon_x, \quad y_\text{obs} = y + \epsilon_y, \quad z_\text{obs} = z + \epsilon_z
# 
# - where
# 
# - 
# 
#   .. math::
# 
# 
#        \epsilon_x \sim \mathcal{N}(0, \sigma_x^2), \quad 
#        \epsilon_y \sim \mathcal{N}(0, \sigma_y^2), \quad 
#        \epsilon_z \sim \mathcal{N}(0, \sigma_z^2)
# 
# - with noise levels set as
# 
# - 
# 
#   .. math::
# 
# 
#      \sigma_x = \sigma_y = 10^{-3}, \quad \sigma_z = 10^{-5}
# 


######################################################################
# - The function below generates the synthetic data with the specified
#   noise levels.
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
# - Cell below is used for validating our setup and setting up the scaling
#   and unscaling functions, along with a ``build_neptune_vector`` method
#   that helps us build the scaled version of our model depending on what
#   we are inverting for. For example - if it’s just the mass then all
#   other parameters, velocities and positions, would be derived from the
#   true values.
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
# - In the cell below, the ``predict_U`` method defines our full forward
#   model and uses the ``acceleration`` and the ``rk4_step`` methods
#   defined previously.
# 
# - We also use the ``jacobian`` and the ``residual`` methods below, to be
#   used by ``CoFI`` for inversion.
# 
# - The cell below sets up the starting model for our inversion and some
#   pre-defined scales to be used for scaling while running our inversion.
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
# 3.1 Running the Inversion on Synthetic Data
# -------------------------------------------
# 

inv_problem = BaseProblem()
inv_problem.name = "Neptune Orbit Determination - Config Driven"
inv_problem.set_data(U_true)
inv_problem.set_forward(predict_U, args = [T, dt])
inv_problem.set_initial_model(np.atleast_1d(m_start_scaled))
inv_problem.set_residual(residual, args = (U_true, 0, T, dt))  # Pass U_true as an argument to residual function
inv_problem.set_jacobian(jacobian, args = (U_true, 0, T, dt))  # Pass U_true as an argument to jacobian function
bounds_lower_scaled = scale_param(np.array([bound[0] for bound in PARAM_BOUNDS]))
bounds_upper_scaled = scale_param(np.array([bound[1] for bound in PARAM_BOUNDS]))

inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.least_squares")
inv_options.set_params(
    # bounds=(bounds_lower_scaled, bounds_upper_scaled),    # Uncomment to use bounds when using trust region reflective method
    method="trf",   # Trust Region Reflective method, you can also try 'lm' (Levenberg-Marquardt) if you prefer
    max_nfev=100,
    verbose=2,
    ftol=1e-12,
    xtol=1e-12
)

print("\nRunning inversion...")
try:
    inv = Inversion(inv_problem, inv_options)
    inv_result = inv.run()
    result = inv_result.model
    result_unscaled = unscale_param(result)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    param_names = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    result_unscaled = np.atleast_1d(result_unscaled)
    
    print("\nInverted parameters:")
    for i, param_idx in enumerate(INVERT_INDICES):
        param_name = param_names[param_idx]
        estimated = result_unscaled[i]
        true_val = m_0[param_idx]
        starting = STARTING_POINTS[i]
        
        print(f"  {param_name}: {estimated:.6e} (true: {true_val:.6e})")
    
    final_pred = predict_U(result, T, dt)
    final_residual = U_true - final_pred
    print(f"\nFinal residual norm: {np.linalg.norm(final_residual):.6f}")
    print(f"Final residual by component:")
    print(f"  X component: {np.linalg.norm(final_residual[:T]):.6f}")
    print(f"  Y component: {np.linalg.norm(final_residual[T:2*T]):.6f}")
    print(f"  Z component: {np.linalg.norm(final_residual[2*T:]):.6f}")
    
    print(f'initial residual norm: {np.linalg.norm(residual(m_start_scaled, U_true)):.6f}')
    print(f'final residual norm: {np.linalg.norm(final_residual):.6f}')
    
    
    if 0 in INVERT_INDICES:
        mass_idx = INVERT_INDICES.index(0)
        neptune_mass = result_unscaled[mass_idx]
        print(f"\nEstimated Neptune mass: {neptune_mass:.6e} solar masses")
        print(f"Estimated Neptune mass: {neptune_mass * 1.989e30:.6e} kg")
    
    print("Inversion completed successfully!")
    print("="*50)
    
except Exception as e:
    print(f"Inversion failed: {e}")
    import traceback
    traceback.print_exc()


######################################################################
#


######################################################################
# 3.2 Plotting the results
# ------------------------
# 

from neptune_deterministic_methods import plot_uranus_orbits

predicted_uranus_trajectory = predict_U(result_unscaled, T = T, dt = dt)

plot_uranus_orbits(predicted_uranus_trajectory, U_true, T)

######################################################################
#

from setup_inversion import get_arrow_data
from neptune_deterministic_methods import plot_neptune_orbits

plot_neptune_orbits(result_unscaled, initial_conditions, T = T, dt = 1)

######################################################################
#


######################################################################
# 3.3 Using Regularisation to get the best regularisation parameter.
# ------------------------------------------------------------------
# 


######################################################################
# - In the following cells, we are going to demonstrate how ``CoFI`` can
#   be used to plot the regularisation curve or **L-Curve** in order to
#   get the best regularisation parameter for our inversion.
# 
# - This will ensure that our model inferred i.e the parameters of Neptune
#   are meaningfull and the trajectories are not overshooting.
# 
# - We will then use the best regularisation parameter, inferred from our
#   synthetic data, for running our final inversion on real data.
# 
# - Note that the results may differ depending on where the method starts
#   the inversion and therefore one may not get the exact same l-curve.
# 

alphas = np.logspace(-4, 2, 20)

######################################################################
#

import cofi
from neptune_deterministic_methods import callback_func, set_lcurve_inversion_params
alphas = np.logspace(-4, 2, 10)
lcurve_problems = []
m_start = initial_conditions['Neptune'].copy()
m_start = scale_param(m_start)

for alpha in alphas:
    
    inv_problem_alpha = BaseProblem()
    inv_problem_alpha.name = f"Neptune Orbit Determination alpha={alpha}"
    inv_problem_alpha.set_data(U_true)
    inv_problem_alpha.set_forward(predict_U, args = [T, dt])
    inv_problem_alpha.set_initial_model(np.atleast_1d(m_start_scaled))
    inv_problem_alpha.set_residual(residual, args = (U_true, alpha, T, dt))  # Pass U_true as an argument to residual function
    inv_problem_alpha.set_jacobian(jacobian, args = (U_true, alpha, T, dt))  # Pass U_true and alpha as arguments to jacobian function
    lcurve_problems.append(inv_problem_alpha)
    
    
inv_options_alpha = InversionOptions()
inv_options_alpha.set_tool("scipy.optimize.least_squares")
inv_options_alpha.set_params(
    bounds=(bounds_lower_scaled, bounds_upper_scaled),
    method="trf",
    max_nfev=100,
    # verbose=2, 
    ftol=1e-14,
    xtol=1e-14
)


inversion_pool = cofi.utils.InversionPool(
    list_of_inv_problems=lcurve_problems,
    list_of_inv_options=inv_options_alpha,
    callback=callback_func,
    parallel=False,  # Use parallel processing if available, works only in Windows/linux due to Multiprocessing library not being able to pickle up the forward model on MacOS
    
)

######################################################################
#

all_res, all_cb_returns = inversion_pool.run()

l_curve_points = list(zip(*all_cb_returns))
residual_norm, regularization_norm = np.array(l_curve_points)

######################################################################
#

plt.figure(figsize=(10, 8))
plt.plot(residual_norm, regularization_norm, 'k.-')
plt.xlabel(r'Norm of residual $||g(m)-d||_2$')
plt.ylabel(r'Norm of regularization term $||Rm||_2$')

for damping, res_norm, reg_norm in zip(alphas, residual_norm, regularization_norm):
    plt.plot(res_norm, reg_norm, 'ro')
    plt.text(
        res_norm - res_norm * 2e-3,
        reg_norm - reg_norm * 2e-3,
        s=f"{damping:.1e}",  # Label as scientific notation
        va='top',
        ha='right',
        fontsize=8,
        color='r'
    )

plt.title("L-curve with damping parameters")
plt.grid(True)
plt.show()

######################################################################
#


######################################################################
# 4. Real Data Inversion
# ----------------------
# 
# - We then apply our deterministic inversion via CoFI on actual
#   observational data obtained from `NASA JPL
#   Horizons <https://ssd.jpl.nasa.gov/horizons/app.html#/>`__.
# 
# - The data consists of geometric Cartesian position and velocity vectors
#   of **Uranus**, relative to the **Solar System Barycenter**, with the
#   following settings:
# 
#   - **Target body**: Uranus (799)
#   - **Center body**: Solar System Barycenter (0)
#   - **Reference frame**: Ecliptic of J2000.0
#   - **Time span**: A.D. 1775-Jan-01 to 2125-Jan-02
#   - **Step size**: 1 calendar year
#   - **Output format**: Cartesian position and velocity (AU, AU/day)
#   - **Output type**: GEOMETRIC states
#   - **Calendar mode**: Mixed Julian/Gregorian
#   - **Ephemeris source**: ``ura183_merged`` (Uranus), ``DE441`` (Solar
#     System)
# 
# This dataset provides real-world observations to test the robustness of
# our inversion pipeline.
# 

#uncomment to install the astroquery package
# !pip install astroquery

######################################################################
#

from astroquery.jplhorizons import Horizons     
from neptune_deterministic_methods import get_actual_data   
T = 190
dt = 1
U_true = get_actual_data(z_scaling=False, T=T)

m_0 = set_true_m()

initial_conditions = set_initial_conditions()
PARAM_BOUNDS = get_param_bounds()
PARAM_SCALES = get_param_scales()
INVERT_INDICES = get_inversion_indices()
STARTING_POINTS = get_starting_points()

validate_config()

######################################################################
#

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

alpha = 2.2e-1  # from the l curve

inv_problem = BaseProblem()
inv_problem.name = "Neptune Orbit Determination - Config Driven"
inv_problem.set_data(U_true)
inv_problem.set_forward(predict_U, args = [T, dt])
inv_problem.set_initial_model(np.atleast_1d(m_start_scaled))
inv_problem.set_residual(residual, args = (U_true, alpha, T, dt))  # Pass U_true as an argument to residual function
inv_problem.set_jacobian(jacobian, args = (U_true, alpha, T, dt))  # Pass U_true as an argument to jacobian function
bounds_lower_scaled = scale_param(np.array([bound[0] for bound in PARAM_BOUNDS]))
bounds_upper_scaled = scale_param(np.array([bound[1] for bound in PARAM_BOUNDS]))

inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.least_squares")
inv_options.set_params(
    bounds=(bounds_lower_scaled, bounds_upper_scaled),    # Uncomment to use bounds when using trust region reflective method
    method="trf",   # Trust Region Reflective method, you can also try 'lm' (Levenberg-Marquardt) if you prefer
    max_nfev=100,
    verbose=2,
    ftol=1e-12,
    xtol=1e-12
)

print("\nRunning inversion...")
try:
    inv = Inversion(inv_problem, inv_options)
    inv_result = inv.run()
    result = inv_result.model
    result_unscaled = unscale_param(result)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    param_names = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    result_unscaled = np.atleast_1d(result_unscaled)
    
    print("\nInverted parameters:")
    for i, param_idx in enumerate(INVERT_INDICES):
        param_name = param_names[param_idx]
        estimated = result_unscaled[i]
        true_val = m_0[param_idx]
        starting = STARTING_POINTS[i]
        
        print(f"  {param_name}: {estimated:.6e} (true: {true_val:.6e}, started: {starting:.6e})")
    
    final_pred = predict_U(result, T, dt)
    final_residual = U_true - final_pred
    print(f"\nFinal residual norm: {np.linalg.norm(final_residual):.6f}")
    print(f"Final residual by component:")
    print(f"  X component: {np.linalg.norm(final_residual[:T]):.6f}")
    print(f"  Y component: {np.linalg.norm(final_residual[T:2*T]):.6f}")
    print(f"  Z component: {np.linalg.norm(final_residual[2*T:]):.6f}")
    
    print(f'initial residual norm: {np.linalg.norm(residual(m_start_scaled, U_true)):.6f}')
    print(f'final residual norm: {np.linalg.norm(final_residual):.6f}')
    
    
    if 0 in INVERT_INDICES:
        mass_idx = INVERT_INDICES.index(0)
        neptune_mass = result_unscaled[mass_idx]
        print(f"\nEstimated Neptune mass: {neptune_mass:.6e} solar masses")
        print(f"Estimated Neptune mass: {neptune_mass * 1.989e30:.6e} kg")
    
    print("Inversion completed successfully!")
    print("="*50)
    
except Exception as e:
    print(f"Inversion failed: {e}")
    import traceback
    traceback.print_exc()


######################################################################
#

predicted_uranus_trajectory = predict_U(result_unscaled, T = T, dt = dt)

plot_uranus_orbits(predicted_uranus_trajectory, U_true, T)

######################################################################
#

from setup_inversion import get_arrow_data
from neptune_deterministic_methods import plot_neptune_orbits

plot_neptune_orbits(result_unscaled, initial_conditions, T = T, dt = 1)

######################################################################
#


######################################################################
# 5. Watermark
# ============
# 
# - For version of libraries used.
# 

watermark_list = ["numba", "cofi", "tqdm", "numpy", "matplotlib", "astroquery"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))


######################################################################
#
# sphinx_gallery_thumbnail_number = -1