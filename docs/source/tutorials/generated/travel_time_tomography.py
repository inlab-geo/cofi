"""
2 - linear & non-linear travel time tomography
==============================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/tutorials/2_travel_time_tomography.ipynb
# 


######################################################################
# --------------
# 
# What we do in this notebook
# ---------------------------
# 
# Here we apply CoFI to two geophysical examples:
# 
# -  a **linear seismic travel time tomography** problem
# -  a **nonlinear travel time tomography** cross borehole problem
# 
# --------------
# 


######################################################################
# Learning outcomes
# -----------------
# 
# -  A demonstration of running CoFI for a regularized linear parameter
#    estimation problem. Can be used as an example of a CoFI **template**.
# -  A demonstration of how a (3rd party) nonlinear forward model can be
#    imported from geo-espresso and used. Fast Marching algorithm for
#    first arriving raypaths.
# -  See how nonlinear iterative matrix solvers can be accessed in CoFI.
# 

# Environment setup (uncomment code below)

# !pip install -U cofi geo-espresso

######################################################################
#


######################################################################
# Problem description
# -------------------
# 
# The goal in **travel-time tomography** is to infer details about the
# velocity structure of a medium, given measurements of the minimum time
# taken for a wave to propagate from source to receiver.
# 
# At first glance, this may seem rather similar to the X-ray tomography
# problem. However, there is an added complication: as we change our
# model, the route of the fastest path from source to receiver also
# changes. Thus, every update we apply to the model will inevitably be (in
# some sense) based on incorrect assumptions.
# 
# Provided the ‘true’ velocity structure is not *too* dissimilar from our
# initial guess, travel-time tomography can be treated as a weakly
# non-linear problem.
# 
# In this notebook, we illustrate both linear and one non-linear
# tomography.
# 
# In the first example the straight ray paths are fixed and independent of
# the medium through which they pass. This would be the case for X-ray
# tomography, where the data represent amplitude changes across the
# medium, or seismic tomography under the fixed ray assumption, where the
# data represent travel times across the medium.
# 
# In the second example we iteratively update seismic travel times and ray
# paths as the seismic velocity model changes, which creates a nonlinear
# tomographic problem.
# 
# In the seismic case, the travel-time of an individual ray can be
# computed as
# 
# .. math:: t = \int_\mathrm{path} \frac{1}{v(\mathbf{x})}\,\mathrm{d}\mathbf{x}
# 
# This points to an additional complication: even for a fixed path, the
# relationship between velocities and observations is not linear. However,
# if we define the ‘slowness’ to be the inverse of velocity,
# :math:`s(\mathbf{x}) = v^{-1}(\mathbf{x})`, we can write
# 
# .. math:: t = \int_\mathrm{path} {s(\mathbf{x})}\,\mathrm{d}\mathbf{x}
# 
# which *is* linear.
# 
# We will assume that the object we are interested in is 2-dimensional
# slowness field. If we discretize this model, with :math:`N_x` cells in
# the :math:`x`-direction and :math:`N_y` cells in the
# :math:`y`-direction, we can express :math:`s(\mathbf{x})` as an
# :math:`N_x \times N_y` vector :math:`\boldsymbol{s}`.
# 
# **For the linear case**, this is related to the data by
# 
# .. math:: d_i = A_{ij}s_j 
# 
# where :math:`d_i` is the travel time of the :math:`i` th path, and where
# :math:`A_{ij}` represents the path length of raypath :math:`i` in cell
# :math:`j` of the discretized model.
# 
# **For the nonlinear case**, this is related to the data by
# 
# .. math:: \delta d_i = A_{ij}\delta s_j 
# 
# where :math:`\delta d_i` is the difference in travel time, of the
# :math:`i` th path, between the observed time and the travel time in the
# reference model, and the parameters :math:`\delta s_j` are slowness
# perturbations to the reference model.
# 

import numpy as np
import matplotlib.pyplot as plt

import cofi
import espresso

######################################################################
#


######################################################################
# 1. Linear Travel Time Tomography
# --------------------------------
# 


######################################################################
# To illustrate the setting we plot a reference model supplied through the
# *espresso* Xray example, together with 100 raypaths in the dataset.
# 

linear_tomo_example = espresso.XrayTomography()

######################################################################
#

# linear_tomo_example.plot_model(linear_tomo_example.good_model, paths=True);
# linear_tomo_example.plot_model(linear_tomo_example.good_model);
plt.plot(0.5, 0.5, marker="$?$", markersize=130)
for p in linear_tomo_example._paths[:100]:
     plt.plot([p[0],p[2]],[p[1],p[3]],'y',linewidth=0.5)
print(' Data set contains ',len(linear_tomo_example._paths),' ray paths')

######################################################################
#


######################################################################
# Step 1. Define CoFI ``BaseProblem``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Now we: - set up the BaseProblem in CoFI, - supply it the data vector
# from espresso example, (i.e. the :math:`\mathbf{d}` vector) - supply it
# the Jacobian of the linear system (i.e. the :math:`A` matrix)
# 

linear_tomo_problem = cofi.BaseProblem()
linear_tomo_problem.set_data(linear_tomo_example.data)
linear_tomo_problem.set_jacobian(linear_tomo_example.jacobian(linear_tomo_example.starting_model)) # supply matrix A
sigma = 0.1 # set noise level of data
data_cov_inv = np.identity(linear_tomo_example.data_size) * (1/sigma**2)
linear_tomo_problem.set_data_covariance_inv(data_cov_inv)

######################################################################
#


######################################################################
# Since :math:`\mathbf{d}` and :math:`G` have been defined then this
# implies a linear system. Now we choose to regularize the linear system
# and solve the problem
# 
# .. math::  \min_{\mathbf s} \phi({\mathbf d},{\mathbf s}) = ({\mathbf d} - A {\mathbf s})^T C_d^{-1} ({\mathbf d} - A {\mathbf s})~ + ~ \lambda ~{\mathbf s}D^TD{\mathbf s}
# 
# The matrix system we are solving is
# 
# .. math::
# 
# 
#    (\mathbf{A}^T \textbf{C}_d^{-1} \textbf{A} + \lambda \mathbf D^T\mathbf D) \textbf{s} = \textbf{A}^T \mathbf C_d^{-1} \textbf{d}
# 

# set up regularization
lamda = 0.5   # choose regularization constant
reg_damping = lamda * cofi.utils.QuadraticReg(
    model_shape=(linear_tomo_example.model_size,)
)
linear_tomo_problem.set_regularization(reg_damping)
print('Number of slowness parameters to be solved for = ',linear_tomo_example.model_size)

######################################################################
#


######################################################################
# and lets print a summary of the set up.
# 

linear_tomo_problem.summary()

######################################################################
#


######################################################################
# Step 2. Define CoFI ``InversionOptions``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Here we choose the backend tool for solving the tomographic system,
# which is scipy’s least squares solver.
# 

tomo_options = cofi.InversionOptions()
tomo_options.set_tool("scipy.linalg.lstsq")

######################################################################
#


######################################################################
# Step 3. Define CoFI ``Inversion`` and run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

tomo_inv = cofi.Inversion(linear_tomo_problem, tomo_options)
tomo_inv_result = tomo_inv.run()
tomo_inv_result.summary()

######################################################################
#


######################################################################
# Lets plot the image to see what we got.
# 

ax = linear_tomo_example.plot_model(tomo_inv_result.model);

######################################################################
#


######################################################################
# Challenge: Fewer ray paths for linear travel time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Try and construct a tomographic solution with **fewer ray paths**.
# 
# Here we use 10416 ray paths with indices 0,10415. Try a different range
# and see what you get.
# 
# How many ray paths do you need before the image becomes recognizable?
# 
# |Upload to Jamboard 1|
# 
# Start from the code template below:
# 
# ::
# 
#    # data range
#    idx_from, idx_to = (<CHANGE ME>, <CHANGE ME>)
# 
#    # basic settings
#    d = linear_tomo_example.data
#    G = linear_tomo_example.jacobian(linear_tomo_example.starting_model)
# 
#    # now attach all the info to a BaseProblem object
#    mytomo = cofi.BaseProblem()
#    mytomo.set_data(d[idx_from:idx_to])
#    mytomo.set_jacobian(G[idx_from:idx_to,:])
# 
#    # run your problem (with the same InversionOptions) again
#    mytomo_inv = cofi.Inversion(mytomo, tomo_options)
#    mytomo_result = mytomo_inv.run()
# 
#    # check result
#    fig = linear_tomo_example.plot_model(mytomo_result.model)
#    plt.title(f'Recovered model from range ({idx_from}, {idx_to})')
#    plt.figure()
#    plt.title(' Raypaths')
#    for p in linear_tomo_example._paths[idx_from:idx_to]:
#        plt.plot([p[0],p[2]],[p[1],p[3]],'y',linewidth=0.05)
# 
# .. |Upload to Jamboard 1| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Jamboard-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
#    :target: https://jamboard.google.com/d/15UiYLe84zlkgLmi_ssbGuxRKyU-s4XuHSHsL8VppKJs/edit?usp=sharing
# 

# Copy the template above, Replace <CHANGE ME> with your answer



######################################################################
#

#@title Solution

# data range
idx_from, idx_to = (0, 3000)                    # TODO try a different range

# basic settings
d = linear_tomo_example.data
G = linear_tomo_example.jacobian(linear_tomo_example.starting_model)

# now attach all the info to a BaseProblem object
mytomo = cofi.BaseProblem()
mytomo.set_data(d[idx_from:idx_to])
mytomo.set_jacobian(G[idx_from:idx_to,:])

# run your problem (with the same InversionOptions) again
mytomo_inv = cofi.Inversion(mytomo, tomo_options)
mytomo_result = mytomo_inv.run()

# check result
fig = linear_tomo_example.plot_model(mytomo_result.model)
plt.title(f'Recovered model from range ({idx_from}, {idx_to})')
plt.figure()
plt.title(' Raypaths')
for p in linear_tomo_example._paths[idx_from:idx_to]:
    plt.plot([p[0],p[2]],[p[1],p[3]],'y',linewidth=0.05)

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 2. Non-linear Travel Time Tomography
# ------------------------------------
# 


######################################################################
# Now we demonstrate CoFI on a nonlinear iterative tomographic problem in
# a cross borehole setting.
# 
# We use a different tomographic example from espresso. Here we import the
# example module and plot the reference seismic model.
# 

nonlinear_tomo_example = espresso.FmmTomography()

nonlinear_tomo_example.plot_model(nonlinear_tomo_example.good_model, with_paths=True,lw=0.5);

######################################################################
#


######################################################################
# Solving the tomographic system with optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Now we solve the tomographic system of equations using either CoFI’s
# optimization method interface, or its iterative matrix-solver interface.
# 
# **For the optimization interface:**
# 
# We choose an objective function of the form.
# 
# .. math:: \phi(\mathbf{d},\mathbf{s}) = \frac{1}{\sigma^2}|| \mathbf{d} - \mathbf{g}(\mathbf{s})||_2^2 + \lambda_1 ||\mathbf{s}- \mathbf{s}_{0}||_2^2  + \lambda_2 ||D~\mathbf{s}||_2^2
# 
# where :math:`\mathbf{g}(\mathbf{s})` represents the predicted travel
# times in the slowness model :math:`\mathbf{s}`, :math:`\sigma^2` is the
# noise variance on the travel times, :math:`(\lambda_1,\lambda_2)` are
# weights of damping and smoothing regularization terms respectively,
# :math:`\mathbf{s}_{0}` is the reference slowness model provided by the
# espresso example, and :math:`D` is a second derivative finite difference
# stencil for the slowness model with shape ``model_shape``.
# 
# In the set up below this objective function is defined outside of CoFI
# in the function ``objective_func`` together with its gradient and
# Hessian, ``gradient`` and ``hessian`` with respect to slowness
# parameters. For convenience the regularization terms are constructed
# with CoFI utility routine ``QuadraticReg``.
# 
# For the optimization case CoFI passes ``objective_func`` and optionally
# the ``gradient`` and ``Hessian`` functions to a thrid party optimization
# backend tool such as ``scipy.minimize`` to produce a solution.
# 
# **For the iterative matrix solver interface:**
# 
# For convenience, CoFI also has its own Gauss-Newton Solver for
# optimization of a general objective function of the form.
# 
# .. math::
# 
# 
#    \phi(\mathbf{d},\mathbf{s}) = \psi((\mathbf{d},\mathbf{s}) + \sum_{r=1}^R \lambda_r \chi_r(\mathbf{s}),
# 
# where :math:`\psi` represents a data misfit term, and :math:`\chi_r` one
# or more regularization terms, with weights :math:`\lambda_r`. The
# objective function above is a special case of this. In general an
# iterative Gauss-Newton solver takes the form
# 
# .. math::
# 
#     
#    \mathbf{s}_{k+1} = \mathbf{s}_{k} - \cal{H}^{-1}(\mathbf{s}_k) \nabla \phi(\mathbf{s}_k), \quad {(k=0,1,\dots)},
# 
# where :math:`\cal{H}(\mathbf{s}_k)` is the Hessian of the objective
# function, and :math:`\nabla \phi(\mathbf{s}_k)` its gradient evaluated
# at the model :math:`\mathbf{s}_k`.
# 
# For the objective function above this becomes the simple iterative
# matrix solver
# 
# .. math::  \mathbf{s}_{k+1} = \mathbf{s}_k + (A^T C_d^{-1}A + \lambda_2\mathbf{I} +\lambda_2D^TD )^{-1} [A^T C_d^{-1} (\mathbf{d} - g(\mathbf{s}_k)) -  \lambda_2 (\mathbf{s - s}_{0}) - \lambda_2 D^TD \mathbf{s}], \quad (k=0,1,\dots)
# 
# with :math:`C_d^{-1} = \sigma^{-2} I`.
# 


######################################################################
# Step 1. Define CoFI ``BaseProblem``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# get problem information from  espresso FmmTomography
model_size = nonlinear_tomo_example.model_size               # number of model parameters
model_shape = nonlinear_tomo_example.model_shape             # 2D spatial grid shape
data_size = nonlinear_tomo_example.data_size                 # number of data points
ref_start_slowness = nonlinear_tomo_example.starting_model   # use the starting guess supplied by the espresso example

######################################################################
#


######################################################################
# Here we define the baseproblem object and a starting velocity model
# guess.
# 

# define CoFI BaseProblem
nonlinear_problem = cofi.BaseProblem()
nonlinear_problem.set_initial_model(ref_start_slowness)

######################################################################
#


######################################################################
# Here we define regularization of the tomographic system.
# 

# add regularization: damping / flattening / smoothing
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

def objective_func(slowness, reg, sigma, data_subset=None):
    if data_subset is None: 
        data_subset = np.arange(0, nonlinear_tomo_example.data_size)
    ttimes = nonlinear_tomo_example.forward(slowness)
    residual = nonlinear_tomo_example.data[data_subset] - ttimes[data_subset]
    data_misfit = residual.T @ residual / sigma**2
    model_reg = reg(slowness)
    return  data_misfit + model_reg

def gradient(slowness, reg, sigma, data_subset=None):
    if data_subset is None: 
        data_subset = np.arange(0, nonlinear_tomo_example.data_size)
    ttimes, A = nonlinear_tomo_example.forward(slowness, return_jacobian=True)
    ttimes = ttimes[data_subset]
    A = A[data_subset]
    data_misfit_grad = -2 * A.T @ (nonlinear_tomo_example.data[data_subset] - ttimes) / sigma**2
    model_reg_grad = reg.gradient(slowness)
    return  data_misfit_grad + model_reg_grad

def hessian(slowness, reg, sigma, data_subset=None):
    if data_subset is None: 
        data_subset = np.arange(0, nonlinear_tomo_example.data_size)
    A = nonlinear_tomo_example.jacobian(slowness)[data_subset]
    data_misfit_hess = 2 * A.T @ A / sigma**2 
    model_reg_hess = reg.hessian(slowness)
    return data_misfit_hess + model_reg_hess

######################################################################
#

sigma = 0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals

nonlinear_problem.set_objective(objective_func, args=[reg, sigma, None])
nonlinear_problem.set_gradient(gradient, args=[reg, sigma, None])
nonlinear_problem.set_hessian(hessian, args=[reg, sigma, None])

######################################################################
#


######################################################################
# Step 2. Define CoFI ``InversionOptions``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

nonlinear_options = cofi.InversionOptions()

# cofi's own simple newton's matrix-based optimization solver
nonlinear_options.set_tool("cofi.simple_newton")
nonlinear_options.set_params(num_iterations=5, step_length=1, verbose=True)

# scipy's Newton-CG solver (alternative approach with similar results)
# nonlinear_options.set_tool("scipy.optimize.minimize")
# nonlinear_options.set_params(method="Newton-CG", options={"xtol":1e-16})

######################################################################
#

nonlinear_options.summary()

######################################################################
#


######################################################################
# Step 3. Define CoFI ``Inversion`` and run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

nonlinear_inv = cofi.Inversion(nonlinear_problem, nonlinear_options)
nonlinear_inv_result = nonlinear_inv.run()
nonlinear_tomo_example.plot_model(nonlinear_inv_result.model);

######################################################################
#


######################################################################
# Now lets plot the true model for comparison.
# 

nonlinear_tomo_example.plot_model(nonlinear_tomo_example.good_model);

######################################################################
#


######################################################################
# Challenge: Change the number of tomographic data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# First try and repeat this tomographic reconstruction with fewer data and
# see what the model looks like.
# 
# There are 100 raypaths in the full dataset and you can tell CoFI to
# select a subset by passing an additional array of indices to the
# functions that calculate objective, gradient and hessian.
# 
# |Upload to Jamboard 2|
# 
# Start from the code template below:
# 
# ::
# 
#    # Set a subset of raypaths here
#    data_subset = np.arange(<CHANGE ME>)
# 
#    # select BaseProblem
#    my_own_nonlinear_problem = cofi.BaseProblem()
#    my_own_nonlinear_problem.set_objective(objective_func, args=[reg, sigma, data_subset])
#    my_own_nonlinear_problem.set_gradient(gradient, args=[reg, sigma, data_subset])
#    my_own_nonlinear_problem.set_hessian(hessian, args=[reg, sigma, data_subset])
#    my_own_nonlinear_problem.set_initial_model(ref_start_slowness)
# 
#    # run inversion with same options as previously
#    my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
#    my_own_result = my_own_inversion.run()
# 
#    # check results
#    my_own_result.summary()
# 
#    # plot inverted model
#    fig, paths = nonlinear_tomo_example.plot_model(my_own_result.model, return_paths=True)
#    print(f"Number of paths used: {len(data_subset)}")
# 
#    # plot paths used
#    for p in np.array(paths, dtype=object)[data_subset]:
#        fig.axes[0].plot(p[:,0], p[:,1], "g", alpha=0.5,lw=0.5)
# 
# .. |Upload to Jamboard 2| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Jamboard-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
#    :target: https://jamboard.google.com/d/1TlHvC6_vHLDaZzWT3cG2hV3KCrh3M6aoxDVAJ2RGJBw/edit?usp=sharing
# 

# Copy the template above, Replace <CHANGE ME> with your answer



######################################################################
#

#@title Solution

# Set a subset of raypaths here
data_subset = np.arange(30, 60)

# select BaseProblem
my_own_nonlinear_problem = cofi.BaseProblem()
my_own_nonlinear_problem.set_objective(objective_func, args=[reg, sigma, data_subset])
my_own_nonlinear_problem.set_gradient(gradient, args=[reg, sigma, data_subset])
my_own_nonlinear_problem.set_hessian(hessian, args=[reg, sigma, data_subset])
my_own_nonlinear_problem.set_initial_model(ref_start_slowness)

# run inversion with same options as previously
my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
my_own_result = my_own_inversion.run()

# check results
my_own_result.summary()

# plot inverted model
fig, paths = nonlinear_tomo_example.plot_model(my_own_result.model, return_paths=True)
print(f"Number of paths used: {len(data_subset)}")

# plot paths used
for p in np.array(paths, dtype=object)[data_subset]:
    fig.axes.plot(p[:,0], p[:,1], "g", alpha=0.5,lw=0.5)

######################################################################
#


######################################################################
# Challenge: Change regularization settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In the solution above we used ``damping_factor = 50``, and
# ``smoothing_factor = 5.0E-3`` and ``flattening_factor = 0``.
# 
# Experiment with these choices, e.g increasing all of them to say 100 and
# repeat the tomographic solution to see how the model changes.
# 
# Try to turn off smoothing all together but retain damping and flattening
# and see what happens.
# 
# With some choices you can force an under-determined problem which is not
# solvable.
# 
# (Note that here we revert back to using all of the data by removing the
# ``data_subset`` argument to the objective function.)
# 
# To repeat this solver with other settings for smoothing and damping
# strength. See the documentation for
# `cofi.utils.QuadraticReg <https://cofi.readthedocs.io/en/latest/api/generated/cofi.utils.QuadraticReg.html>`__.
# 
# |Upload to Jamboard 3|
# 
# You can start from the template below:
# 
# ::
# 
#    # change the combination of damping, flattening and smoothing regularizations
#    damping_factor = <CHANGE ME>                # select damping factor here to force solution toward reference slowness model 
#    flattening_factor = <CHANGE ME>             # increase flattening factor here to force small first derivatives in slowness solution
#    smoothing_factor = <CHANGE ME>              # increase smoothing factor here to force small second derivatives in slowness solution
# 
#    reg_damping = damping_factor * cofi.utils.QuadraticReg(
#        model_shape=model_shape,
#        weighting_matrix="damping",
#        reference_model=ref_start_slowness
#    )
#    reg_flattening = flattening_factor * cofi.utils.QuadraticReg(
#        model_shape=model_shape,
#        weighting_matrix="flattening"
#    )
#    reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
#        model_shape=model_shape,
#        weighting_matrix="smoothing"
#    )
#    my_own_reg = reg_damping + reg_flattening + reg_smoothing
# 
#    # set Baseproblem
#    my_own_nonlinear_problem = cofi.BaseProblem()
#    my_own_nonlinear_problem.set_objective(objective_func, args=[my_own_reg, sigma, None])
#    my_own_nonlinear_problem.set_gradient(gradient, args=[my_own_reg, sigma, None])
#    my_own_nonlinear_problem.set_hessian(hessian, args=[my_own_reg, sigma, None])
#    my_own_nonlinear_problem.set_initial_model(ref_start_slowness.copy())
# 
#    # run inversion with same options as previously
#    my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
#    my_own_result = my_own_inversion.run()
# 
#    # check results
#    fig = nonlinear_tomo_example.plot_model(my_own_result.model)
#    fig.suptitle(f"Damping {damping_factor}, Flattening {flattening_factor}, Smoothing {smoothing_factor}");
# 
# .. |Upload to Jamboard 3| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Jamboard-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
#    :target: https://jamboard.google.com/d/15FrdSczK_TK_COOLxfSJZ5CWMzH3qMoQKySJTAp5n-4/edit?usp=sharing
# 

# Copy the template above, Replace <CHANGE ME> with your answer



######################################################################
#

#@title Reference Solution

# change the combination of damping, flattening and smoothing regularizations
damping_factor = 100                # select damping factor here to force solution toward reference slowness model 
flattening_factor = 100             # increase flattening factor here to force small first derivatives in slowness solution
smoothing_factor = 0                # increase smoothing factor here to force small second derivatives in slowness solution

reg_damping = damping_factor * cofi.utils.QuadraticReg(
    model_shape=model_shape,
    weighting_matrix="damping",
    reference_model=ref_start_slowness
)
reg_flattening = flattening_factor * cofi.utils.QuadraticReg(
    model_shape=model_shape,
    weighting_matrix="flattening"
)
reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
    model_shape=model_shape,
    weighting_matrix="smoothing"
)
my_own_reg = reg_damping + reg_flattening + reg_smoothing

# set Baseproblem
my_own_nonlinear_problem = cofi.BaseProblem()
my_own_nonlinear_problem.set_objective(objective_func, args=[my_own_reg, sigma, None])
my_own_nonlinear_problem.set_gradient(gradient, args=[my_own_reg, sigma, None])
my_own_nonlinear_problem.set_hessian(hessian, args=[my_own_reg, sigma, None])
my_own_nonlinear_problem.set_initial_model(ref_start_slowness.copy())

# run inversion with same options as previously
my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
my_own_result = my_own_inversion.run()

# check results
ax = nonlinear_tomo_example.plot_model(my_own_result.model)
ax.get_figure().suptitle(f"Damping {damping_factor}, Flattening {flattening_factor}, Smoothing {smoothing_factor}");

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# ---------
# 

watermark_list = ["cofi", "espresso", "numpy", "scipy", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1