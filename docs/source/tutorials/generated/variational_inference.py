"""
Variational Inference for Nonlinear Inverse Problems
====================================================

"""


######################################################################
#    **Work in progress:** this tutorial is still being developed and
#    reviewed. Its text, code and results may change.
# 


######################################################################
# --------------
# 
# What we do in this notebook
# ---------------------------
# 
# We demonstrate **variational inference (VI)** as a middle ground between
# point estimation (MAP/Laplace) and full MCMC sampling for Bayesian
# inversion. Using a nonlinear forward model that produces an asymmetric
# posterior, we compare three approaches:
# 
# 1. **Laplace approximation** – Gaussian centred at the MAP with
#    Hessian-based covariance
# 2. **Gaussian VI** – natural-gradient optimisation of a Gaussian
#    approximation to the posterior
# 3. **Sinh-arcsinh flow** – extends Gaussian VI to capture per-parameter
#    skewness and tail weight
# 
# Learning outcomes
# -----------------
# 
# -  Understand the Evidence Lower Bound (ELBO) and how VI optimises it
# -  Set up a ``BaseProblem`` with the components needed for VI (forward,
#    Jacobian, data covariance, prior precision)
# -  Run Gaussian VI and sinh-arcsinh flow VI through CoFI
# -  Use ``to_arviz()`` for posterior analysis and visualisation
# -  Appreciate when Gaussian VI is sufficient vs when a flow is needed
# 


######################################################################
# Environment setup
# -----------------
# 

# Uncomment below to install CoFI if needed
# !pip install -U cofi arviz

######################################################################
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg

from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

######################################################################
#


######################################################################
# Background: Three Roads to Posterior Uncertainty
# ------------------------------------------------
# 
# Given data :math:`d` and a forward model :math:`d = f(m) + \varepsilon`,
# Bayesian inversion seeks the posterior
# :math:`p(m|d) \propto p(d|m)\,p(m)`. Three approaches form a
# cost–fidelity hierarchy:
# 
# **Laplace approximation** (cheapest). Find the MAP estimate
# :math:`\hat{m}` and approximate
# :math:`p(m|d) \approx \mathcal{N}(\hat{m}, H^{-1})` where :math:`H` is
# the Gauss–Newton Hessian at the mode. Exact for linear forward models;
# for nonlinear problems the Hessian at a single point can misrepresent
# uncertainty elsewhere.
# 
# **Variational inference** (middle). Optimise a parametric family
# :math:`q(m)` to minimise :math:`\mathrm{KL}(q \| p(m|d))`. The precision
# update averages the Hessian over samples from :math:`q`, not just at the
# MAP. With a sinh-arcsinh flow, VI also captures per-parameter skewness
# and tail weight.
# 
# **Full MCMC** (most expensive). Generate correlated samples from the
# posterior via Markov chain Monte Carlo. Captures the full multivariate
# posterior including nonlinear correlations.
# 


######################################################################
# Problem Setup: A Nonlinear Forward Model
# ----------------------------------------
# 
# We construct a 1D inverse problem with :math:`N = 40` model parameters
# and :math:`M = 25` observations.
# 
# The forward model applies an elementwise **softplus** nonlinearity
# before averaging by a sparse kernel:
# 
# .. math:: d_j = \sum_i w_{ji}\, g(m_i) + \varepsilon_j, \qquad g(m) = \frac{1}{\alpha}\log(1 + e^{\alpha m})
# 
# The softplus has one-sided sensitivity: :math:`g'(m) = \sigma(\alpha m)`
# (sigmoid), meaning positive parameters are fully visible to data while
# negative parameters become nearly invisible. This creates **asymmetric
# posteriors** that a Gaussian cannot capture.
# 

# Problem dimensions
N = 40   # model parameters
M = 25   # observations
alpha_nl = 3.0  # nonlinearity strength

# Grid and true model
x_grid = np.linspace(0, 1, N)
m_true = np.sin(2 * np.pi * x_grid) + 0.5 * np.cos(4 * np.pi * x_grid)

# Sensor locations and sparse kernel
x_sens = np.linspace(0.02, 0.98, M)
bw = 0.15  # kernel bandwidth

rows, cols, vals = [], [], []
for j in range(M):
    dists = np.abs(x_grid - x_sens[j])
    mask = dists < 2 * bw
    wts = np.exp(-0.5 * (dists[mask] / (bw / (2 * N)))**2)
    wts /= wts.sum()
    for i, w in zip(np.where(mask)[0], wts):
        rows.append(j); cols.append(i); vals.append(w)

W = sparse.csr_matrix((vals, (rows, cols)), shape=(M, N))
print(f"Kernel W: {M}x{N}, nnz={W.nnz}, density={W.nnz / (M * N):.1%}")

######################################################################
#

# Forward model and Jacobian
def g_nonlin(m):
    """Elementwise softplus: g(m) = log(1 + exp(alpha*m)) / alpha."""
    return np.log1p(np.exp(alpha_nl * m)) / alpha_nl

def g_nonlin_deriv(m):
    """Derivative: g'(m) = sigmoid(alpha*m)."""
    return 1.0 / (1.0 + np.exp(-alpha_nl * m))

def forward(m):
    return np.asarray(W @ g_nonlin(m)).ravel()

def jacobian(m):
    return W.multiply(g_nonlin_deriv(m)[np.newaxis, :])

######################################################################
#

# Synthetic data
noise_std = 0.05
d_true = forward(m_true)
d_obs = d_true + noise_std * np.random.randn(M)

# Data covariance inverse
Cd_inv = sparse.diags(np.ones(M) / noise_std**2)

######################################################################
#

# Prior: Matern SPDE tridiagonal precision (penalises roughness)
m_prior = np.zeros(N)
alpha_prior = 2.0

diag_main = alpha_prior * 2 * np.ones(N)
diag_main[0] = diag_main[-1] = alpha_prior
diag_off = -alpha_prior * np.ones(N - 1)
Qp = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csc")

print(f"Prior precision Qp: {N}x{N}, nnz={Qp.nnz} (tridiagonal)")

######################################################################
#

# Visualise the problem
fig, axes = plt.subplots(1, 3, figsize=(11, 3))

axes[0].plot(x_grid, m_true, "k-", lw=1.5)
axes[0].set(xlabel="x", ylabel="m", title="True model")

axes[1].spy(W, markersize=3, aspect="auto")
axes[1].set(xlabel="parameter index", ylabel="sensor index",
            title=f"Kernel W (nnz={W.nnz})")

axes[2].errorbar(x_sens, d_obs, yerr=noise_std, fmt="ko", ms=4, capsize=2, label="observed")
axes[2].plot(x_sens, d_true, "r-", lw=1, label="true")
axes[2].set(xlabel="x", ylabel="d", title="Data")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

######################################################################
#


######################################################################
# 1. Laplace Approximation (Baseline)
# -----------------------------------
# 
# First, we find the MAP estimate via Gauss–Newton iteration and compute
# the Laplace approximation:
# 
# .. math:: q_\text{Laplace}(m) = \mathcal{N}(m_\text{MAP}, H_\text{GN}^{-1}(m_\text{MAP}))
# 
# This evaluates the Hessian at a single point. For our nonlinear model,
# this can misrepresent uncertainty.
# 

def gradient(m):
    res = forward(m) - d_obs
    J = jacobian(m)
    return np.asarray(J.T @ (Cd_inv @ res) + Qp @ (m - m_prior)).ravel()

def gauss_newton_hessian(m):
    J = jacobian(m)
    return (J.T @ Cd_inv @ J + Qp).tocsc()

# Gauss-Newton MAP
m_map = m_prior.copy()
for it in range(30):
    g = gradient(m_map)
    H = gauss_newton_hessian(m_map)
    dm = splinalg.spsolve(H, -g)
    m_map += dm
    if np.linalg.norm(dm) < 1e-8:
        print(f"MAP converged at iteration {it}")
        break

# Laplace: marginal stds from diagonal of H^{-1}
H_map = gauss_newton_hessian(m_map)
laplace_std = np.zeros(N)
for i in range(N):
    e = np.zeros(N); e[i] = 1.0
    laplace_std[i] = np.sqrt(splinalg.spsolve(H_map, e)[i])

print(f"MAP objective: {0.5 * np.sum(((forward(m_map) - d_obs) / noise_std)**2) + 0.5 * (m_map - m_prior) @ (Qp @ (m_map - m_prior)):.4f}")

######################################################################
#


######################################################################
# 2. Gaussian VI with CoFI
# ------------------------
# 
# Now we use CoFI’s ``cofi.gaussian_vi`` solver. This runs
# natural-gradient VI:
# 
# .. math:: \mu \leftarrow \mu - \rho_\mu\, \Omega^{-1}\, \mathbb{E}_q[\nabla_m \mathcal{L}(m)]
# 
# .. math:: \Omega \leftarrow (1 - \rho_\Omega)\,\Omega + \rho_\Omega\, \mathbb{E}_q[H_\text{GN}(m)]
# 
# Unlike Laplace, VI averages the Hessian over the distribution :math:`q`
# rather than evaluating it at a single point.
# 
# Step 1: Define CoFI ``BaseProblem``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

inv_problem = BaseProblem()
inv_problem.set_forward(forward)
inv_problem.set_jacobian(jacobian)
inv_problem.set_data(d_obs)
inv_problem.set_data_covariance_inv(Cd_inv)
inv_problem.set_initial_model(m_prior.copy())

######################################################################
#


######################################################################
# Step 2: Define CoFI ``InversionOptions``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The VI solver requires ``prior_precision`` as a solver option (the
# sparse precision matrix :math:`Q_p`). Other options control the VI loop.
# 

inv_options_gvi = InversionOptions()
inv_options_gvi.set_tool("cofi.gaussian_vi")
inv_options_gvi.set_params(
    prior_precision=Qp,
    num_iterations=120,
    num_samples=8,
    learning_rate_mean=0.02,
    learning_rate_precision=0.05,
    verbose=False,
    random_seed=42,
)

######################################################################
#


######################################################################
# Step 3: Run the inversion
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

inv_gvi = Inversion(inv_problem, inv_options_gvi)
result_gvi = inv_gvi.run()

print(f"Result type: {type(result_gvi).__name__}")
print(f"Success: {result_gvi.success}")
print(f"Iterations: {result_gvi.num_iterations}")

######################################################################
#

# ELBO convergence
plt.figure(figsize=(6, 3))
plt.plot(result_gvi.elbo_history, "b-", lw=1)
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("Gaussian VI convergence")
plt.tight_layout()
plt.show()

######################################################################
#

# Extract Gaussian VI marginal stds
Omega_gvi = result_gvi.precision
mu_gvi = result_gvi.model
gvi_std = np.zeros(N)
for i in range(N):
    e = np.zeros(N); e[i] = 1.0
    if sparse.issparse(Omega_gvi):
        gvi_std[i] = np.sqrt(splinalg.spsolve(Omega_gvi, e)[i])
    else:
        gvi_std[i] = np.sqrt(np.linalg.solve(Omega_gvi, e)[i])

######################################################################
#


######################################################################
# Comparison: Laplace vs Gaussian VI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, mu, std, title, colour in [
    (axes[0], m_map, laplace_std, "Laplace approximation", "C1"),
    (axes[1], mu_gvi, gvi_std, "Gaussian VI", "C0"),
]:
    ax.fill_between(x_grid, mu - 2*std, mu + 2*std, alpha=0.3, color=colour,
                    label=r"$\mu \pm 2\sigma$")
    ax.plot(x_grid, mu, f"{colour}-", lw=1.5, label="mean")
    ax.plot(x_grid, m_true, "k--", lw=1, label="true")
    ax.set(xlabel="x", ylabel="m", title=title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

######################################################################
#


######################################################################
# 3. Sinh-Arcsinh Flow with CoFI
# ------------------------------
# 
# Gaussian VI is symmetric by construction. For our nonlinear problem, the
# true posterior has skewed marginals (long tails toward negative values
# where data provides little constraint).
# 
# The **sinh-arcsinh flow** applies an elementwise transform:
# 
# .. math:: m_i = T(z_i) = \sinh(a_i \cdot \mathrm{arcsinh}(z_i) + b_i)
# 
# where :math:`z \sim \mathcal{N}(\mu, \Omega^{-1})` is the base Gaussian.
# Parameters :math:`a_i` control tail weight and :math:`b_i` control
# skewness, at :math:`O(N)` extra cost.
# 
# We enable this by setting ``enable_flow=True``.
# 

inv_options_sas = InversionOptions()
inv_options_sas.set_tool("cofi.gaussian_vi")
inv_options_sas.set_params(
    prior_precision=Qp,
    enable_flow=True,
    flow_num_iterations=300,
    flow_num_samples=32,
    learning_rate_mean=0.02,
    learning_rate_precision=0.05,
    flow_learning_rate=0.003,
    verbose=False,
    random_seed=42,
)

######################################################################
#

inv_sas = Inversion(inv_problem, inv_options_sas)
result_sas = inv_sas.run()

print(f"Success: {result_sas.success}")
print(f"Flow params a range: [{result_sas.flow_params['a'].min():.3f}, {result_sas.flow_params['a'].max():.3f}]")
print(f"Flow params b range: [{result_sas.flow_params['b'].min():.3f}, {result_sas.flow_params['b'].max():.3f}]")

######################################################################
#

# SAS flow ELBO convergence
plt.figure(figsize=(6, 3))
plt.plot(result_sas.elbo_history, "r-", lw=1)
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("Sinh-arcsinh flow VI convergence")
plt.tight_layout()
plt.show()

######################################################################
#


######################################################################
# 4. Post-Sampling Analysis with arviz
# ------------------------------------
# 
# The VI result is a ``SamplingResult``, so we can use ``to_arviz()`` to
# generate posterior samples and build an ``InferenceData`` object.
# 

import arviz as az

# Generate posterior samples from the Gaussian VI result
idata_gvi = result_gvi.to_arviz(num_samples=500)
print(f"Gaussian VI InferenceData: {idata_gvi}")
print(f"Posterior shape: {idata_gvi.posterior['model'].shape}")

######################################################################
#

# Generate posterior samples from the SAS flow result
idata_sas = result_sas.to_arviz(num_samples=500)
print(f"SAS flow InferenceData: {idata_sas}")
print(f"Posterior shape: {idata_sas.posterior['model'].shape}")

######################################################################
#


######################################################################
# Marginal comparison: selected parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let’s compare the marginal distributions from all three methods at a few
# selected parameter indices.
# 

# Draw samples for comparison
gvi_samples = idata_gvi.posterior["model"].values[0]  # (num_samples, N)
sas_samples = idata_sas.posterior["model"].values[0]

# Laplace samples (for comparison)
L_map = np.linalg.cholesky(H_map.toarray())
laplace_samples = np.array([
    m_map + np.linalg.solve(L_map.T, np.random.randn(N))
    for _ in range(500)
])

# Plot marginals at selected indices
indices = [5, 15, 25, 35]
fig, axes = plt.subplots(1, len(indices), figsize=(14, 3))

for ax, idx in zip(axes, indices):
    ax.hist(laplace_samples[:, idx], bins=30, alpha=0.4, density=True, color="C1", label="Laplace")
    ax.hist(gvi_samples[:, idx], bins=30, alpha=0.4, density=True, color="C0", label="Gaussian VI")
    ax.hist(sas_samples[:, idx], bins=30, alpha=0.4, density=True, color="C3", label="SAS flow")
    ax.axvline(m_true[idx], color="k", ls="--", lw=1, label="true")
    ax.set_title(f"Parameter {idx} (x={x_grid[idx]:.2f})")
    ax.set_xlabel("m")
    if idx == indices[0]:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

######################################################################
#


######################################################################
# Flow parameters: where is skewness detected?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The flow parameters :math:`a_i` (tail weight) and :math:`b_i` (skewness)
# tell us where the posterior deviates from Gaussianity.
# 

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].plot(x_grid, result_sas.flow_params["a"], "C3-", lw=1.5)
axes[0].axhline(1.0, color="gray", ls="--", lw=0.8, label="identity (a=1)")
axes[0].set(xlabel="x", ylabel="a", title="Tail weight parameter")
axes[0].legend(fontsize=8)

axes[1].plot(x_grid, result_sas.flow_params["b"], "C3-", lw=1.5)
axes[1].axhline(0.0, color="gray", ls="--", lw=0.8, label="identity (b=0)")
axes[1].set(xlabel="x", ylabel="b", title="Skewness parameter")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

######################################################################
#


######################################################################
# Challenge: Experiment with VI Settings
# --------------------------------------
# 
# Try modifying the VI solver parameters and observe their effect:
# 
# 1. **Reduce ``num_samples`` to 1** – the ELBO trace becomes noisier, but
#    the damped updates act as a low-pass filter. Does the final result
#    change much?
# 2. **Increase ``learning_rate_mean``** – what happens if it’s too large?
# 3. **Increase ``num_iterations``** – does the ELBO continue to improve?
# 

# Your experiments here
# inv_options_exp = InversionOptions()
# inv_options_exp.set_tool("cofi.gaussian_vi")
# inv_options_exp.set_params(
#     prior_precision=Qp,
#     num_iterations=200,
#     num_samples=1,  # <-- try changing this
#     learning_rate_mean=0.02,
#     verbose=False,
#     random_seed=42,
# )
# result_exp = Inversion(inv_problem, inv_options_exp).run()
# plt.plot(result_exp.elbo_history)
# plt.show()

######################################################################
#


######################################################################
# Summary
# -------
# 
# +---------+------+---------------------------+-----------------------+
# | Method  | Cost | Captures correlations     | Captures skewness     |
# +=========+======+===========================+=======================+
# | Laplace | 1    | Yes (at MAP only)         | No                    |
# |         | MAP  |                           |                       |
# |         | s    |                           |                       |
# |         | olve |                           |                       |
# +---------+------+---------------------------+-----------------------+
# | G       | ~100 | Yes (averaged over q)     | No                    |
# | aussian | for  |                           |                       |
# | VI      | ward |                           |                       |
# |         | e    |                           |                       |
# |         | vals |                           |                       |
# +---------+------+---------------------------+-----------------------+
# | SAS     | ~300 | Yes (averaged over q)     | Yes (elementwise)     |
# | flow VI | for  |                           |                       |
# |         | ward |                           |                       |
# |         | e    |                           |                       |
# |         | vals |                           |                       |
# +---------+------+---------------------------+-----------------------+
# | Full    | ~10, | Yes (full)                | Yes (full)            |
# | MCMC    | 000+ |                           |                       |
# |         | for  |                           |                       |
# |         | ward |                           |                       |
# |         | e    |                           |                       |
# |         | vals |                           |                       |
# +---------+------+---------------------------+-----------------------+
# 
# VI offers a practical middle ground: richer uncertainty than Laplace at
# a fraction of MCMC cost.
# 


######################################################################
# --------------
# 
# Where to next?
# --------------
# 
# -  Gaussian VI at scale: the same solver on a nonlinear ambient noise
#    tomography problem with about 19,000 unknowns and ray tracing - `link
#    to
#    notebook <https://github.com/inlab-geo/cofi-examples/blob/main/tutorials/surface_wave_tomography/surface_wave_tomography_variational_inference_extension.ipynb>`__
# 


######################################################################
# Watermark
# ---------
# 

import cofi
print(f"cofi: {cofi.__version__}")
print(f"numpy: {np.__version__}")
import scipy; print(f"scipy: {scipy.__version__}")
print(f"arviz: {az.__version__}")

######################################################################
#



######################################################################
#
# sphinx_gallery_thumbnail_number = -1