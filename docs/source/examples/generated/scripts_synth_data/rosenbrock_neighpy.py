"""
Rosenbrock function with the Neighbourhood Algorithm
====================================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/metaheuristic_optimiser_tests/rosenbrock_neighpy.ipynb
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
# 1.1 Rosenbrock function 
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The Rosenbrock function is a classic optimisation test function given by
# 
# .. math:: f(x,y) = (a - x)^2 + b(y - x^2)^2
# 
# with :math:`a = 1` and :math:`b = 100`. The function has a global
# minimum at :math:`(a, a^2) = (1, 1)` where :math:`f(1,1) = 0`.
# 
# The minimum lies inside a narrow, curved valley. Finding the valley is
# easy, but converging to the minimum within it is challenging for many
# optimisers.
# 
# We work with the :math:`\log_{10}`-scaled version to avoid very
# large/small values:
# 
# .. math:: F(x,y) = \log_{10}\left[(a - x)^2 + b(y - x^2)^2\right]
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
# Because of the multi-phase nature of the NA, ``cofi`` gives you 3
# options: 1. Run both phases, using the Direct Search Phase samples as
# the initial ensemble for the Appraisal Phase (tool: ``neighpy``) 2. Only
# run the Direct Search Phase (tool: ``neighpyI``) 3. Only run the
# Appraisal Phase, using samples obtained from any method (tool:
# ``neighpyII``)
# 
# We will look at all of these options in this notebook.
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

# !pip install -U cofi

######################################################################
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi, voronoi_plot_2d

from cofi import BaseProblem, InversionOptions, Inversion

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

def rosenbrock(params, a=1, b=100):
    """Log10-scaled Rosenbrock function."""
    return np.log10((a - params[0])**2 + b * (params[1] - params[0]**2)**2)

######################################################################
#

# Contour plot of the Rosenbrock function
X, Y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-1, 3, 1000))
Z = np.log10((1 - X)**2 + 100 * (Y - X**2)**2)

plt.figure(figsize=(8, 6))
plt.imshow(Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto")
plt.colorbar(label="$\\log_{10}(f)$")
plt.scatter(1, 1, c="r", marker="x", s=100, zorder=10, label="Global minimum (1, 1)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rosenbrock function")
plt.legend(loc="upper center")

######################################################################
#

# Define the problem in CoFI
inv_problem = BaseProblem()
inv_problem.name = "Rosenbrock Function"
inv_problem.set_objective(rosenbrock)

inv_problem.summary()

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 4. Full NA (direct search + appraisal) 
# ---------------------------------------
# 
# Here we run both phases of the NA in one go using the ``neighpy`` tool.
# 

bounds = [(-2, 2), (-1, 3)]
n_initial_samples = 100
n_samples_per_iteration = 70
n_cells_to_resample = 10
n_iterations = 20
n_resample = 50000
n_walkers = 10

inv_options = InversionOptions()
inv_options.set_tool("neighpy")
inv_options.set_params(
    bounds=bounds,
    n_initial_samples=n_initial_samples,
    n_samples_per_iteration=n_samples_per_iteration,
    n_cells_to_resample=n_cells_to_resample,
    n_iterations=n_iterations,
    n_resample=n_resample,
    n_walkers=n_walkers,
)

######################################################################
#

inv = Inversion(inv_problem, inv_options)
inv_result = inv.run()
inv_result.summary()

######################################################################
#

best = inv_result.model
ds_samples = inv_result.direct_search_samples
ds_objectives = inv_result.direct_search_objectives
appraisal_samples = inv_result.appraisal_samples

print(f"Best model: x={best[0]:.4f}, y={best[1]:.4f}")
print(f"True minimum: x=1, y=1")

######################################################################
#


######################################################################
# Voronoi cells from direct search
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

fig = voronoi_plot_2d(
    Voronoi(ds_samples), show_vertices=False, line_width=0.5, line_colors="w"
)
ax = fig.gca()
im = ax.imshow(Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto")
fig.colorbar(im)
_truth = ax.scatter(1, 1, c="r", marker="x", s=100, zorder=10, label="True minimum")
_best = ax.scatter(*best, c="k", marker="+", s=100, zorder=10, label="Best sample (NA-I)")
_voronoi = Line2D([0], [0], marker="o", label="Voronoi samples (NA-I)", markersize=5, linewidth=0)
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(handles=[_truth, _best, _voronoi], framealpha=1, edgecolor="black")
ax.set_title("NA Direct Search - Voronoi cells on Rosenbrock function")

######################################################################
#


######################################################################
# Appraisal samples
# ~~~~~~~~~~~~~~~~~
# 

fig = voronoi_plot_2d(
    Voronoi(ds_samples), show_vertices=False, line_width=0.5, line_colors="w"
)
ax = fig.gca()
im = ax.imshow(Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto")
fig.colorbar(im)
_truth = ax.scatter(1, 1, c="r", marker="x", s=100, zorder=2, label="True minimum")
_best = ax.scatter(*best, c="k", marker="+", s=100, zorder=2, label="Best sample (NA-I)")
_resample = ax.scatter(
    *appraisal_samples.T, s=0.5, c="grey", zorder=0, label="Resampled points (NA-II)"
)
_voronoi = Line2D([0], [0], marker="o", label="Voronoi samples (NA-I)", markersize=5, linewidth=0)
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(
    handles=[_truth, _best, _voronoi, _resample], framealpha=1, edgecolor="black"
)
ax.set_title("NA-I and NA-II on Rosenbrock function")

######################################################################
#


######################################################################
# Conditional distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

fig, axs = plt.subplots(
    2, 2,
    gridspec_kw=dict(height_ratios=[1, 5], width_ratios=[5, 1]),
    figsize=(7, 7),
    tight_layout=True,
)
axs[0, 1].set_visible(False)

# Conditional posterior samples p(x|y=1)
y_hist_edges = np.histogram_bin_edges(appraisal_samples[:, 1], bins=50, range=(-1, 3))
best_ind_y = np.digitize(1, y_hist_edges)
x_given_y = appraisal_samples[
    (appraisal_samples[:, 1] > y_hist_edges[best_ind_y - 1])
    & (appraisal_samples[:, 1] < y_hist_edges[best_ind_y]),
    0,
]
axs[0, 0].hist(x_given_y, bins=50, orientation="vertical", color="grey")
axs[0, 0].axvline(1, c="r", ls="--", lw=1)
axs[0, 0].set_xlim(-2, 2)
axs[0, 0].set_xticks([])
axs[0, 0].text(
    0.05, 0.9, "p(x|y=1)",
    transform=axs[0, 0].transAxes, fontsize=12, verticalalignment="top",
)

# Conditional posterior samples p(y|x=1)
x_hist_edges = np.histogram_bin_edges(appraisal_samples[:, 0], bins=50, range=(-2, 2))
best_ind_x = np.digitize(1, x_hist_edges)
y_given_x = appraisal_samples[
    (appraisal_samples[:, 0] > x_hist_edges[best_ind_x - 1])
    & (appraisal_samples[:, 0] < x_hist_edges[best_ind_x]),
    1,
]
axs[1, 1].hist(y_given_x, bins=50, orientation="horizontal", color="grey")
axs[1, 1].axhline(1, c="r", ls="--", lw=1)
axs[1, 1].set_ylim(-1, 3)
axs[1, 1].set_yticks([])
axs[1, 1].text(
    0.05, 0.9, "p(y|x=1)",
    transform=axs[1, 1].transAxes, fontsize=12, verticalalignment="bottom",
)

# Full posterior samples
im = axs[1, 0].imshow(
    Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto"
)
_truth = axs[1, 0].scatter(
    1, 1, c="r", marker="x", s=50, zorder=2, label="True minimum"
)
_best = axs[1, 0].scatter(
    *best, c="k", marker="+", s=50, zorder=2, label="Best sample (NA-I)"
)
_resample = axs[1, 0].scatter(
    *appraisal_samples.T, s=0.5, c="grey", zorder=0, label="Resampled points (NA-II)"
)
axs[1, 0].set_xlim(-2, 2)
axs[1, 0].set_ylim(-1, 3)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
axs[1, 0].legend(
    handles=[_truth, _best, _resample], framealpha=1, edgecolor="black",
)

fig.suptitle("Conditional distributions from NA-II on Rosenbrock function")

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 5. Direct search only 
# ----------------------
# 
# If you only want to optimise the objective function and find a point
# estimate, you can run just the direct search phase using the
# ``neighpyI`` tool.
# 

inv_options_ds = InversionOptions()
inv_options_ds.set_tool("neighpyI")
inv_options_ds.set_params(
    bounds=bounds,
    n_initial_samples=n_initial_samples,
    n_samples_per_iteration=n_samples_per_iteration,
    n_cells_to_resample=n_cells_to_resample,
    n_iterations=n_iterations,
)

inv_ds = Inversion(inv_problem, inv_options_ds)
inv_result_ds = inv_ds.run()
inv_result_ds.summary()

######################################################################
#

best_ds = inv_result_ds.model
ds_samples_only = inv_result_ds.samples
ds_objectives_only = inv_result_ds.objectives

print(f"Best model: x={best_ds[0]:.4f}, y={best_ds[1]:.4f}")

######################################################################
#

fig = voronoi_plot_2d(
    Voronoi(ds_samples_only), show_vertices=False, line_width=0.5, line_colors="w"
)
ax = fig.gca()
im = ax.imshow(Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto")
fig.colorbar(im)
_truth = ax.scatter(1, 1, c="r", marker="x", s=100, zorder=10, label="True minimum")
_best = ax.scatter(*best_ds, c="k", marker="+", s=100, zorder=10, label="Best sample (NA-I)")
_voronoi = Line2D([0], [0], marker="o", label="Voronoi samples (NA-I)", markersize=5, linewidth=0)
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(handles=[_truth, _best, _voronoi], framealpha=1, edgecolor="black")
ax.set_title("NA-I Direct Search on Rosenbrock function")

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 6. Appraisal only 
# ------------------
# 
# The appraisal phase is implemented in the ``neighpyII`` tool. It takes a
# set of samples and their corresponding log posterior probability
# density.
# 
# Note that the direct search *minimises* an objective, but the appraisal
# *maximises* a posterior. So we pass ``-objectives`` as ``log_ppd``.
# 

inv_options_app = InversionOptions()
inv_options_app.set_tool("neighpyII")
inv_options_app.set_params(
    bounds=bounds,
    initial_ensemble=ds_samples_only,
    log_ppd=-ds_objectives_only,
    n_resample=n_resample,
    n_walkers=n_walkers,
)

inv_app = Inversion(inv_problem, inv_options_app)
inv_result_app = inv_app.run()
inv_result_app.summary()

######################################################################
#

appraisal_samples_only = inv_result_app.new_samples

######################################################################
#


######################################################################
# Appraisal samples
# ~~~~~~~~~~~~~~~~~
# 

fig = voronoi_plot_2d(
    Voronoi(ds_samples_only), show_vertices=False, line_width=0.5, line_colors="w"
)
ax = fig.gca()
im = ax.imshow(Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto")
fig.colorbar(im)
_truth = ax.scatter(1, 1, c="r", marker="x", s=100, zorder=2, label="True minimum")
_best = ax.scatter(*best_ds, c="k", marker="+", s=100, zorder=2, label="Best sample (NA-I)")
_resample = ax.scatter(
    *appraisal_samples_only.T, s=0.5, c="grey", zorder=0, label="Resampled points (NA-II)"
)
_voronoi = Line2D([0], [0], marker="o", label="Voronoi samples (NA-I)", markersize=5, linewidth=0)
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(
    handles=[_truth, _best, _voronoi, _resample], framealpha=1, edgecolor="black"
)
ax.set_title("NA-II Appraisal on Rosenbrock function")

######################################################################
#


######################################################################
# Conditional distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

fig, axs = plt.subplots(
    2, 2,
    gridspec_kw=dict(height_ratios=[1, 5], width_ratios=[5, 1]),
    figsize=(7, 7),
    tight_layout=True,
)
axs[0, 1].set_visible(False)

# Conditional posterior samples p(x|y=1)
y_hist_edges = np.histogram_bin_edges(appraisal_samples_only[:, 1], bins=50, range=(-1, 3))
best_ind_y = np.digitize(1, y_hist_edges)
x_given_y = appraisal_samples_only[
    (appraisal_samples_only[:, 1] > y_hist_edges[best_ind_y - 1])
    & (appraisal_samples_only[:, 1] < y_hist_edges[best_ind_y]),
    0,
]
axs[0, 0].hist(x_given_y, bins=50, orientation="vertical", color="grey")
axs[0, 0].axvline(1, c="r", ls="--", lw=1)
axs[0, 0].set_xlim(-2, 2)
axs[0, 0].set_xticks([])
axs[0, 0].text(
    0.05, 0.9, "p(x|y=1)",
    transform=axs[0, 0].transAxes, fontsize=12, verticalalignment="top",
)

# Conditional posterior samples p(y|x=1)
x_hist_edges = np.histogram_bin_edges(appraisal_samples_only[:, 0], bins=50, range=(-2, 2))
best_ind_x = np.digitize(1, x_hist_edges)
y_given_x = appraisal_samples_only[
    (appraisal_samples_only[:, 0] > x_hist_edges[best_ind_x - 1])
    & (appraisal_samples_only[:, 0] < x_hist_edges[best_ind_x]),
    1,
]
axs[1, 1].hist(y_given_x, bins=50, orientation="horizontal", color="grey")
axs[1, 1].axhline(1, c="r", ls="--", lw=1)
axs[1, 1].set_ylim(-1, 3)
axs[1, 1].set_yticks([])
axs[1, 1].text(
    0.05, 0.9, "p(y|x=1)",
    transform=axs[1, 1].transAxes, fontsize=12, verticalalignment="bottom",
)

# Full posterior samples
im = axs[1, 0].imshow(
    Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto"
)
_truth = axs[1, 0].scatter(
    1, 1, c="r", marker="x", s=50, zorder=2, label="True minimum"
)
_best = axs[1, 0].scatter(
    *best_ds, c="k", marker="+", s=50, zorder=2, label="Best sample (NA-I)"
)
_resample = axs[1, 0].scatter(
    *appraisal_samples_only.T, s=0.5, c="grey", zorder=0, label="Resampled points (NA-II)"
)
axs[1, 0].set_xlim(-2, 2)
axs[1, 0].set_ylim(-1, 3)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
axs[1, 0].legend(
    handles=[_truth, _best, _resample], framealpha=1, edgecolor="black",
)

fig.suptitle("Conditional distributions from NA-II on Rosenbrock function")

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
# sphinx_gallery_thumbnail_number = -1