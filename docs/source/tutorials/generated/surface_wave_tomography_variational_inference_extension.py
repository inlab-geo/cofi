"""
Gaussian variational inference for surface wave tomography
==========================================================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/tutorials/surface_wave_tomography/surface_wave_tomography_variational_inference_extension.ipynb
# 


######################################################################
#    **Work in progress:** this notebook is still under development, both
#    as a tutorial and as an example of the method. Its text, code and
#    results may change.
# 


######################################################################
# --------------
# 
# What we do in this notebook
# ---------------------------
# 
# Here we apply **Gaussian variational inference (VI)** to the nonlinear
# ambient noise tomography problem of the `ambient noise tomography
# tutorial <surface_wave_tomography.ipynb>`__, using CoFI’s
# ``cofi.gaussian_vi`` solver:
# 
# -  travel times are predicted with fast-marching ray tracing
#    (``pyfm2d``), which bends rays through the current velocity model;
# -  the prior is a sparse SPDE Matérn operator, and the data weight is
#    set through the data covariance in a two-pass scheme;
# -  the solver finds the MAP model, then refines a Gaussian approximation
#    of the posterior with natural-gradient updates, giving a mean model
#    and a pixel-wise uncertainty map for about 19,000 unknowns.
# 
# --------------
# 
# Learning outcomes
# -----------------
# 
# -  See how a large nonlinear inverse problem is set up for CoFI’s
#    ``variational inference`` solving method, with sparse prior and
#    data-covariance matrices.
# -  Understand why natural-gradient updates on a sparse precision matrix
#    make Gaussian VI feasible at this scale.
# -  Learn how to calibrate the data-versus-prior balance through the data
#    covariance, and how to read an ELBO curve for VI started at the MAP.
# -  Compare what Gaussian VI delivers (a mean and a covariance from
#    optimisation) with what sampling delivers.
# 

# Environment setup (uncomment code below)

# cofi.gaussian_vi is not in a released cofi yet: install cofi from its main branch.
# Switch to a released `cofi` (>=0.2.13) once it is published.
# scikit-sparse needs the SuiteSparse (CHOLMOD) library: `brew install suite-sparse` (macOS)
# or `sudo apt-get install libsuitesparse-dev` (Debian/Ubuntu/Colab).

# !pip install -U "cofi @ git+https://github.com/inlab-geo/cofi@main" "scikit-sparse>=0.5" cartopy cmcrameri tqdm git+https://github.com/inlab-geo/pyfm2d
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/tutorials/surface_wave_tomography

######################################################################
#


######################################################################
# Overview
# --------
# 
# This notebook applies **Gaussian Variational Inference** (VI) to ambient
# noise surface wave tomography across Australia, using CoFI’s
# ``cofi.gaussian_vi`` solver. This extends the `ambient noise tomography
# tutorial <surface_wave_tomography.ipynb>`__. Gaussian VI provides a
# scalable approximation to the posterior distribution by fitting a
# multivariate Gaussian, yielding both a posterior mean and a full
# covariance (uncertainty) estimate.
# 
#    New to variational inference? The `variational inference
#    tutorial <../variational_inference/variational_inference.ipynb>`__
#    introduces it on a small synthetic problem, comparing the Laplace
#    approximation, Gaussian VI and a sinh-arcsinh flow.
# 
# Method
# ~~~~~~
# 
# CoFI’s ``cofi.gaussian_vi`` solver implements natural-gradient Gaussian
# VI in three phases:
# 
# 1. **Phase 1 — MAP initialisation**: Gauss-Newton iteration finds the
#    maximum a posteriori estimate, providing a good starting point.
# 2. **Phase 2 — Gaussian VI**: Natural-gradient updates on the mean and
#    precision of a Gaussian approximation to the posterior, tracked via
#    the Evidence Lower Bound (ELBO).
# 3. **Phase 3 — Optional normalising flow**: A sinh-arcsinh
#    transformation (Jones & Pewsey, 2009) can capture non-Gaussian
#    features (skewness, heavy tails) in the posterior.
# 
# The natural-gradient updates behind Phases 1-2 are those of the Bayesian
# learning rule (Khan & Lin, 2017; Khan & Rue, 2023). All matrices remain
# sparse throughout (the solver uses CHOLMOD for sparse Cholesky
# factorisation; Chen et al., 2008), making this feasible for the
# ~19,000-parameter grid.
# 
# Prior and data covariance
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The prior precision :math:`\mathbf{Q} = \mathbf{R}^\top\mathbf{R}` is
# built from an SPDE Matérn operator (Lindgren et al., 2011) with
# physically meaningful parameters:
# 
# -  **:math:`\sigma`** (prior std dev) encodes how much slowness can
#    deviate from the background
# -  **:math:`\ell`** (correlation length) controls spatial smoothness
# 
# The data-vs-prior balance is controlled through the **data covariance**
# :math:`\mathbf{C}_d`, not by scaling Q. This keeps the prior physically
# interpretable: :math:`\sigma` means what it says, and tightening
# :math:`\mathbf{C}_d` (reducing assumed data noise) increases the data’s
# relative weight.
# 

# Third-party numerical libraries
import numpy as np
import scipy
import scipy.sparse

# Visualisation
import matplotlib.pyplot as plt
import cmcrameri.cm as scm

# Cartopy for map projections
import cartopy.crs as ccrs

# Fast-marching ray tracing for the forward model
import pyfm2d

# Inversion framework
import cofi
from cofi import BaseProblem, InversionOptions, Inversion

######################################################################
#


######################################################################
# Why natural gradients, and why sparsity is everything
# -----------------------------------------------------
# 
# VI replaces sampling with optimisation: choose
# :math:`q(m) = \mathcal{N}(\mu,
# \mathbf{\Omega}^{-1})` and maximise the ELBO, which is the same as
# minimising :math:`\mathrm{KL}(q\,\|\,p(m \mid d))`.
# 
# Not by ordinary gradient ascent, though. An ordinary gradient depends on
# the coordinates, so parameterising :math:`q` by its precision
# :math:`\mathbf{\Omega}` or by its covariance
# :math:`\mathbf{\Sigma} = \mathbf{\Omega}^{-1}` would give different
# steps for the same distribution. The **natural gradient** — the gradient
# in the space of distributions — is invariant to that choice (Khan & Rue,
# 2023), and for a Gaussian it is closed-form:
# 
# .. math::
# 
#    \mathbf{\Omega} \leftarrow (1-\rho_\Omega)\,\mathbf{\Omega} + \rho_\Omega\,
#    \mathbb{E}_q\!\left[\mathbf{J}^\top \mathbf{C}_d^{-1} \mathbf{J} + \mathbf{Q}\right],
#    \qquad
#    \mu \leftarrow \mu - \rho_\mu\,\mathbf{\Omega}^{-1}\,
#    \mathbb{E}_q\!\left[\nabla_m \mathcal{L}(m)\right]
# 
# with :math:`\mathcal{L} = -\log p(d, m)`, the objective Gauss-Newton
# already minimises for the MAP. The mean step is a Newton step
# preconditioned by the current uncertainty; the precision step is a
# **convex combination**, not a gradient step, so it cannot overshoot into
# a non-positive-definite :math:`\mathbf{\Omega}`. The only new ingredient
# over MAP is :math:`\mathbb{E}_q[\cdot]` — gradient and Hessian averaged
# over samples of :math:`q` rather than evaluated at one point. That
# average is what turns a mode into a distribution, and it is why every VI
# iteration costs ``num_samples`` ray traces.
# 
# **Both updates need :math:`\mathbf{\Omega}^{-1}`** — a linear solve for
# the mean, a Cholesky factor to draw the samples — and that is affordable
# only because :math:`\mathbf{\Omega}` stays sparse. On this grid
# (:math:`N = 19{,}456` cells, 13,299 rays):
# 
# -  :math:`\mathbf{Q} = \mathbf{R}^\top\mathbf{R}` is a local SPDE
#    stencil — 12.9 nonzeros per row.
# -  :math:`\mathbf{J}` is sparse because a ray only samples the cells it
#    crosses — 67 per row, 0.3% of the matrix.
# -  :math:`\mathbf{J}^\top\mathbf{C}_d^{-1}\mathbf{J}` couples every pair
#    of cells that share a ray, so it is denser than :math:`\mathbf{J}`
#    but still structured: :math:`\mathbf{\Omega}` lands at 8.2M nonzeros,
#    2.2% dense, ~100 MB. CHOLMOD factorises that in ~2 s with only
#    1.6\ :math:`\times` fill-in — negligible beside the ray trace that
#    produced it.
# 
# (Measured at :math:`m_0`. The forward model is nonlinear, so every
# sampled model bends its rays a little differently and
# :math:`\mathbf{\Omega}` ends up carrying the union of their patterns:
# one draw from :math:`\mathcal{N}(m_0, \mathbf{\Omega}^{-1})` adds ~10%
# more nonzeros, taking it from 2.2% to 2.4% dense. That growth tracks the
# ray paths, not the algebra — the update does no elimination, so nothing
# fills in the way forming :math:`\mathbf{\Omega}^{-1}` would.)
# 
# Parameterising by the covariance instead would collapse the scheme:
# :math:`\mathbf{\Sigma} = \mathbf{\Omega}^{-1}` is dense even when
# :math:`\mathbf{\Omega}` is not — 3.0 GB to store at this grid size, and
# :math:`O(N^3)` to factorise, every iteration. Sparsity is not an
# optimisation here; it is what makes the method run at all.
# 


######################################################################
# Fast or full run
# ~~~~~~~~~~~~~~~~
# 
# The notebook can run in two modes, selected by ``RUN_MODE`` in the next
# cell. The default, ``"fast"``, uses a 1% random subset of the rays so
# that the many ray traces needed by VI finish in minutes. Set
# ``RUN_MODE = "full"`` for all 13,299 station pairs: the solver then
# needs about 76 ray traces of roughly a minute each (about 80 minutes in
# the run behind the published results; the fast mode took about 11
# minutes on a 10-core laptop).
# 
# ========= ================== ==================================
# Setting   ``fast`` (default) ``full``
# ========= ================== ==================================
# Rays used 1%                 100% (13,299 unique station pairs)
# ========= ================== ==================================
# 
# ..
# 
#    **Note:** the numbers and interpretive comments in this notebook
#    describe the **full** run. The fast run follows exactly the same
#    workflow, but with only 1% of the rays the posterior mean is a much
#    smoother, less detailed map, and the uncertainty map is nearly
#    uniform: velocity is 1/slowness, so a few poorly constrained pixels
#    get very large standard deviations and stretch the colour scale. Use
#    ``RUN_MODE = "full"`` for the structured uncertainty map described
#    below.
# 

# Choose "fast" (minutes) or "full" (about 80 minutes)
RUN_MODE = "fast"

presets = {
    "fast": {"ray_percentage": 1},
    "full": {"ray_percentage": 100},
}[RUN_MODE]
print(f"Run mode: {RUN_MODE}")
for k, v in presets.items():
    print(f"  {k}: {v}")

######################################################################
#


######################################################################
# Data
# ----
# 
# The dataset comprises 5-second period Rayleigh wave phase velocity
# measurements from ambient noise cross-correlations between ~1,100
# seismic stations across Australia (Magrini et al. 2023). The raw dataset
# has 15,661 measurements, which reduce to 13,299 unique inter-station
# travel time observations after averaging duplicate station pairs (full
# mode).
# 

ray_percentage = presets['ray_percentage']
random_seed = 42

# Load the shared dataset: slowness observations and the coordinates of the two stations of each pair
data = np.load("../../data/sw_tomography/sw_tomography.npz", allow_pickle=True)
all_observations = data["slowness"]                   # (15661,) slowness in s/m
station_coords = data["station_coords"]               # (15661, 4) [lat1, lon1, lat2, lon2]

# Unique stations [lat, lon], and the indices [sta1_idx, sta2_idx] of each observation's station pair
sta1, sta2 = station_coords[:, :2], station_coords[:, 2:]
stations, station_index = np.unique(np.vstack([sta1, sta2]), axis=0, return_inverse=True)
station_index = station_index.ravel()
all_station_pairs = np.column_stack([station_index[:len(sta1)], station_index[len(sta1):]]).astype(np.int32)

n_total = len(all_station_pairs)

# Apply ray subset selection
if ray_percentage < 100:
    rng = np.random.default_rng(random_seed)
    n_selected = max(1, int(n_total * ray_percentage / 100))
    selected_indices = rng.choice(n_total, size=n_selected, replace=False)
    selected_indices.sort()  # Keep order for reproducibility
    observations = all_observations[selected_indices]
    station_pairs = all_station_pairs[selected_indices]
    print(f"Ray subset: {ray_percentage}% selected ({n_selected:,} of {n_total:,} rays, seed={random_seed})")
else:
    observations = all_observations
    station_pairs = all_station_pairs
    print(f"Using all {n_total:,} rays")

# Remove duplicate station pairs (average observations for duplicates)
unique_pairs, inverse_indices = np.unique(station_pairs, axis=0, return_inverse=True)
if len(unique_pairs) < len(station_pairs):
    # Average observations for duplicate pairs
    unique_obs = np.zeros(len(unique_pairs))
    counts = np.zeros(len(unique_pairs))
    for obs_idx, unique_idx in enumerate(inverse_indices):
        unique_obs[unique_idx] += observations[obs_idx]
        counts[unique_idx] += 1
    unique_obs /= counts
    print(f"  Removed {len(station_pairs) - len(unique_pairs)} duplicate pairs (averaged observations)")
    station_pairs = unique_pairs
    observations = unique_obs

######################################################################
#

# Define grid parameters
grid_resolution = 0.3  # degrees
latmin, latmax = -46.2, -8.1
lonmin, lonmax = 110.9, 156.2

######################################################################
#

# Convert observed slowness to travel times using great circle distances
# observations is slowness in s/m; convert to travel times in seconds
R_earth = 6371000.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great circle distance in meters."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))

distances = np.array([
    haversine_distance(
        stations[s1, 0], stations[s1, 1],
        stations[s2, 0], stations[s2, 1]
    )
    for s1, s2 in station_pairs
])

d_obs_tt = observations * distances
print(f"Observed travel times computed from real data:")
print(f"  Distance range: [{distances.min()/1000:.1f}, {distances.max()/1000:.1f}] km")
print(f"  Travel time range: [{d_obs_tt.min():.1f}, {d_obs_tt.max():.1f}] s")

######################################################################
#


######################################################################
# Forward model: pyfm2d ray tracing
# ---------------------------------
# 
# Travel times are computed using **pyfm2d**, which solves the eikonal
# equation via a fast marching method (Rawlinson & Sambridge, 2004).
# Unlike the linear (great circle) approximation, pyfm2d accounts for ray
# bending through the velocity structure. The Jacobian (Fréchet
# derivative) is computed alongside the travel times for use in the
# Gauss-Newton updates within VI.
# 

# Grid extent for pyfm2d [lonmin, lonmax, latmin, latmax]
# Matches the seislib grid exactly — all stations are at least 2.7° inside
# these bounds, so no additional buffer is needed for pyfm2d.
fm2d_extent = [lonmin, lonmax, latmin, latmax]

# Compute grid shape from extent and resolution
# pyfm2d expects velocity as (nx, ny) = (n_lon, n_lat)
n_lon = int(np.round((fm2d_extent[1] - fm2d_extent[0]) / grid_resolution)) + 1
n_lat = int(np.round((fm2d_extent[3] - fm2d_extent[2]) / grid_resolution)) + 1
fm2d_grid_shape = (n_lon, n_lat)

# Note: the source-receiver associations below cover the same geometry as the
# linear inversion, but seislib handles that internally (inside compile_coefficients)
# and does not expose it in a form pyfm2d can use. pyfm2d requires an explicit
# associations matrix, local index mappings, and a reorder array, so the
# geometry must be reconstructed here from the same raw station_pairs data.

# Get unique sources and receivers from (possibly subsetted) station_pairs
unique_src_idx = np.unique(station_pairs[:, 0])
unique_rec_idx = np.unique(station_pairs[:, 1])

# Create index mappings
src_map = {sta_idx: local_idx for local_idx, sta_idx in enumerate(unique_src_idx)}
rec_map = {sta_idx: local_idx for local_idx, sta_idx in enumerate(unique_rec_idx)}

# Build associations matrix
n_src = len(unique_src_idx)
n_rec = len(unique_rec_idx)
fm2d_associations = np.zeros((n_rec, n_src), dtype=np.int32)
for src_idx, rec_idx in station_pairs:
    fm2d_associations[rec_map[rec_idx], src_map[src_idx]] = 1

# Sources and receivers in [lon, lat] format for pyfm2d
fm2d_sources = stations[unique_src_idx][:, ::-1]
fm2d_receivers = stations[unique_rec_idx][:, ::-1]

# Build mapping from pyfm2d output order to station_pairs order
# pyfm2d with multithreading outputs: for each source, receivers with associations==1
# Map this back to station_pairs order
pyfm2d_to_pairs = {}  # (src_local, rec_local) -> index in station_pairs
for pair_idx, (src_idx, rec_idx) in enumerate(station_pairs):
    src_local = src_map[src_idx]
    rec_local = rec_map[rec_idx]
    pyfm2d_to_pairs[(src_local, rec_local)] = pair_idx

# Build reorder indices: pyfm2d output order -> station_pairs order
# pyfm2d outputs in source-major order: for each source, list receivers
fm2d_reorder = []
for src_local in range(n_src):
    for rec_local in range(n_rec):
        if fm2d_associations[rec_local, src_local] == 1:
            fm2d_reorder.append(pyfm2d_to_pairs[(src_local, rec_local)])
fm2d_reorder = np.array(fm2d_reorder)

print(f"pyfm2d configuration:")
print(f"  Grid shape: {fm2d_grid_shape} ({n_lat} lat x {n_lon} lon = {n_lat * n_lon:,} cells)")
print(f"  Sources: {n_src}, Receivers: {n_rec}, Pairs: {len(station_pairs):,}")

######################################################################
#

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from pyfm2d_worker import compute_single_source  # helper module next to this notebook
import os

# Determine number of parallel workers (use all available CPUs)
n_workers = os.cpu_count() or 1

def forward_pyfm2d(slowness_model, return_jacobian=False, progress=True):
    """Compute travel times (and optionally Jacobian) using pyfm2d.

    Uses ProcessPoolExecutor for parallel execution over sources with progress reporting.
    Process-based parallelism is required because pyfm2d's C/Fortran code is not thread-safe.
    Workers are set to os.cpu_count() so the pool uses all available cores.

    Parameters
    ----------
    slowness_model : array-like
        Slowness model as a 1D array (n_cells,) in s/km.
    return_jacobian : bool
        If True, return (travel_times, jacobian).
    progress : bool or callable
        True shows a tqdm bar, False reports nothing. A callable is invoked
        as progress(done, n_tasks, stage) while the trace runs, with stage
        "trace" after each source completes and "assemble" once when the
        sparse Jacobian blocks are being stacked -- see `_traced` below,
        which uses it to print a throttled line-oriented progress log.

    Returns
    -------
    travel_times : ndarray
        Predicted travel times for each source-receiver pair, in station_pairs order.
    jacobian : sparse matrix (optional)
        Frechet derivative matrix (n_obs, n_cells).
    """
    # Convert slowness (s/km) to velocity (km/s) and reshape to grid
    velocity = 1.0 / slowness_model.reshape(fm2d_grid_shape)
    n_sources = fm2d_sources.shape[0]

    # Parallel execution over sources with progress bar
    # Using ProcessPoolExecutor because pyfm2d C/Fortran code is not thread-safe
    all_ttimes = {}
    all_frechet = {} if return_jacobian else None

    # Build task arguments as tuples for the worker function
    tasks = []
    for src_idx in range(n_sources):
        rec_mask = fm2d_associations[:, src_idx] == 1
        if not np.any(rec_mask):
            continue
        source = fm2d_sources[src_idx:src_idx+1]
        receivers = fm2d_receivers[rec_mask]
        tasks.append((src_idx, velocity, receivers, source, fm2d_extent, return_jacobian))

    on_progress = progress if callable(progress) else None

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(compute_single_source, task): task[0] for task in tasks}

        iterator = as_completed(futures)
        if progress is True:
            iterator = tqdm(iterator, total=len(futures), desc="Ray tracing", unit="src")

        _done = 0
        for future in iterator:
            src_idx, ttimes, frechet_data = future.result()
            if ttimes is not None:
                all_ttimes[src_idx] = ttimes
                if return_jacobian and frechet_data is not None:
                    all_frechet[src_idx] = frechet_data
            _done += 1
            if on_progress is not None:
                on_progress(_done, len(futures), "trace")

    # Stacking ~1100 sparse Frechet blocks and permuting them back into
    # station_pairs order runs after the last source is done, so it is the
    # one stretch a per-source counter cannot cover. Measured at ~0.02 s for
    # this grid (4.7M nnz), so the line marks the hand-off rather than a
    # wait -- but it is the boundary between "all sources traced" and
    # "forward returned", which is where a silent tail would show up.
    if on_progress is not None:
        on_progress(len(futures), len(futures), "assemble")

    # Combine results in source order and reorder to station_pairs order
    ttimes_list = []
    frechet_list = [] if return_jacobian else None

    for src_idx in range(n_sources):
        if src_idx in all_ttimes:
            ttimes_list.append(all_ttimes[src_idx])
            if return_jacobian:
                # Reconstruct sparse matrix from tuple (data, indices, indptr, shape)
                data, indices, indptr, shape = all_frechet[src_idx]
                frechet_mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
                frechet_list.append(frechet_mat)

    ttimes_combined = np.concatenate(ttimes_list)
    ttimes_out = np.empty(len(fm2d_reorder))
    ttimes_out[fm2d_reorder] = ttimes_combined

    if return_jacobian:
        frechet_combined = scipy.sparse.vstack(frechet_list)
        frechet_out = frechet_combined.tocsr()[np.argsort(fm2d_reorder), :]
        return ttimes_out, frechet_out
    return ttimes_out

######################################################################
#

# SPDE Matern v=1 parameters
ell = np.sqrt(2) * 5 * grid_resolution  # Matern length scale in degrees (practical range rho = 2*ell)
# Prior std dev of slowness perturbations (s/km).
# Sets max observed deviation (~0.11 s/km) at ~2.75 sigma, so the prior
# actively shapes the posterior rather than being vacuous.
sigma = 0.04

######################################################################
#

# Starting model: uniform 3 km/s (slowness in s/km for pyfm2d)
_n_params = fm2d_grid_shape[0] * fm2d_grid_shape[1]
m0_pyfm2d = np.full(_n_params, 1.0 / 3.0)  # 3 km/s -> 1/3 s/km

######################################################################
#

_n_lon, _n_lat = fm2d_grid_shape
_n_params = _n_lon * _n_lat

# SPDE Matern v=1 via cofi.utils. Uses Neumann (zero-flux) BCs, which
# avoids spurious boundary anchoring present in plain tridiagonal truncation.
regularization_pyfm2d = cofi.utils.SPDEMaternReg(
    model_shape=fm2d_grid_shape,
    ell=ell,
    sigma=sigma,
    grid_spacing=grid_resolution,
    reference_model=m0_pyfm2d,
)
_ell_km = ell * 111.0
reg_label = (
    "SPDE Matern v=1 via cofi.utils "
    f"(ell={ell:.2f} deg ~ {_ell_km:.0f} km, rho=2ell~{2*ell:.2f} deg, sigma={sigma} s/km)"
)

R_pyfm2d = regularization_pyfm2d.matrix
print(f"Prior regularization ({reg_label}):")
print(f"  Grid: {_n_lon} lon x {_n_lat} lat = {_n_params:,} cells")
print(f"  R shape: {R_pyfm2d.shape}")

######################################################################
#

#@title Plotting function (hidden)
# Set up map projection and boundaries for plotting
proj = ccrs.LambertConformal(
    central_longitude=135,
    central_latitude=-27,
    cutoff=80,
    standard_parallels=(-18, -36)
)
transform = ccrs.PlateCarree()
map_boundaries = [113, 153, -45, -8]

def plot_map(phase_velocity, title=None, grid_shape=None, extent=None, vmin=None, vmax=None):
    """Plot a phase-velocity map on the pyfm2d grid."""
    fig = plt.figure(figsize=(5, 6.5))
    ax = plt.subplot(111, projection=proj)
    ax.coastlines()
    _lon = np.linspace(extent[0], extent[1], grid_shape[0])
    _lat = np.linspace(extent[2], extent[3], grid_shape[1])
    _lon_grid, _lat_grid = np.meshgrid(_lon, _lat)
    img = ax.pcolormesh(_lon_grid, _lat_grid, phase_velocity.reshape(grid_shape).T,
                        cmap=scm.roma, transform=transform, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
    cb.set_label('Phase velocity [km/s]')
    ax.set_extent(map_boundaries, crs=transform)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig

######################################################################
#

cmin = 2.1  # minimum velocity for colourbar
cmax = 3.8  # maximum velocity for colourbar

######################################################################
#


######################################################################
# Gaussian Variational Inference
# ------------------------------
# 
# The prior precision :math:`\mathbf{Q} = \mathbf{R}^\top\mathbf{R}` uses
# the SPDE Matern operator directly — no additional scaling factor. The
# physically meaningful :math:`\sigma` ensures the prior is correctly
# calibrated.
# 
# The data covariance :math:`\mathbf{C}_d = \sigma_d^2 \mathbf{I}`
# controls how strongly the data pulls the solution away from the prior. A
# two-pass scheme is used:
# 
# 1. **Pass 1**: Estimate :math:`\sigma_d` from the residual at
#    :math:`m_0` (rough), run MAP-only to find a good solution.
# 2. **Pass 2**: Re-estimate :math:`\sigma_d` from the residual at the MAP
#    (accurate), then run the full VI with the updated
#    :math:`\mathbf{C}_d`.
# 
# This ensures :math:`\sigma_d` reflects the true data misfit, not the gap
# between :math:`m_0` and the data, giving a properly calibrated
# data-vs-prior balance.
# 

# Problem and options objects reconfigured by the two-pass VI cell below
inv_problem = BaseProblem()
inv_options = InversionOptions()

######################################################################
#

_n_obs = len(d_obs_tt)

# Forward and Jacobian for VI, both served from one logged, cached ray trace.
#
# Progress: the solver's `verbose=True` output is one line per MAP/VI
# iteration, but an iteration hides several full ray traces over ~1100
# sources, so without a line per trace the run looks hung for minutes.
# A trace is itself minutes long, so one line per trace is still not enough
# -- it only prints once the trace is over. `_progress` below therefore
# reports from inside the trace as sources complete.
#
# Cost: pyfm2d returns the travel times and the Frechet derivatives from
# the same eikonal solve -- the derivatives are accumulated while
# backtracking rays through a travel-time field that has already been
# computed (`rpaths2` in fm2dss.f90, gated on `cfd`). Measured on this grid
# they cost ~3% extra (268 ms/source vs 276 ms/source), so we always ask
# for both and cache them rather than tracing twice.
#
# cofi has no combined forward+Jacobian hook, so it evaluates the two
# separately at the same model: once per MAP iteration, once per VI sample.
# A MAP iteration's accepted line-search model is also bitwise the model
# the next iteration starts from. A single-entry cache therefore turns
# about half the ray traces into free lookups, with identical results.
import time as _time

_nfev = {"traces": 0, "hits": 0}
_cache = {"m": None, "d": None, "G": None}

# A single trace is minutes of work, so it also reports from inside itself:
# one line when it starts, one every _PROGRESS_EVERY seconds as sources
# complete, and one when the per-source Jacobians are stacked. Throttling on
# wall-clock rather than on a fraction of sources is deliberate -- sources
# carry very different receiver counts, so any fixed fraction still leaves
# long silent stretches, while a time budget bounds them by construction.
_PROGRESS_EVERY = 10.0
_prog = {"t0": 0.0, "last": 0.0}

def _progress(done, total, stage):
    """Throttled within-trace progress, so a trace is never silent for long."""
    _now = _time.perf_counter()
    if stage == "assemble":
        print(f"           assembling Jacobian from {total} sources...", flush=True)
        return
    if done < total and _now - _prog["last"] < _PROGRESS_EVERY:
        return
    _prog["last"] = _now
    _elapsed = _now - _prog["t0"]
    print(
        f"           {done:>5}/{total} sources  {_elapsed:6.1f}s"
        f"  eta {_elapsed * (total - done) / done:5.1f}s",
        flush=True,
    )

def _traced(m, tag=""):
    """Travel times and Jacobian at `m`, from cache or one logged ray trace."""
    if _cache["m"] is not None and np.array_equal(_cache["m"], m):
        _nfev["hits"] += 1
        return _cache["d"], _cache["G"]
    _nfev["traces"] += 1
    _label = f"    [{_nfev['traces']:>4}] trace{' ' + tag if tag else '':<5}"
    print(f"{_label} started", flush=True)
    _t0 = _time.perf_counter()
    _prog.update(t0=_t0, last=_t0)
    d, G = forward_pyfm2d(m, return_jacobian=True, progress=_progress)
    # copy the model: the solver reuses and mutates its own arrays
    _cache.update(m=np.array(m, copy=True), d=d, G=G)
    _rms = np.sqrt(np.mean((d_obs_tt - d) ** 2))
    print(
        f"{_label} {_time.perf_counter() - _t0:6.1f}s  rms={_rms:.4f}s"
        f"  ({_nfev['hits']} served from cache)",
        flush=True,
    )
    return d, G

def vi_forward(m):
    return _traced(m)[0]

def vi_jacobian(m):
    return _traced(m)[1]

# Prior precision Q = R.T @ R (SPDE Matern, sparse)
# No mu scaling — sigma is physically meaningful, C_d controls the balance
Q_prior = (R_pyfm2d.T @ R_pyfm2d).tocsc()

# --- Pass 1: rough sigma_d from residual at m0, run MAP only ---
_d_pred_m0 = _traced(m0_pyfm2d, tag="m0")[0]
sigma_d_m0 = np.std(d_obs_tt - _d_pred_m0)
print(f"Pass 1: rough sigma_d at m0 = {sigma_d_m0:.4f} s")

_Cd_inv_m0 = scipy.sparse.diags(
    [1.0 / sigma_d_m0**2], [0], shape=(_n_obs, _n_obs), format='csc'
)

inv_problem.set_data(d_obs_tt)
inv_problem.set_initial_model(m0_pyfm2d)
inv_problem.set_forward(vi_forward)
inv_problem.set_jacobian(vi_jacobian)
inv_problem.set_data_covariance_inv(_Cd_inv_m0)

inv_options.set_solving_method("variational inference")
inv_options.set_tool("cofi.gaussian_vi")
inv_options.set_params(
    prior_precision=Q_prior,
    prior_mean=m0_pyfm2d,
    num_iterations=0,            # MAP only
    num_samples=1,
    map_num_iterations=10,
    learning_rate_mean=0.02,
    learning_rate_precision=0.05,
    random_seed=42,
    verbose=True,
)

print("Running Pass 1 (MAP only)...")
print("-" * 60)
_pass1 = Inversion(inv_problem, inv_options).run()
print("-" * 60)
m_map = _pass1.map_model

# --- Pass 2: update sigma_d from residual at MAP, run full VI ---
_d_pred_map = _traced(m_map, tag="map")[0]
sigma_d = np.std(d_obs_tt - _d_pred_map)
print(f"\nPass 2: updated sigma_d at MAP = {sigma_d:.4f} s")

Cd_inv = scipy.sparse.diags(
    [1.0 / sigma_d**2], [0], shape=(_n_obs, _n_obs), format='csc'
)

inv_problem.set_initial_model(m_map)
inv_problem.set_data_covariance_inv(Cd_inv)

inv_options.set_params(
    prior_precision=Q_prior,
    prior_mean=m0_pyfm2d,
    num_iterations=10,
    num_samples=4,
    map_num_iterations=5,        # MAP already near-converged from Pass 1
    learning_rate_mean=0.02,
    learning_rate_precision=0.05,
    random_seed=42,
    verbose=True,
)

print("Running Pass 2 (full VI with updated C_d)...")
print("-" * 60)
vi_result = Inversion(inv_problem, inv_options).run()
print("-" * 60)

c_vi = 1.0 / vi_result.model
vi_elbo = vi_result.elbo_history
vi_sampler = vi_result.sampler
m_map_vi = vi_result.map_model

print(f"\nGaussian VI result:")
print(f"  Velocity range: [{c_vi.min():.2f}, {c_vi.max():.2f}] km/s")
print(f"  MAP velocity range: [{(1.0/m_map_vi).min():.2f}, {(1.0/m_map_vi).max():.2f}] km/s")
print(f"  Final ELBO: {vi_elbo[-1]:.2f}")

######################################################################
#

# ELBO convergence plot
fig_elbo, _ax = plt.subplots(figsize=(6, 3.5))
_ax.plot(np.arange(1, len(vi_elbo) + 1), vi_elbo, 'k-', lw=1.2)
_ax.set_xlabel('VI iteration')
_ax.set_ylabel('ELBO')
_ax.set_title('Gaussian VI convergence')
fig_elbo.tight_layout()

######################################################################
#


######################################################################
# **Note — a nearly flat ELBO is expected here.** The VI mean is
# initialised at the MAP estimate (Pass 1), so the posterior mode is
# already found before the first VI iteration. The VI phase then only
# refines the *covariance* (the uncertainty), which moves the ELBO by a
# small fraction of its total magnitude. The remaining
# iteration-to-iteration jitter is Monte-Carlo noise from estimating the
# ELBO with just ``num_samples`` draws. A small, noisy change is therefore
# correct behaviour for VI started at the MAP — not a sign of
# non-convergence.
# 

fig_vi_mean = plot_map(c_vi, title='Gaussian VI Posterior Mean',
                       grid_shape=fm2d_grid_shape, extent=fm2d_extent, vmin=cmin, vmax=cmax)

######################################################################
#

# Draw posterior samples and compute uncertainty
_samples = vi_sampler.sample(200)  # (200, n_params) in slowness
_vel_samples = 1.0 / _samples      # convert to velocity
_vel_std = np.std(_vel_samples, axis=0)

# Plot uncertainty map
_proj = ccrs.LambertConformal(
    central_longitude=135, central_latitude=-27,
    cutoff=80, standard_parallels=(-18, -36),
)
_transform = ccrs.PlateCarree()
_lon = np.linspace(fm2d_extent[0], fm2d_extent[1], fm2d_grid_shape[0])
_lat = np.linspace(fm2d_extent[2], fm2d_extent[3], fm2d_grid_shape[1])
_lon_grid, _lat_grid = np.meshgrid(_lon, _lat)

fig_vi_std = plt.figure(figsize=(5, 6.5))
_ax = plt.subplot(111, projection=_proj)
_ax.coastlines()
_img = _ax.pcolormesh(
    _lon_grid, _lat_grid, _vel_std.reshape(fm2d_grid_shape).T,
    cmap=scm.bilbao, transform=_transform, shading='auto',
)
_cb = fig_vi_std.colorbar(_img, ax=_ax, orientation='horizontal', shrink=0.8, pad=0.05)
_cb.set_label('Posterior std deviation [km/s]')
_ax.set_extent([113, 153, -45, -8], crs=_transform)
_ax.set_title('Gaussian VI Posterior Uncertainty')
plt.tight_layout()

######################################################################
#


######################################################################
# Results
# -------
# 
# The Gaussian VI posterior mean provides a velocity map comparable to the
# deterministic nonlinear inversions (see the `ambient noise tomography
# tutorial <surface_wave_tomography.ipynb>`__), but now accompanied by
# pixel-wise uncertainty estimates. The uncertainty map highlights regions
# where the data provides less constraint — typically at domain edges and
# where ray coverage is sparse.
# 
# Limitations
# ~~~~~~~~~~~
# 
# The Gaussian approximation assumes a unimodal, symmetric posterior. It
# cannot capture multi-modality, skewness, or the variable-complexity
# parameterisation that a trans-dimensional approach provides. For fully
# nonparametric uncertainty estimates, see the trans-dimensional Bayesian
# sampling section of the `ambient noise tomography
# tutorial <surface_wave_tomography.ipynb>`__.
# 


######################################################################
# References
# ----------
# 
# Chen, Y., Davis, T. A., Hager, W. W., & Rajamanickam, S. (2008).
# Algorithm 887: CHOLMOD, supernodal sparse Cholesky factorization and
# update/downdate. *ACM Transactions on Mathematical Software*, 35(3),
# Article 22. https://doi.org/10.1145/1391989.1391995
# 
# Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions.
# *Biometrika*, 96(4), 761-780. https://doi.org/10.1093/biomet/asp053
# 
# Khan, M. E., & Lin, W. (2017). Conjugate-computation variational
# inference: Converting variational inference in non-conjugate models to
# inferences in conjugate models. *Proceedings of the 20th International
# Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR
# 54, 878-887. https://proceedings.mlr.press/v54/khan17a.html
# 
# Khan, M. E., & Rue, H. (2023). The Bayesian learning rule. *Journal of
# Machine Learning Research*, 24(281), 1-46.
# https://www.jmlr.org/papers/v24/22-0291.html
# 
# Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between
# Gaussian fields and Gaussian Markov random fields: The stochastic
# partial differential equation approach. *Journal of the Royal
# Statistical Society: Series B (Statistical Methodology)*, 73(4),
# 423-498. https://doi.org/10.1111/j.1467-9868.2011.00777.x
# 
# Magrini, F., Kastle, E., Pilia, S., Rawlinson, N., & De Siena, L.
# (2023). A new shear-velocity model of continental Australia based on
# multi-scale surface-wave tomography. *Journal of Geophysical Research:
# Solid Earth*, 128(7), e2023JB026688.
# https://doi.org/10.1029/2023JB026688
# 
# Rawlinson, N., & Sambridge, M. (2004). Wave front evolution in strongly
# heterogeneous layered media using the fast marching method. *Geophysical
# Journal International*, 156(3), 631-647.
# https://doi.org/10.1111/j.1365-246X.2004.02153.x
# 


######################################################################
# --------------
# 
# Where to next?
# --------------
# 
# -  The ambient noise tomography tutorial, with linear, nonlinear and
#    trans-dimensional Bayesian inversions of the same data - `link to
#    notebook <https://github.com/inlab-geo/cofi-examples/blob/main/tutorials/surface_wave_tomography/surface_wave_tomography.ipynb>`__
# -  Variational inference for nonlinear inverse problems, including the
#    sinh-arcsinh flow - `link to
#    notebook <https://github.com/inlab-geo/cofi-examples/blob/main/tutorials/variational_inference/variational_inference.ipynb>`__
# 


######################################################################
# --------------
# 
# Watermark
# ---------
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib", "cmcrameri", "cartopy", "pyfm2d", "sksparse"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1