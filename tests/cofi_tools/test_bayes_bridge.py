import numpy as np
import random
from pysurf96 import surf96
import cofi


# -------------- Setting up constants, fwd func, synth data
VP_VS = 1.77
RAYLEIGH_STD = 0.02
LOVE_STD = 0.02
RF_STD = 0.03
LAYERS_MIN = 3
LAYERS_MAX = 15
LAYERS_INIT_RANGE = 0.3
VS_PERTURB_STD = 0.15
VS_UNIFORM_MIN = 2.7
VS_UNIFORM_MAX = 5
VORONOI_PERTURB_STD = 8
VORONOI_POS_MIN = 0
VORONOI_POS_MAX = 130
N_CHAINS = 10

def forward_sw(model, periods, wave="rayleigh", mode=1):
    sites = model[:len(model)/2]
    depths = (sites[:-1] + sites[1:]) / 2
    thickness = np.hstack((depths[0], depths[1:]-depths[:-1], 0))
    vs = model[len(model)/2:]
    vp = vs * VP_VS
    rho = 0.32 * vp + 0.77
    return surf96(
        thickness,
        vp,
        vs,
        rho,
        periods,
        wave=wave,
        mode=mode,
        velocity="phase",
        flat_earth=False,
    )

true_thickness = np.array([10, 10, 15, 20, 20, 20, 20, 20, 0])
true_voronoi_positions = np.array([5, 15, 25, 45, 65, 85, 105, 125, 145])
true_vs = np.array([3.38, 3.44, 3.66, 4.25, 4.35, 4.32, 4.315, 4.38, 4.5])
true_model = np.hstack((true_thickness, true_vs))

periods1 = np.linspace(4, 80, 20)
rayleigh1 = forward_sw(true_model, periods1, "rayleigh", 1)
rayleigh1_noisy = rayleigh1 + np.random.normal(0, RAYLEIGH_STD, rayleigh1.size)
love1 = forward_sw(true_model, periods1, "love", 1)
love1_noisy = love1 + np.random.normal(0, LOVE_STD, love1.size)


# -------------- Implement distribution functions
_N = 100_000   # possible positions, to be cancelled out in the final ratio

def log_prior(model):
    sites = model[:len(model)/2]
    vs = model[len(model)/2:]
    k = len(sites)
    # p(c|k) prior on voronoi cell positions given #layers
    log_p_c_k = np.math.factorial(k) / np.prod(np.arange(_N-k+1, _N+1))
    # p(v|k) prior on param value given #layers
    log_p_v_k = float("-inf") \
        if any(vs > VS_UNIFORM_MAX or vs < VS_UNIFORM_MIN) \
            else - k * np.log(VS_UNIFORM_MAX - VS_UNIFORM_MIN)
    # p(k) prior on #layers
    log_p_k = - np.log(LAYERS_MAX - LAYERS_MIN)
    return log_p_c_k + log_p_v_k + log_p_k

def log_likelihood(model):
    rayleigh_synth = forward_sw(model, periods1, "rayleigh", 1)
    rayleigh_residual = rayleigh1_noisy - rayleigh_synth
    rayleigh_loglike = -0.5 * np.sum(
        (rayleigh_residual/RAYLEIGH_STD)**2 + np.log(2 * np.pi * RAYLEIGH_STD**2)
    )
    love_synth = forward_sw(model, periods1, "love", 1)
    love_residual = love1_noisy - love_synth
    love_loglike = -0.5 * np.sum(
        (love_residual/LOVE_STD) ** 2 + np.log(2 * np.pi * LOVE_STD**2)
    )
    return rayleigh_loglike + love_loglike


# -------------- Implement perturbation functions
def perturbation_vs(model):
    sites = model[:len(model)/2]
    vs = model[len(model)/2:]
    k = len(sites)
    # randomly choose a Voronoi site to perturb the value
    isite = random.randint(0, k-1)
    # randomly perturb the value
    while True:
        random_deviate = random.normalvariate(0, VS_PERTURB_STD)
        new_value = vs[isite] + random_deviate
        if new_value > VS_UNIFORM_MAX or new_value < VS_UNIFORM_MIN:
            continue
        break
    # integrate into a new model variable
    new_vs = vs.copy()
    new_vs[isite] = new_value
    new_model = np.hstack((sites, new_vs))
    return new_model, 0

def perturbation_voronoi_site(model):
    sites = model[:len(model)/2]
    vs = model[len(model)/2:]
    k = len(sites)
    # randomly choose a Voronoi site to perturb the position
    isite = random.randint(0, k-1)
    old_site = sites[isite]
    # randomly perturb the position
    while True:
        random_deviate = random.normalvariate(0, VORONOI_PERTURB_STD)
        new_site = old_site + random_deviate
        if new_site < VORONOI_POS_MIN or new_site > VORONOI_POS_MAX or \
            np.any(np.abs(new_site - sites) < 1e-2):
                continue
        break
    # integrate into a new model variable
    new_sites = sites.copy()
    new_sites[isite] = new_site
    new_model = np.hstack((new_sites, vs))
    return new_model, 0

def perturbation_birth(model):
    sites = model[:len(model)/2]
    vs = model[len(model)/2:]
    k = len(sites)
    if k == LAYERS_MAX:
        raise RuntimeError("Maximum layers reached")
    # randomly choose a new Voronoi site position
    while True:
        new_site = random.uniform(VORONOI_POS_MIN, VORONOI_POS_MAX)
        # abort if it's too close to existing positions
        if np.any(np.abs(new_site - sites) < 1e-2):
            continue
        break
    # randomly sample the value for the new site
    new_vs_isite = random.uniform(VS_UNIFORM_MIN, VS_UNIFORM_MAX)
    # integrate into a new model variable and sort properly
    new_sites = sites.copy()
    new_sites.append(new_site)
    new_vs = vs.copy()
    new_vs.append(new_vs_isite)
    isort = np.argsort(new_sites)
    new_sites = new_sites[isort]
    new_vs = new_vs[isort]
    new_model = np.hstack((new_sites, new_vs))
    # calculate proposal ratio
    log_proposal_ratio = np.log((_N-k) / (k+1) / (VS_UNIFORM_MAX - VS_UNIFORM_MIN))
    return new_model, log_proposal_ratio

def perturbation_death(model):
    sites = model[:len(model)/2]
    vs = model[len(model)/2:]
    k = len(sites)
    if k == LAYERS_MIN:
        raise RuntimeError("Minimum layers reached")
    # randomly choose an existing Voronoi site to remove
    isite = random.randint(0, k-1)
    # integrate into a new model variable
    new_sites = np.delete(sites, isite)
    new_vs = np.delete(vs, isite)
    new_model = np.hstack((new_sites, new_vs))
    # calculate proposal ratio
    log_proposal_ratio = np.log(k * (VS_UNIFORM_MAX - VS_UNIFORM_MIN) / (_N-k+1))
    return new_model, log_proposal_ratio


# -------------- Initialize walkers
init_max = int((LAYERS_MAX - LAYERS_MIN) * LAYERS_INIT_RANGE + LAYERS_MIN)
walkers_start = []
for i in range(N_CHAINS):
    n_sites = random.randint(LAYERS_MIN, init_max)
    sites = np.sort(np.random.uniform(VORONOI_POS_MIN, VORONOI_POS_MAX, n_sites))
    vs = np.sort(np.random.uniform(VS_UNIFORM_MIN, VS_UNIFORM_MAX, n_sites))
    model = np.hstack((sites, vs))
    walkers_start.append(model)


# -------------- Define CoFI problem
sw_problem = cofi.BaseProblem()
sw_problem.set_log_prior(log_prior)
sw_problem.set_log_likelihood(log_likelihood)


# -------------- Define CoFI inversion options
inv_options = cofi.InversionOptions()
inv_options.set_tool("bayesbridge")
inv_options.set_params(
    perturbations = [
        perturbation_vs, 
        perturbation_voronoi_site, 
        perturbation_birth, 
        perturbation_death, 
    ], 
    walkers_starting_pos = walkers_start, 
)


# -------------- Run CoFI inversion 
sw_inversion = cofi.Inversion(sw_problem, inv_options)
sw_inversion.run()
