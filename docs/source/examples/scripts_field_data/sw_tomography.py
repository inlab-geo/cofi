"""
Surface-Wave Tomography
=======================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/sw_tomography/sw_tomography.ipynb
# 


######################################################################
# In this notebook, we will apply
# `CoFI <https://github.com/inlab-geo/cofi>`__ to measurements of
# surface-wave velocity collected across the
# `USArray <http://www.usarray.org/>`__ from the ambient seismic noise.
# Specifically, we will retrieve, through CoFI and
# `SeisLib <https://pypi.org/project/seislib/>`__, a Rayleigh-wave phase
# velocity map of the Conterminous United States at the surface-wave
# period of 5 s. The employed velocity measurements belong to the data set
# compiled by `Magrini et
# al. (2022) <https://doi.org/10.1093/gji/ggac236>`__.
# 


######################################################################
#    If you are running this notebook locally, make sure you’ve followed
#    `these
#    steps <https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally>`__
#    to set up the environment. (This
#    `environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/envs/environment.yml>`__
#    file specifies a list of packages required to run the notebooks)
# 


######################################################################
# Theoretical Background
# ----------------------
# 

# display theory background
from IPython.display import display, Markdown

with open("../../theory/geo_surface_wave_seislib.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#


######################################################################
# 1. Data and Parameterization
# ----------------------------
# 


######################################################################
# As mentioned earlier, the
# `data <https://github.com/inlab-geo/cofi-examples/blob/main/examples/sw_tomography/data.txt>`__
# used in this notebook consist of inter-station measurements of
# Rayleigh-wave phase velocity at 5 s period. We parameterize the Earth’s
# surface through equal-area blocks of :math:`1^{\circ} \times 1^{\circ}`.
# 

from seislib.tomography import SeismicTomography

tomo = SeismicTomography(cell_size=1) # Parameterization

# To reproduce the results locally, download data.txt and change the below path

tomo.add_data(src='../../data/sw_tomography/data.txt')

######################################################################
#


######################################################################
# Overall, 171,353 velocity measurements are available (check
# ``tomo.velocity``), each associated with a different pair of receveirs
# (check ``tomo.data_coords``, consisting of a matrix of 171,353 rows and
# 4 columns: :math:`\theta_1`, :math:`\phi_1`, :math:`\theta_2`, and
# :math:`\phi_2`).
# 


######################################################################
# 2. Jacobian
# -----------
# 


######################################################################
# We use the information about the data coordinates to calculate the
# matrix :math:`\bf A` (i.e. the Jacobian). In doing so, we will discard
# all blocks parameterizing the Earth’s surface that are not intersected
# by at least one inter-station great-circle path. These model parameters
# (referred to as “grid cells” in the below output) have no sensitivity to
# our data.
# 

# This discards all blocks that are far away from the study area

tomo.grid.set_boundaries(latmin=tomo.latmin_data, 
                         latmax=tomo.latmax_data, 
                         lonmin=tomo.lonmin_data, 
                         lonmax=tomo.lonmax_data)

######################################################################
#

# Computes the coefficients of the A matrix, while discarding all model parameters that are not constrained by our data.
tomo.compile_coefficients(keep_empty_cells=False)

######################################################################
#


######################################################################
# The Jacobian can now be accessed by typing ``tomo.A``, and the
# associated parameterization can be visualized by typing
# 

tomo.grid.plot()

######################################################################
#


######################################################################
# 3. Inversion – SeisLib style
# ----------------------------
# 


######################################################################
# The lateral variations in phase velocity can now simply be retrieved,
# via SeisLib, through
# 

mu = 5e-2 # Roughness damping coefficient, arbitrarily chosen

# The output of tomo.solve is slowness, hence we take the reciprocal

c = 1 / tomo.solve(rdamp=mu) # in km/s

######################################################################
#


######################################################################
# Let’s have a look at the results.
# 

from seislib.plotting import plot_map
import seislib.colormaps as scm

img, cb = plot_map(tomo.grid.mesh, c, cmap=scm.roma, show=False)
cb.set_label('Phase velocity [km/s]')

######################################################################
#


######################################################################
# 4. Inversion – CoFI style
# -------------------------
# 


######################################################################
# Let’s now reproduce the above results through CoFI. First, we need to
# define a starting model :math:`{\bf x}_0` to compute the residuals
# :math:`{\bf r} = {\bf d} - {\bf A} \cdot {\bf x}_0`, as in equation (3).
# 

import numpy as np

A = tomo.A # Jacobian
x0 = np.full(A.shape[1], 1 / tomo.refvel) # tomo.refvel is the average inter-station phase velocity
d = 1 / tomo.velocity # measurements of (average) inter-station slowness
r = d - A @ x0 # residuals

######################################################################
#


######################################################################
# We now need to define the roughness operator :math:`\bf R`. This is done
# under the hood by SeisLib through the “private” method
# ``_derivatives_lat_lon``.
# 

from seislib.tomography._ray_theory._tomography import _derivatives_lat_lon

# coordinates of each parameterization block, tomo.grid.mesh, should be in radians

R_lat, R_lon = _derivatives_lat_lon(np.radians(tomo.grid.mesh))
R = np.row_stack((R_lat, R_lon))

######################################################################
#


######################################################################
# Almost everything is ready to carry out the inversion through CoFI.
# Before doing so, we need to define our inverse problem (through
# ``BaseProblem``) and pass to it the data and the Jacobian (through
# ``set_data`` and ``set_jacobian``). Finally, we will specify the
# regularizazion criterion (through ``set_regularization``).
# 

from cofi import BaseProblem
from cofi.utils import QuadraticReg

problem = BaseProblem()
problem.set_data(r) # our data are now the residuals defined above
problem.set_jacobian(A)

# As opposed to SeisLib, CoFI does not square the damping coefficient.
problem.set_regularization(mu**2 * QuadraticReg(R, (A.shape[1],)))   # L2 norm of R, i.e. R.T @ R

problem.summary()

######################################################################
#


######################################################################
# We now carry out the inversion through ``scipy.linalg.lstsq``.
# 

from cofi import Inversion, InversionOptions

options = InversionOptions()
options.set_tool("scipy.linalg.lstsq")

inv = Inversion(problem, options)
inv_results = inv.run()
inv.summary()

######################################################################
#


######################################################################
# 5. Cross validation
# -------------------
# 


######################################################################
# The inversion converged. Let’s now check whether the results are
# consistent with those obtained from SeisLib. To do so, remember that we
# need to add back, to the retrieved model parameters, the initial
# reference model :math:`{\bf x}_0`.
# 

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# the reference model x0 is added back to get absolute values of slowness

c_cofi = 1 / ( inv_results.model + x0 )

fig = plt.figure(figsize=(10, 8))

# SeisLib map

ax1 = plt.subplot(121, projection=ccrs.Mercator())
ax1.coastlines()
img1, cb1 = plot_map(tomo.grid.mesh, c, ax=ax1, cmap=scm.roma, show=False)
cb1.set_label('Phase velocity [km/s]')
ax1.set_title('SeisLib')

# CoFI map

ax2 = plt.subplot(122, projection=ccrs.Mercator())
ax2.coastlines()
img2, cb2 = plot_map(tomo.grid.mesh, c_cofi, ax=ax2, cmap=scm.roma, show=False)
cb2.set_label('Phase velocity [km/s]')
ax2.set_title('CoFI')

plt.tight_layout()
plt.show()

print('Are the results obtained from seislib and cofi the same?', np.allclose(c, c_cofi))


######################################################################
#


######################################################################
# Watermark
# ---------
# 

libraries_used = ["cartopy", "cofi", "matplotlib", "numpy", "seislib"]
for lib in libraries_used:
    lib_var = __import__(lib)
    print(lib, getattr(lib_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1