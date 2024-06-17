"""
Xray Tomography
===============

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/xray_tomography/xray_tomography.ipynb
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
# *Adapted from notebooks by Andrew Valentine & Malcolm Sambridge -
# Research School of Earth Sciences, The Australian National University*
# 
# In this notebook, we look at an linear inverse problem based on Xray
# Tomography. We will use ``cofi`` to run a linear system solver
# (optionally with Tikhonov regularization and noise estimation) for this
# problem.
# 


######################################################################
# 0. Import modules
# -----------------
# 
# The package ``geo-espresso`` contains the forward code for this problem.
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi geo-espresso
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/examples/xray_tomography

######################################################################
#

import numpy as np
from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import QuadraticReg
from espresso import XrayTomography

######################################################################
#


######################################################################
# 1. Define the problem
# ---------------------
# 

# display theory on the inference problem
from IPython.display import display, Markdown

with open("../../theory/geo_xray_tomography.md", "r") as f:
    content = f.read()

display(Markdown(content))

######################################################################
#


######################################################################
# Firstly, we get some information from the ``geo-espresso`` module. These
# include the dataset and the Jacobian matrix. In the Xray Tomography
# example, the Jacobian matrix is related to the lengths of paths within
# each grid. Since the paths are fixed, the Jacobian matrix stays
# constant.
# 

xrt = XrayTomography()

######################################################################
#

xrt_problem = BaseProblem()
xrt_problem.set_data(xrt.data)
xrt_problem.set_jacobian(xrt.jacobian(xrt.starting_model))

######################################################################
#


######################################################################
# We do some estimation on data noise and further perform a
# regularization.
# 

sigma = 0.002
lamda = 50
data_cov_inv = np.identity(xrt.data_size) * (1/sigma**2)

######################################################################
#

xrt_problem.set_data_covariance_inv(data_cov_inv)
xrt_problem.set_regularization(lamda * QuadraticReg(model_shape=(xrt.model_size,)))

######################################################################
#


######################################################################
# Review what information is included in the ``BaseProblem`` object:
# 

xrt_problem.summary()

######################################################################
#


######################################################################
# 2. Define the inversion options
# -------------------------------
# 

my_options = InversionOptions()
my_options.set_tool("scipy.linalg.lstsq")

######################################################################
#


######################################################################
# Review what’s been defined for the inversion we are about to run:
# 

my_options.summary()

######################################################################
#


######################################################################
# 3. Start an inversion
# ---------------------
# 
# We can now solve the inverse problem using the Tikhonov-regularized form
# of least-squares,
# 
# .. math:: \mathbf{m}=\left(\mathbf{A^TA}+\epsilon^2\sigma^2\mathbf{I}\right)^\mathbf{-1}\mathbf{A^Td}
# 
# where :math:`\sigma^2` is the variance of the expected noise on the
# attenuation data.
# 
# For this dataset, we’ve taken :math:`\sigma = 0.002`\ s and chosen
# :math:`\epsilon^2 = 50`.
# 

inv = Inversion(xrt_problem, my_options)
inv_result = inv.run()
inv_result.summary()

######################################################################
#


######################################################################
# 4. Plotting
# -----------
# 
# Below the two figures refers to the inferred model and true model
# respectively.
# 

xrt.plot_model(inv_result.model, clim=(1, 1.5));       # inferred model
xrt.plot_model(xrt.good_model, clim=(1, 1.5));          # true model

######################################################################
#


######################################################################
# 5. Estimated uncertainties
# --------------------------
# 
# We can now find the uncertainty on the recovered slowness parameters,
# which describes how noise in the data propagate into the slowness
# parameters with this data set. For the Tikhonov-regularised form of
# least-squares, the model covariance matrix is a square matrix of size
# :math:`M\times M`, where there are :math:`M` cells in the model.
# 
# .. math:: \mathbf{C_m}=\sigma^2\left(\mathbf{A^TA}+\epsilon^2\sigma^2\mathbf{I}\right)^\mathbf{-1}
# 
# .
# 
# This matrix was calculated as part of the solver routine above. The
# square roots of the diagonal entries of this matrix are the
# :math:`\sigma` errors in the slowness in each cell.
# 

Cm = inv_result.model_covariance

######################################################################
#


######################################################################
# Lets plot the slowness uncertainties as a function of position across
# the cellular model.
# 

xrt.plot_model(np.sqrt(np.diag(Cm)));

######################################################################
#


######################################################################
# Uncertainty is uniformly low across the entire model and only
# significant near the corners where there are few ray paths.
# 
# Similarly we can calculate uncertainty in velocity parameters using some
# calculus.
# 
# .. math::  \Delta v = \left | \frac{\partial s}{\partial v}  \right | \Delta s 
# 
# and since :math:`s = 1/v` we get
# 
# .. math::  \Delta v = s^2\Delta s 
# 
# which gives the uncertainty image on velocity, which looks very similar.
# 

xrt.plot_model(np.sqrt(np.diag(Cm)) * inv_result.model);

######################################################################
#


######################################################################
# By clipping the colour range you can see an imprint of the true image,
# indicating that high slowness/low velcoity areas have slightly higher
# uncertainty.
# 


######################################################################
# --------------
# 
# Watermark
# ---------
# 
# .. raw:: html
# 
#    <!-- Feel free to add more modules in the watermark_list below, if more packages are used -->
# 
# .. raw:: html
# 
#    <!-- Otherwise please leave the below code cell unchanged -->
# 

watermark_list = ["cofi", "espresso", "numpy", "scipy", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1