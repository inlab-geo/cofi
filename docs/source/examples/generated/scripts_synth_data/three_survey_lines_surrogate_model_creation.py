"""
Create a surrogate model for the objective function for the three survey line example
=====================================================================================

.. raw:: html

   <!-- Please leave the cell below as it is -->

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/airborne_em/airborne_em_three_lines_transmitters.ipynb
# 


######################################################################
# .. raw:: html
# 
#    <!-- Again, please don't touch the markdown cell above. We'll generate badge 
#         automatically from the above cell. -->
# 
# .. raw:: html
# 
#    <!-- This cell describes things related to environment setup, so please add more text 
#         if something special (not listed below) is needed to run this notebook -->
# 
# ..
# 
#    If you are running this notebook locally, make sure you’ve followed
#    `steps
#    here <https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally>`__
#    to set up the environment. (This
#    `environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/envs/environment.yml>`__
#    file specifies a list of packages required to run the notebooks)
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi
# !pip install git+https://github.com/JuergHauser/PyP223.git
# !pip install smt
# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/examples/vtem_max

######################################################################
#

# If this notebook is run locally PyP223 and smt need to be installed separately by uncommenting the following lines, 
# that is by removing the # and the white space between it and the exclamation mark.
# !pip install git+https://github.com/JuergHauser/PyP223.git
# !pip install smt

######################################################################
#

import pickle
import numpy
import matplotlib.pyplot as plt
import cofi
import smt
import smt.sampling_methods
import smt.surrogate_models 
import smt.utils.misc
import tqdm
from vtem_max_forward_lib import (
    problem_setup, 
    system_spec, 
    survey_setup, 
    ForwardWrapper
)

numpy.random.seed(42)
numpy.set_printoptions()

######################################################################
#


######################################################################
# Background
# ==========
# 
# This example use the surrogate modelling toolbox to create a surrogate
# model for the objective function used in the synthetic
# `example <./three_survey_lines.ipynb>`__ where we invert the vertical
# component of three survey lines of a VTEM max survey. This notebook
# employs the `Kriging
# approach <https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/gpr/krg.html>`__
# using the training and test samples generated
# `here <three_survey_lines_latin_hypercube_sampling.ipynb>`__ to create a
# surrogate model.
# 

with open('three_survey_lines_lhs.npy', 'rb') as f:
    ndim=int(numpy.load(f))
    xlimits=numpy.load(f) 
    xtrain=numpy.load(f)
    ytrain=numpy.load(f)
    xtest=numpy.load(f)
    ytest=numpy.load(f)

xlimits=xlimits.astype('double')
xtrain=xtrain[0:150].astype('double')
ytrain=ytrain[0:150].astype('double')
xtest=xtest[0:25].astype('double')
ytest=ytest[0:25].astype('double')

######################################################################
#


######################################################################
# Training
# --------
# 

# The variable 'theta0' is a list of length ndim.
t = smt.surrogate_models.KRG(theta0=[1e-2]*ndim,print_prediction = False)
t.set_training_values(xtrain,ytrain)
t.train()

######################################################################
#


######################################################################
# Validation
# ----------
# 

# Prediction of the validation points
y = t.predict_values(xtest)
# Estimated variance for the validation points
s2 = t.predict_variances(xtest)
#plot with the associated interval confidence
yerr= 2*3*numpy.sqrt(s2) #in order to use +/- 3 x standard deviation: 99% confidence interval estimation

# Plot the function, the prediction and the 99% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(ytest, ytest, '-', label='$y_{true}$')
plt.plot(ytest, y, 'r.', label='$\\hat{y}$')
plt.errorbar(numpy.squeeze(ytest), numpy.squeeze(y), yerr=numpy.squeeze(yerr), fmt = 'none', capsize = 5, ecolor = 'lightgray', elinewidth = 1, capthick = 0.5, label='confidence estimate 99%')
plt.xlabel('$y_{true}$')
plt.ylabel('$\\hat{y}$')

plt.legend(loc='upper left')
plt.title('Validation of the model prediction with confidence estimates')   
plt.show()

######################################################################
#


######################################################################
# Save the surrogate model for subsequent use
# -------------------------------------------
# 

filename = "kriging_surrogate_model.pkl"
with open(filename, "wb") as f:
   pickle.dump(t, f)

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# =========
# 
# .. raw:: html
# 
#    <!-- Feel free to add more modules in the watermark_list below, if more packages are used -->
# 
# .. raw:: html
# 
#    <!-- Otherwise please leave the below code cell unchanged -->
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib","smt"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#



######################################################################
#
# sphinx_gallery_thumbnail_number = -1