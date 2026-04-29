"""
Linear regression
=================

"""


######################################################################
# |Open In Colab|
# 
# .. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
#    :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/tutorials/linear_regression/linear_regression.ipynb
# 


######################################################################
# --------------
# 
# What we do in this notebook
# ---------------------------
# 
# Here we demonstrate use of CoFI on a simple **linear regression**
# problem, where we fit a polynomial function to data, using three
# different algorithms:
# 
# - by solution of a linear system of equations,
# - by optimization of a data misfit function
# - by Bayesian sampling of a Likelihood multiplied by a prior.
# 
# --------------
# 
# Learning outcomes
# -----------------
# 
# - A demonstration of running CoFI for a class of parameter fitting
#   problem. Example of a CoFI **template**.
# - A demonstration of how CoFI may be used to **experiment with different
#   inference approaches** under a common interface.
# - A demonstration of CoFI’s **expandability** in that it may be used
#   with pre-set, or user defined, misfits, likelihood or priors.
# 

# Environment setup (uncomment code below)

# !pip install -U cofi

######################################################################
#


######################################################################
# Linear regression
# -----------------
# 
# Lets start with some (x,y) data.
# 

import numpy as np
import matplotlib.pyplot as plt

######################################################################
#

# here is some (x,y) data
data_x = np.array([1.1530612244897958, -0.07142857142857162, -1.7857142857142858, 
                1.6428571428571423, -2.642857142857143, -1.0510204081632653, 
                1.1530612244897958, -1.295918367346939, -0.806122448979592, 
                -2.2755102040816326, -2.2755102040816326, -0.6836734693877551, 
                0.7857142857142856, 1.2755102040816322, -0.6836734693877551, 
                -3.2551020408163267, -0.9285714285714288, -3.377551020408163, 
                -0.6836734693877551, 1.7653061224489797])

data_y = np.array([-7.550931153863841, -6.060810406314714, 3.080063056254076, 
                -4.499764131508964, 2.9462042659962333, -0.4645899453212615, 
                -7.43068837808917, 1.6273774547833582, -0.05922697815443567, 
                3.8462283231266903, 3.425851020301113, -0.05359797104829345, 
                -10.235538857712598, -5.929113775071286, -1.1871766078924957, 
                -4.124258811692425, 0.6969191559961637, -4.454022624935177, 
                -2.352842192972056, -4.25145590011172])
sigma = 1   # estimation on the data noise

######################################################################
#


######################################################################
# And now lets plot the data.
# 

def plot_data(sigma=None):
    if(sigma is None):
        plt.scatter(data_x, data_y, color="lightcoral", label="observed data")
    else:
        plt.errorbar(data_x, data_y, yerr=sigma, fmt='.',color="lightcoral",ecolor='lightgrey',ms=10)
plot_data()

######################################################################
#


######################################################################
# Problem description
# -------------------
# 
# To begin with, we will work with polynomial curves,
# 
# .. math:: y(x) = \sum_{j=0}^M m_j x^j\,.
# 
# Here, :math:`M` is the ‘order’ of the polynomial: if :math:`M=1` we have
# a straight line with 2 parameters, if :math:`M=2` it will be a quadratic
# with 3 parameters, and so on. The :math:`m_j, (j=0,\dots M)` are the
# ‘model coefficients’ that we seek to constrain from the data.
# 
# For this class of problem the forward operator takes the following form:
# 
# .. math:: \left(\begin{array}{c}y_0\\y_1\\\vdots\\y_N\end{array}\right) = \left(\begin{array}{ccc}1&x_0&x_0^2&x_0^3\\1&x_1&x_1^2&x_1^3\\\vdots&\vdots&\vdots\\1&x_N&x_N^2&x_N^3\end{array}\right)\left(\begin{array}{c}m_0\\m_1\\m_2\\m_3\end{array}\right)
# 
# This clearly has the required general form,
# :math:`\mathbf{d} =G{\mathbf m}`.
# 
# where:
# 
# - :math:`\textbf{d}` is the vector of data values,
#   (:math:`y_0,y_1,\dots,y_N`);
# - :math:`\textbf{m}` is the vector of model parameters,
#   (:math:`m_0,m_1,m_2`);
# - :math:`G` is the basis matrix (or design matrix) of this linear
#   regression problem (also called the **Jacobian** matrix for this
#   linear problem).
# 
# We have a set of noisy data values, :math:`y_i (i=0,\dots,N)`, measured
# at known locations, :math:`x_i (i=0,\dots,N)`, and wish to find the best
# fit degree 3 polynomial.
# 
# The function that generated our data is : :math:`y=-6-5x+2x^2+x^3`, and
# we have added Gaussian random noise, :math:`{\cal N}(0,\sigma^2)`, with
# :math:`\sigma=1.0`.
# 


######################################################################
# We now build the Jacobian/G matrix for this problem and define a forward
# function which simply multiplies :math:`\mathbf m` by :math:`G`.
# 

nparams = 4 # Number of model parameters to be solved for

def jacobian(x=data_x, n=nparams):
    return np.array([x**i for i in range(n)]).T

def forward(model):
    return jacobian().dot(model)

def Cd_inv(sigma=sigma, ndata=len(data_x)):
    return 1/sigma**2 * np.identity(ndata)

######################################################################
#


######################################################################
# Define the true model for later.
# 

# True model for plotting
x = np.linspace(-3.5,2.5)              # x values to plot
true_model = np.array([-6, -5, 2, 1])  # we know it for this case which will be useful later for comparison.

true_y = jacobian(x,4).dot(true_model) # y values for true curve

######################################################################
#


######################################################################
# Now lets plot the data with the curve from the true polynomial
# coefficients.
# 

# Some plotting utilities
def plot_model(x,y, label, color=None):
    #x = np.linspace(-3.5,2.5)
    #y = jacobian(x).dot(model)
    plt.plot(x, y, color=color or "green", label=label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

def plot_models(models, label="Posterior samples", color="seagreen", alpha=0.1):
    x = np.linspace(-3.5,2.5)
    G = jacobian(x)
    plt.plot(x, G.dot(models[0]), color=color, label=label, alpha=alpha)
    for m in models:
        plt.plot(x, G.dot(m), color=color, alpha=alpha)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

######################################################################
#

plot_data(sigma=sigma)
plot_model(x,true_y, "true model")

######################################################################
#


######################################################################
# Now we have the data and the forward model we can start to try and
# estimate the coefficients of the polynomial from the data.
# 


######################################################################
# The structure of CoFI 
# ----------------------
# 
# In the workflow of ``cofi``, there are three main components:
# ``BaseProblem``, ``InversionOptions``, and ``Inversion``.
# 
# - ``BaseProblem`` defines the inverse problem including any user
#   supplied quantities such as data vector, number of model parameters
#   and measure of fit between model predictions and data.
# 
#   .. code:: python
# 
#      inv_problem = BaseProblem()
#      inv_problem.set_objective(some_function_here)
#      inv_problem.set_jacobian(some_function_here)
#      inv_problem.set_initial_model(a_starting_point) # if needed, e.g. we are solving a nonlinear problem by optimization
# 
#    
# 
# - ``InversionOptions`` describes details about how one wants to run the
#   inversion, including the backend tool and solver-specific parameters.
#   It is based on the concept of a ``method`` and ``tool``.
# 
#   .. code:: python
# 
#      inv_options = InversionOptions()
#      inv_options.suggest_solving_methods()
#      inv_options.set_solving_method("matrix solvers")
#      inv_options.suggest_tools()
#      inv_options.set_tool("scipy.linalg.lstsq")
#      inv_options.summary()
# 
#    
# 
# - ``Inversion`` can be seen as an inversion engine that takes in the
#   above two as information, and will produce an ``InversionResult`` upon
#   running.
# 
#   .. code:: python
# 
#      inv = Inversion(inv_problem, inv_options)
#      result = inv.run()
# 
# Internally CoFI decides the nature of the problem from the quantities
# set by the user and performs internal checks to ensure it has all that
# it needs to solve a problem.
# 


######################################################################
# 1. Linear system solver
# -----------------------
# 

from cofi import BaseProblem, InversionOptions, Inversion

######################################################################
#


######################################################################
# Step 1. Define CoFI ``BaseProblem``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

inv_problem = BaseProblem()
inv_problem.set_data(data_y)
inv_problem.set_jacobian(jacobian())
inv_problem.set_data_covariance_inv(Cd_inv())

######################################################################
#


######################################################################
# Step 2. Define CoFI ``InversionOptions``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

inv_options = InversionOptions()

######################################################################
#


######################################################################
# Using the information supplied, we can ask CoFI to suggest some solving
# methods.
# 

inv_options.suggest_solving_methods()

######################################################################
#


######################################################################
# We can ask CoFI to suggest some specific software tools as well.
# 

inv_options.suggest_tools()

######################################################################
#

inv_options.set_solving_method("matrix solvers") # lets decide to use a matrix solver.
inv_options.summary()

######################################################################
#

# below is optional, as this has already been the default tool under "linear least square"
inv_options.set_tool("scipy.linalg.lstsq")

######################################################################
#


######################################################################
# Step 3. Define CoFI ``Inversion`` and run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Our choices so far have defined a linear parameter estimation problem
# (without any regularization) to be solved within a least squares
# framework. In this case the selection of a ``matrix solvers`` method
# will mean we are calculating the standard least squares solution
# 
# .. math::
# 
# 
#    m = (G^T C_d^{-1} G)^{-1} G^T C_d^{-1} d
# 
# and our choice of backend tool ``scipy.linalg.lstsq``, means that we
# will employ scipy’s ``linalg`` package to perform the numerics.
# 
# Lets run CoFI.
# 

inv = Inversion(inv_problem, inv_options)
inv_result = inv.run()

######################################################################
#

print(f"The inversion result from `scipy.linalg.lstsq`: {inv_result.model}\n")
inv_result.summary()

######################################################################
#


######################################################################
# Lets plot the solution.
# 

plot_data()
plot_model(x,jacobian(x).dot(inv_result.model), "linear system solver", color="seagreen")
plot_model(x,true_y, "true model", color="darkorange")

######################################################################
#


######################################################################
# 2. Optimizer
# ------------
# 
# The same overdetermined linear problem,
# :math:`\textbf{d} = G\textbf{m}`, with Gaussian data noise can also be
# solved by minimising the squares of the residual of the linear
# equations, e.g. :math:`\textbf{r}^T \textbf{C}_d^{-1}\textbf{r}` where
# :math:`\textbf{r}=\textbf{d}-G\textbf{m}`. The above matrix solver
# solution gives us the best data fitting model, but a direct optimisation
# approach could also be used, say when the number of unknowns is large
# and we do not wish, or are unable to provide the Jacobian function.
# 
# So we use a plain optimizer ``scipy.optimize.minimize`` to demonstrate
# this ability.
# 
# .. raw:: html
# 
#    <!-- For this backend solver to run successfully, some additional information should be provided, otherwise
#    you'll see an error to notify what additional information is required by the solver.
# 
#    There are several ways to provide the information needed to solve an inverse problem with 
#    CoFI. In the example below we provide functions to calculate the data and the optional 
#    regularisation. CoFI then generates the objective function for us based on the information 
#    provided. The alternative to this would be to directly provide objective function to CoFI. -->
# 

######## CoFI BaseProblem - provide additional information
inv_problem.set_initial_model(np.ones(nparams))
inv_problem.set_forward(forward)
inv_problem.set_data_misfit("squared error")

# inv_problem.set_objective(your_own_misfit_function)    # (optionally) if you'd like to define your own misfit
# inv_problem.set_gradient(your_own_gradient_of_misfit_function)    # (optionally) if you'd like to define your own misfit gradient

######## CoFI InversionOptions - set a different tool
inv_options_2 = InversionOptions()
inv_options_2.set_tool("scipy.optimize.minimize")
inv_options_2.set_params(method="Nelder-Mead")

######## CoFI Inversion - run it
inv_2 = Inversion(inv_problem, inv_options_2)
inv_result_2 = inv_2.run()

######## CoFI InversionResult - check result
print(f"The inversion result from `scipy.optimize.minimize`: {inv_result_2.model}\n")
inv_result_2.summary()

######################################################################
#

plot_data()
plot_model(x,jacobian(x).dot(inv_result_2.model), "optimization solution", color="cornflowerblue")
plot_model(x,true_y, "true model", color="darkorange")

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# Challenge: Change the polynomial degree
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Try and replace the 3rd order polynomial with a 1st order polynomial
# (i.e. :math:`M=1`) by adding the required commands below. What does the
# plot looks like?
# 
# |Upload to Excalidraw_1|
# 
# Start from code below:
# 
# ::
# 
#    inv_problem = BaseProblem()
#    inv_problem.set_data(data_y)
#    inv_problem.set_jacobian(jacobian(n=<CHANGE ME>))
#    inv_problem.set_data_covariance_inv(Cd_inv())
#    inv_options.set_solving_method("matrix solvers") # lets decide to use a matrix solver.
#    inv = Inversion(inv_problem, inv_options)
#    inv_result = inv.run()
# 
#    print("Inferred curve with n = <CHANGE ME> ")
#    plot_data()
#    plot_model(x,jacobian(x,n=<CHANGE ME>).dot(inv_result.model), "optimization solution", color="cornflowerblue")
#    plot_model(x,true_y, "true model", color="darkorange")
# 
# .. |Upload to Excalidraw_1| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Excalidraw-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
#    :target: https://excalidraw.com/#room=351ca2833e3ad8315d6a,HzlQfpPleYrpXT8GaoD_cg
# 

# Copy the template above, Replace <CHANGE ME> with your answer



######################################################################
#

#@title Solution

inv_problem = BaseProblem()
inv_problem.set_data(data_y)
inv_problem.set_jacobian(jacobian(n=2))
inv_problem.set_data_covariance_inv(Cd_inv())
inv_options.set_solving_method("matrix solvers") # lets decide to use a matrix solver.
inv = Inversion(inv_problem, inv_options)
inv_result = inv.run()

print("Inferred curve with n = 2 ")
plot_data()
plot_model(x,jacobian(x,n=2).dot(inv_result.model), "optimization solution", color="cornflowerblue")
plot_model(x,true_y, "true model", color="darkorange")

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# 3. Bayesian sampling
# --------------------
# 


######################################################################
# Likelihood
# ~~~~~~~~~~
# 
# Since data errors follow a Gaussian in this example, we can define a
# Likelihood function, :math:`p({\mathbf d}_{obs}| {\mathbf m})`.
# 
# .. math::
# 
# 
#    p({\mathbf d}_{obs} | {\mathbf m}) \propto \exp \left\{- \frac{1}{2} ({\mathbf d}_{obs}-{\mathbf d}_{pred}({\mathbf m}))^T C_D^{-1} ({\mathbf d}_{obs}-{\mathbf d}_{pred}({\mathbf m})) \right\}
# 
# where :math:`{\mathbf d}_{obs}` represents the observed y values and
# :math:`{\mathbf d}_{pred}({\mathbf m})` are those predicted by the
# polynomial model :math:`({\mathbf m})`. The Likelihood is defined as the
# probability of observing the data actually observed, given a model. In
# practice we usually only need to evaluate the log of the Likelihood,
# :math:`\log p({\mathbf d}_{obs} | {\mathbf m})`. To do so, we require
# the inverse data covariance matrix describing the statistics of the
# noise in the data, :math:`C_D^{-1}` . For this problem the data errors
# are independent with identical standard deviation in noise for each
# datum. Hence :math:`C_D^{-1} = \frac{1}{\sigma^2}I` where
# :math:`\sigma=1`.
# 

sigma = 1.0                                     # common noise standard deviation
Cdinv = np.eye(len(data_y))/(sigma**2)      # inverse data covariance matrix

def log_likelihood(model):
    y_synthetics = forward(model)
    residual = data_y - y_synthetics
    return -0.5 * residual @ (Cdinv @ residual).T

######################################################################
#


######################################################################
# Note that the user could specify **any appropriate Likelihood function**
# of their choosing here.
# 


######################################################################
# Prior
# ~~~~~
# 
# Bayesian sampling requires a prior probability density function. A
# common problem with polynomial coefficients as model parameters is that
# it is not at all obvious what a prior should be. Here we choose a
# uniform prior with specified bounds
# 
# .. math::
# 
# 
#    \begin{align}
#    p({\mathbf m}) &= \frac{1}{V},\quad  l_i \le m_i \le u_i, \quad (i=1,\dots,M)\\
#    \\
#             &= 0, \quad {\rm otherwise},
#    \end{align}
# 
# where :math:`l_i` and :math:`u_i` are lower and upper bounds on the
# :math:`i`\ th model coefficient.
# 
# Here use the uniform distribution with
# :math:`{\mathbf l}^T = (-10.,-10.,-10.,-10.)`, and
# :math:`{\mathbf u}^T = (10.,10.,10.,10.)`.
# 

m_lower_bound = np.ones(nparams) * (-10.)             # lower bound for uniform prior
m_upper_bound = np.ones(nparams) * 10                 # upper bound for uniform prior

def log_prior(model):    # uniform distribution
    for i in range(len(m_lower_bound)):
        if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
    return 0.0 # model lies within bounds -> return log(1)

######################################################################
#


######################################################################
# Note that the user could specify **any appropriate Prior PDF** of their
# choosing here.
# 


######################################################################
# Bayesian sampling
# ~~~~~~~~~~~~~~~~~
# 
# In this aproach we sample a probability distribution rather than find a
# single best fit solution. Bayes’ theorem tells us the the posterior
# distribution is proportional to the Likelihood and the prior.
# 
# .. math:: p(\mathbf{m}|\mathbf{d}) = K p(\mathbf{d}|\mathbf{m})p(\mathbf{m})
# 
# where :math:`K` is some constant. Under the assumptions specified
# :math:`p(\mathbf{m}|\mathbf{d})` gives a probability density of models
# that are supported by the data. We seek to draw random samples from
# :math:`p(\mathbf{m}|\mathbf{d})` over model space and then to make
# inferences from the resulting ensemble of model parameters.
# 
# In this example we make use of *The Affine Invariant Markov chain Monte
# Carlo (MCMC) Ensemble sampler* `Goodman and Weare
# 2010 <https://msp.org/camcos/2010/5-1/p04.xhtml>`__ to sample the
# posterior distribution of the model. (See more details about
# `emcee <https://emcee.readthedocs.io/en/stable/>`__).
# 


######################################################################
# Starting points for random walkers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now we define some hyperparameters (e.g. the number of walkers and
# steps), and initialise the starting positions of walkers. We start all
# walkers in a small ball about a chosen point :math:`(0, 0, 0, 0)`.
# 

nwalkers = 32
ndim = nparams
nsteps = 10000
walkers_start = np.zeros(nparams) + 1e-4 * np.random.randn(nwalkers, ndim)

######################################################################
#


######################################################################
# Add the information and run with CoFI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

######## CoFI BaseProblem - provide additional information
inv_problem.set_log_prior(log_prior)
inv_problem.set_log_likelihood(log_likelihood)
inv_problem.set_model_shape(ndim)

######## CoFI InversionOptions - get a different tool
inv_options_3 = InversionOptions()
inv_options_3.set_tool("emcee")      # Here we use to Affine Invariant McMC sampler from Goodman and Weare (2010).
inv_options_3.set_params(nwalkers=nwalkers, nsteps=nsteps, initial_state=walkers_start, progress=True)

######## CoFI Inversion - run it
inv_3 = Inversion(inv_problem, inv_options_3)
inv_result_3 = inv_3.run()

######## CoFI InversionResult - check result
print(f"The inversion result from `emcee`:")
inv_result_3.summary()

######################################################################
#


######################################################################
# Post-sampling analysis
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# By default the raw sampler resulting object is attached to ``cofi``\ ’s
# inversion result.
# 
# Optionally, you can convert that into an ``arviz`` data structure to
# have access to a range of analysis functions. (See more details in
# `arviz
# documentation <https://python.arviz.org/en/latest/index.html>`__).
# 

import arviz as az

labels = ["m0", "m1", "m2","m3"]

sampler = inv_result_3.sampler
az_idata = az.from_emcee(sampler, var_names=labels)
# az_idata = inv_result_3.to_arviz()      # alternatively

######################################################################
#

az_idata.get("posterior")

######################################################################
#

# a standard `trace` plot
axes = az.plot_trace(az_idata, backend_kwargs={"constrained_layout":True}); 

# add legends
for i, axes_pair in enumerate(axes):
    ax1 = axes_pair[0]
    ax2 = axes_pair[1]
    ax1.axvline(true_model[i], linestyle='dotted', color='red')
    ax1.set_xlabel("parameter value")
    ax1.set_ylabel("density value")
    ax2.set_xlabel("number of iterations")
    ax2.set_ylabel("parameter value")

######################################################################
#

tau = sampler.get_autocorr_time()
print(f"autocorrelation time: {tau}")

######################################################################
#

# a Corner plot

fig, axes = plt.subplots(nparams, nparams, figsize=(12,8))

if(False): # if we are plotting the model ensemble use this
    az.plot_pair(
        az_idata.sel(draw=slice(300,None)), 
        marginals=True, 
        reference_values=dict(zip([f"m{i}" for i in range(4)], true_model.tolist())),
        ax=axes,
    );
else: # if we wish to plot a kernel density plot then use this option
    az.plot_pair(
        az_idata.sel(draw=slice(300,None)), 
        marginals=True, 
        reference_values=dict(zip([f"m{i}" for i in range(4)], true_model.tolist())),
        kind="kde",
        kde_kwargs={
            "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
            "contourf_kwargs": {"cmap": "Blues"},
        },
        ax=axes,
    );

######################################################################
#


######################################################################
# Now we plot the predicted curves for the posterior ensemble of
# solutions.
# 

flat_samples = sampler.get_chain(discard=300, thin=30, flat=True)
inds = np.random.randint(len(flat_samples), size=100) # get a random selection from posterior ensemble

plot_data()
plot_models(flat_samples[inds])
plot_model(x,true_y, "True model", color="darkorange")

######################################################################
#


######################################################################
# Expected values, credible intervals and model covariance matrix from the ensemble
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

print("\n Expected value and 95% credible intervals ")
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
    print(" {} {:7.3f} [{:7.3f}, {:7.3f}]".format(labels[i],mcmc[1],mcmc[0],mcmc[2]))

######################################################################
#

CMpost = np.cov(flat_samples.T)
CM_std= np.std(flat_samples,axis=0)
print('Posterior model covariance matrix\n',CMpost)
print('\n Posterior estimate of model standard deviations in each parameter')
for i in range(ndim):
    print("    {} {:7.4f}".format(labels[i],CM_std[i]))

######################################################################
#


######################################################################
# --------------
# 


######################################################################
# Challenge: Change the prior model bounds
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Replace the previous prior bounds to new values
# 
# The original uniform bounds had
# 
# :math:`{\mathbf l}^T = (-10.,-10.,-10.,-10.)`, and
# :math:`{\mathbf u}^T = (10.,10.,10.,10.)`.
# 
# Lets replace with
# 
# :math:`{\mathbf l}^T = (-1.,-10.,-10.,-10.)`, and
# :math:`{\mathbf u}^T = (2.,10.,10.,10.)`.
# 
# We have only changed the bounds of the first parameter. However since
# the true value of constant term was 6, these bounds are now inconsistent
# with the true model.
# 
# What does this do to the posterior distribution?
# 
# |Upload to Excalidraw_2|
# 
# Start from the code template below:
# 
# ::
# 
#    m_lower_bound = <CHANGE ME>             # lower bound for uniform prior
#    m_upper_bound = <CHANGE ME>             # upper bound for uniform prior
# 
#    def log_prior(model):    # uniform distribution
#        for i in range(len(m_lower_bound)):
#            if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
#        return 0.0 # model lies within bounds -> return log(1)
# 
#    ######## CoFI BaseProblem - update information
#    inv_problem.set_log_prior(log_prior)
# 
#    ######## CoFI Inversion - run it
#    inv_4 = Inversion(inv_problem, inv_options_3)
#    inv_result_4 = inv_4.run()
# 
#    flat_samples = inv_result_4.sampler.get_chain(discard=300, thin=30, flat=True)
#    inds = np.random.randint(len(flat_samples), size=100) # get a random selection from posterior ensemble
# 
#    print("Resulting samples with prior model lower bounds of <CHANGE ME>, upper bounds of <CHANGE ME>")
#    plot_data()
#    plot_models(flat_samples[inds])
#    plot_model(x, true_y, "True model", color="darkorange")
# 
# .. |Upload to Excalidraw_2| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Excalidraw-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
#    :target: https://excalidraw.com/#room=feedcdce7f99f8ccd205,wTOddJkDrExrVTcxKsE6kA
# 

# Copy the template above, Replace <CHANGE ME> with your answer



######################################################################
#

#@title Solution

m_lower_bound = np.array([-1,-10,-10,-10])             # lower bound for uniform prior
m_upper_bound = np.array([2,10,10,10])                 # upper bound for uniform prior

def log_prior(model):    # uniform distribution
    for i in range(len(m_lower_bound)):
        if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
    return 0.0 # model lies within bounds -> return log(1)

######## CoFI BaseProblem - update information
inv_problem.set_log_prior(log_prior)

######## CoFI Inversion - run it
inv_4 = Inversion(inv_problem, inv_options_3)
inv_result_4 = inv_4.run()

flat_samples = inv_result_4.sampler.get_chain(discard=300, thin=30, flat=True)
inds = np.random.randint(len(flat_samples), size=100) # get a random selection from posterior ensemble

print("Resulting samples with prior model lower bounds of [-1,-10,-10,-10], upper bounds of [2,10,10,10]")
plot_data()
plot_models(flat_samples[inds])
plot_model(x, true_y, "True model", color="darkorange")

######################################################################
#


######################################################################
# Why do you think the posterior distribution looks like this?
# 


######################################################################
# --------------
# 


######################################################################
# Challenge: Change the data uncertainty
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# To change the data uncertainty we increase ``sigma`` and then redefine
# the log-Likelihood.
# 
# Here we increase the assumed data standard deviation by a factor of of
# 50! So we are telling the inversion that the data are far less accurate
# than they actually are.
# 

sigma = 50.0                                     # common noise standard deviation
Cdinv = np.eye(len(data_y))/(sigma**2)      # inverse data covariance matrix

def log_likelihood(model):
    y_synthetics = forward(model)
    residual = data_y - y_synthetics
    return -0.5 * residual @ (Cdinv @ residual).T

######################################################################
#


######################################################################
# Lets return the prior to the original bounds.
# 

m_lower_bound = np.ones(4) * (-10.)             # lower bound for uniform prior
m_upper_bound = np.ones(4) * 10                 # upper bound for uniform prior

def log_prior(model):    # uniform distribution
    for i in range(len(m_lower_bound)):
        if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
    return 0.0 # model lies within bounds -> return log(1)

######################################################################
#


######################################################################
# Your challenge is then to tell CoFI that the Likelihood and prior have
# changed and then to rerun the sample, and plot results.
# 
# |Upload to Excalidraw_3|
# 
# Feel free to start from the code below:
# 
# ::
# 
#    ######## CoFI BaseProblem - update information
#    inv_problem.set_log_likelihood(<CHANGE ME>)
#    inv_problem.set_log_prior(<CHANGE ME>)
# 
#    ######## CoFI Inversion - run it
#    inv_5 = Inversion(inv_problem, inv_options_3)
#    inv_result_5 = inv_5.run()
# 
#    flat_samples = inv_result_5.sampler.get_chain(discard=300, thin=30, flat=True)
#    inds = np.random.randint(len(flat_samples), size=100) # get a random selection from posterior ensemble
# 
#    print("Resulting samples from changed data uncertainty")
#    plot_data()
#    plot_models(flat_samples[inds])
#    plot_model(x,true_y, "True model", color="darkorange")
# 
# .. |Upload to Excalidraw_3| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Excalidraw-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
#    :target: https://excalidraw.com/#room=9a22d4eab8a9189d3b9b,9R8JKoHdOsbWlOLBj2V1fQ
# 

# Copy the template above, Replace <CHANGE ME> with your answer



######################################################################
#


######################################################################
# The answer is in the next cells if you want to run them.
# 

#@title Solution

######## CoFI BaseProblem - update information
inv_problem.set_log_likelihood(log_likelihood)
inv_problem.set_log_prior(log_prior)

######## CoFI Inversion - run it
inv_5 = Inversion(inv_problem, inv_options_3)
inv_result_5 = inv_5.run()

flat_samples = inv_result_5.sampler.get_chain(discard=300, thin=30, flat=True)
inds = np.random.randint(len(flat_samples), size=100) # get a random selection from posterior ensemble

print("Resulting samples from changed data uncertainty")
plot_data()
plot_models(flat_samples[inds])
plot_model(x,true_y, "True model", color="darkorange")

######################################################################
#


######################################################################
# Challenge: Change the number of walkers / steps in the McMC algorithm (optional)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now lets decrease the number of steps performed by the McMC algorithm.
# It will be faster but perform less exploration of the model parameters.
# 
# We suggest you reduce the number of steps taken by all 32 random walkers
# and see how it affects the posterior ensemble.
# 
# |Upload to Excalidraw_4|
# 
# You can start from code template below:
# 
# ::
# 
#    # change number of steps
#    nsteps = <CHANGE ME>              # instead of 10000
# 
#    # change number of walkers
#    nwalkers = <CHANGE ME>            # instead of 32
#    walkers_start = np.zeros(nparams) + 1e-4 * np.random.randn(nwalkers, ndim)
# 
#    # let's return to the old uncertainty settings
#    sigma = 1.0                                     # common noise standard deviation
#    Cdinv = np.eye(len(data_y))/(sigma**2)      # inverse data covariance matrix
# 
#    ######## CoFI InversionOptions - get a different tool
#    inv_options_3.set_params(nsteps=nsteps, nwalkers=nwalkers, initial_state=walkers_start)
# 
#    ######## CoFI Inversion - run it
#    inv_6 = Inversion(inv_problem, inv_options_3)
#    inv_result_6 = inv_6.run()
# 
#    ######## CoFI InversionResult - plot result
#    flat_samples = inv_result_6.sampler.get_chain(discard=300, thin=30, flat=True)
#    inds = np.random.randint(len(flat_samples), size=10) # get a random selection from posterior ensemble
# 
#    print(f"Inference results from {nsteps} steps and {nwalkers} walkers")
#    plot_data()
#    plot_models(flat_samples[inds])
#    plot_model(x,true_y, "True model", color="darkorange")
# 
# .. |Upload to Excalidraw_4| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Excalidraw-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
#    :target: https://excalidraw.com/#room=354b4a2d26551847421f,Cqqz1tpuHdT-g0dyMMqJlQ
# 

# Copy the template above, Replace <CHANGE ME> with your answer



######################################################################
#

#@title Solution

# change number of steps
nsteps = 400              # instead of 10000

# change number of walkers
nwalkers = 30             # instead of 32
walkers_start = np.zeros(nparams) + 1e-4 * np.random.randn(nwalkers, ndim)

# let's return to the old uncertainty settings
sigma = 1.0                                     # common noise standard deviation
Cdinv = np.eye(len(data_y))/(sigma**2)      # inverse data covariance matrix

######## CoFI InversionOptions - get a different tool
inv_options_3.set_params(nsteps=nsteps, nwalkers=nwalkers, initial_state=walkers_start)

######## CoFI Inversion - run it
inv_6 = Inversion(inv_problem, inv_options_3)
inv_result_6 = inv_6.run()

######## CoFI InversionResult - plot result
flat_samples = inv_result_6.sampler.get_chain(discard=300, thin=30, flat=True)
inds = np.random.randint(len(flat_samples), size=10) # get a random selection from posterior ensemble

print(f"Inference results from {nsteps} steps and {nwalkers} walkers")
plot_data()
plot_models(flat_samples[inds])
plot_model(x,true_y, "True model", color="darkorange")

######################################################################
#


######################################################################
# --------------
# 
# Where to next?
# --------------
# 
# - Linear regression with Eustatic Sea-level data - `link to
#   notebook <https://github.com/inlab-geo/cofi-examples/blob/main/examples/linear_regression/linear_regression_sealevel.ipynb>`__
# 


######################################################################
# --------------
# 
# Watermark
# ---------
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib", "emcee", "arviz"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#
# sphinx_gallery_thumbnail_number = -1