.. CoFI documentation master file, created by
   sphinx-quickstart on Wed Nov 24 12:24:06 2021.

.. title:: Home

================================
Welcome to CoFI's documentation!
================================

.. What CoFI is

CoFI (**Co**\ mmon **F**\ ramework for **I**\ nference) is an open-source 
initiative for interfacing between generic inference algorithms and specific 
geoscience problems.

This project is led by `InLab <http://www.inlab.edu.au/>`_.

.. With a mission to bridge the gap between the domain expertise and the 
.. inference expertise, this Python package provides an interface across a 
.. wide range of inference algorithms from different sources, as well as ways 
.. of defining inverse problems with examples included.


.. mermaid::

    graph TD;
        cofi(CoFI - Common Framework for Inference):::cls_cofi;
        parameter_estimation(Parameter estimation):::cls_parameter_estimation;
        linear(Matrix based solvers):::cls_parameter_estimation;
        non_linear(Optimization):::cls_parameter_estimation;
        linear_system_solvers(Linear system solvers):::cls_parameter_estimation;
        linear_solverlist(scipy.linalg.lstsq <br>...):::cls_solvers;
        optimization(Non linear):::cls_parameter_estimation;
        optimization2(Linear):::cls_parameter_estimation;
        opt_solverlist(scipy.optimize.minimize <br> torch.optim <br> ROL <br>...):::cls_solvers;
        ensemble_methods(Ensemble methods):::cls_ensemble_methods;
        direct_search(Direct Search):::cls_ensemble_methods;
        amc(Monte Carlo):::cls_ensemble_methods;
        amc_solverlist(Neighbourhood Algorithm <br> Bayesian Optimization <br> Slime mold algorithm<br>...):::cls_solvers;
        ng(Deterministic):::cls_ensemble_methods;
        ng_solverlist(Nested grids <br> Hilbert Curves<br>...):::cls_solvers;
        bs(Bayesian Sampling):::cls_ensemble_methods;
        mcmc(McMC samplers):::cls_ensemble_methods;
        mcmc_solverlist(Basic metropolis<br>Affine Invariance sampler<br>emcee <br> pyMC <br> ...):::cls_solvers;
        rjmcmc(Trans-D McMC):::cls_ensemble_methods;
        rjmcmc_solverlist(Basic Trans-D <br> RJ-mcmc):::cls_solvers;

        cofi --> parameter_estimation;
        parameter_estimation --> linear;
        linear --> linear_system_solvers;
        linear_system_solvers -.- linear_solverlist;
        parameter_estimation --> non_linear;
        non_linear --> optimization;
        non_linear --> optimization2;
        optimization -.- opt_solverlist;
        optimization2 -.- opt_solverlist;

        cofi --> ensemble_methods;
        ensemble_methods --> direct_search;
        direct_search --> amc;
        amc -.- amc_solverlist;
        direct_search --> ng;
        ng -.- ng_solverlist;
        ensemble_methods --> bs;  
        bs --> mcmc;
        mcmc -.- mcmc_solverlist;
        bs --> rjmcmc;
        rjmcmc -.- rjmcmc_solverlist;

    classDef cls_cofi fill: #d4a373, stroke-width:0;
    classDef cls_parameter_estimation fill: #ccd5ae, stroke-width:0;
    classDef cls_ensemble_methods fill: #e9edc9, stroke-width:0;
    classDef cls_solvers fill: #faedcd, stroke-width:0;



.. grid:: 1 2 2 2
    :margin: 3
    :gutter: 4

    .. grid-item-card::
        :link: installation.html
        :text-align: center
        :class-card: card-border

        *New to CoFI?*
        ^^^^^^^^^^^^^^
        🐣 Start here

    .. grid-item-card::
        :link: api/index.html
        :text-align: center
        :class-card: card-border

        *Want details?*
        ^^^^^^^^^^^^^^^
        📑 API reference
    
    .. grid-item-card::
        :link: https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg
        :text-align: center
        :class-card: card-border

        *Have questions?*
        ^^^^^^^^^^^^^^^^^
        💬 Join our Slack workspace
    
    .. grid-item-card::
        :link: contribute.html
        :text-align: center
        :class-card: card-border

        *Contributions welcomed!*
        ^^^^^^^^^^^^^^^^^^^^^^^^^
        🛠 Developer guide


.. Table of contents
.. -----------------

.. toctree::
    :caption: Guides
    :hidden: 
    :maxdepth: 1

    introduction.rst
    installation.rst
    tutorials/generated/index.rst
    examples/generated/index.rst
    gallery/generated/index.rst
    faq.rst

.. toctree::
    :caption: Reference
    :hidden: 
    :maxdepth: 1

    api/index.rst
    changelog.md

.. toctree::
    :caption: Development
    :hidden: 
    :maxdepth: 1

    contribute.rst
    roadmap.md
    licence.rst
