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

This project is led by `the InLab <http://www.inlab.edu.au/>`_.

.. With a mission to bridge the gap between the domain expertise and the 
.. inference expertise, this Python package provides an interface across a 
.. wide range of inference algorithms from different sources, as well as ways 
.. of defining inverse problems with examples included.


.. mermaid::

    graph TD;
        cofi(CoFI - Common Framework for Inference):::cls_cofi;
        parameter_estimation(Parameter estimation):::cls_parameter_estimation;
        linear(Linear):::cls_parameter_estimation;
        non_linear(Non linear):::cls_parameter_estimation;
        linear_system_solvers(Linear system solvers):::cls_parameter_estimation;
        linear_solverlist(scipy.linalg.lstsq <br> ...):::cls_solvers;
        optimisation(Optimisation):::cls_parameter_estimation;
        opt_solverlist(scipy.minimize <br> PETSc <br> Rapid Optimization Library<br>...):::cls_solvers;
        ensemble_methods(Ensemble methods):::cls_ensemble_methods;
        direct_search(Direct Search):::cls_ensemble_methods;
        amc(Monte Carlo):::cls_ensemble_methods;
        amc_solverlist(Neighbourhood Algorithm <br> ...):::cls_solvers;
        ng(Deterministic):::cls_ensemble_methods;
        ng_solverlist(Nested grids <br> ...):::cls_solvers;
        bs(Bayesian Sampling):::cls_ensemble_methods;
        mcmc(McMC samplers):::cls_ensemble_methods;
        mcmc_solverlist(emcee <br> pyMC4 <br> ...):::cls_solvers;
        rjmcmc(Reversible jump McMC):::cls_ensemble_methods;
        rjmcmc_solverlist(RJ-mcmc):::cls_solvers;

        cofi --> parameter_estimation;
        parameter_estimation --> linear;
        linear --> linear_system_solvers;
        linear_system_solvers -.- linear_solverlist;
        parameter_estimation --> non_linear;
        non_linear --> optimisation;
        optimisation -.- opt_solverlist;

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

    classDef cls_cofi fill:#f0ead2, stroke-width:0;
    classDef cls_parameter_estimation fill:#e1eff6, stroke-width:0;
    classDef cls_ensemble_methods fill:#e9edc9, stroke-width:0;
    classDef cls_solvers fill:#eae4e9, stroke-width:0;


.. .. seealso::

..     This project is led by `the InLab <http://www.inlab.edu.au/>`_.


.. panels::
    :header: text-center text-large
    :card: border-1 m-1 text-center

    **Installation**
    ^^^^^^^^^^^^^^^^^^^

    New to CoFI?

    .. link-button:: installation
        :type: ref
        :text: Start here
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Reference documentation**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A list of our functions and classes

    .. link-button:: api
        :type: ref
        :text: API reference
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Need support?**
    ^^^^^^^^^^^^^^

    Ask in our Slack workspace

    .. link-button:: https://inlab-geo.slack.com
        :type: url
        :text: Join the conversation
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Contribute to CoFI**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Features or bug fixes are always welcomed!

    .. link-button:: contribute
        :type: ref
        :text: Developer notes
        :classes: btn-outline-primary btn-block stretched-link


Table of contents
-----------------

.. toctree::
    :caption: Getting started
    :maxdepth: 1

    installation.rst
    tutorial.rst
    cofi-examples/generated/index.rst
    faq.rst

.. toctree::
    :caption: Reference
    :maxdepth: 1

    api/index.rst
    changelog.md

.. toctree::
    :caption: Developer notes
    :maxdepth: 1

    contribute.rst
    license.rst
