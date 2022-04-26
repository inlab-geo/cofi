.. _api:

List of functions and classes (API)
===================================

.. automodule:: cofi

.. attention::

    This package is still under initial development stage, so public APIs are 
    not expected to be stable. Please stay updated and don't hesitate to raise
    feedback or issues through `GitHub issues <https://github.com/inlab-geo/cofi/issues/new/choose>`_ 
    or `Slack workspace <https://inlab-geo.slack.com>`_.

APIs for Basic Usage
--------------------

`BaseProblem` is the base class of all forward problems. It includes methods you
can use so as to define a forward problem.

`InversionOptions` is a holder for you to define how you'd like the inversion problem
to be solved.

With instances of `BaseProblem` and `InversionOptions` defined, they are passed InversionOptions
an `Inversion` and an inversion run gives a result object of the type `InversionResult`.


.. autosummary::
    :toctree: generated/
    :caption: Basic usage

    cofi.BaseProblem
    cofi.InversionOptions
    cofi.Inversion
    cofi.InversionResult


APIs for Advanced Usage
-----------------------

`BaseSolver` is the base class of all backend inversion tools. To plug in your own inversion
tools, simply create a subclass of `BaseSolver` and implements `__init__()` and `__call__()`.
Check the tutorials page (TODO) and reference page below for details.

.. autosummary::
    :toctree: generated/
    :caption: Advanced usage

    cofi.solvers.BaseSolver
    .. cofi.inv_problems.BaseForward
