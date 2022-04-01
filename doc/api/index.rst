.. _api:

List of functions and classes (API)
===================================

.. automodule:: cofi

APIs for Basic Usage
--------------------

`BaseProblem` is the base class of all forward problems. It includes methods you
can use so as to define a forward problem.

`InversionOptions` is a holder for you to define how you'd like the inversion problem
to be solved.

With instances of `BaseProblem` and `InversionOptions` defined, they are passed InversionOptions
an `InversionRunner` and an inversion run gives a result object of the type `InversionResult`.


.. autosummary::
    :toctree: generated/
    :caption: Basic usage

    cofi.BaseProblem
    cofi.InversionOptions
    cofi.InversionRunner
    cofi.InversionResult


APIs for Advanced Usage
-----------------------

`BaseSolver` is 

.. autosummary::
    :toctree: generated/
    :caption: Advanced usage

    cofi.solvers.BaseSolver
    cofi.inv_problems.BaseForward
