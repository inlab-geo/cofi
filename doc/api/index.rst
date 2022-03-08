.. _api:

List of functions and classes (API)
===================================

.. automodule:: cofi

Base solver & objective
-----------------------

`BaseSolver` is the base class of all inversion solvers included in cofi.

`BaseObjective` is the base class of all forward problems.

.. autosummary::
    :toctree: generated/
    :caption: Base classes

    cofi.BaseSolver
    cofi.BaseObjective
    cofi.Model
    cofi.BaseForward


Solvers - optimisation based
----------------------------

.. autosummary::
    :toctree: generated/
    :caption: Inversion - optimisation

    cofi.optimisers.ScipyOptimiserSolver
    cofi.optimisers.ScipyOptimiserLSSolver
    cofi.optimisers.TAOSolver


Solvers - sampling based
------------------------

.. autosummary::
    :toctree: generated/
    :caption: Inversion - sampling


Utility
-------

.. autosummary::
    :toctree: generated/
    :caption: Utility

Examples of CoFI objectives
---------------------------

.. autosummary::
    :toctree: generated/
    :caption: Example objectives

    cofi.cofi_objective.ExpDecay
    cofi.cofi_objective.ReceiverFunctionObjective
    cofi.cofi_objective.ReceiverFunction
    cofi.cofi_objective.XRayTomographyObjective
    cofi.cofi_objective.XRayTomographyForward
