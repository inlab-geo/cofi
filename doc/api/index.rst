.. _api:

List of functions and classes (API)
===================================

.. automodule:: cofi

Base classes for solvers and objectives
---------------------------------------

`BaseSolver` is the base class of all inversion solvers included in cofi.

`BaseObjective` is the base class of all forward problems.

.. autosummary ::
    :toctree: generated/

    cofi.BaseSolver
    cofi.BaseObjective
    cofi.Model
    cofi.BaseForward


Inversion solvers based on optimisation
----------------------------------

.. autosummary ::
    :toctree: generated/

    cofi.optimisers.ScipyOptimiserSolver
    cofi.optimisers.ScipyOptimiserLSSolver
    cofi.optimisers.TAOSolver


Inversion solvers based on sampling
-----------------------------------

.. autosummary ::
    :toctree: generated/


Utility
----------

.. autosummary ::
    :toctree: generated/

Examples of CoFI objectives
---------------------------

.. autosummary ::
    :toctree: generated/

    cofi.cofi_objective.ExpDecay
    cofi.cofi_objective.ReceiverFunctionObjective,
    cofi.cofi_objective.ReceiverFunction
    cofi.cofi_objective.XRayTomographyObjective
    cofi.cofi_objective.XRayTomographyForward
