.. _api:

List of functions and classes (API)
===================================

.. automodule:: cofi

BaseSolver & BaseObjective
--------------------------

`BaseSolver` is the base class of all inversion solvers included in cofi.

`BaseObjective` is the base class of all forward problems.

.. autosummary ::
    :toctree: generated/

    cofi.BaseSolver
    cofi.BaseObjective
    cofi.Model
    cofi.BaseForward


cofi.optimisers
---------------

.. autosummary ::
    :toctree: generated/

    cofi.optimisers.ScipyOptimiserSolver
    cofi.optimisers.ScipyOptimiserLSSolver
    cofi.optimisers.TAOSolver


cofi.samplers
-------------

.. autosummary ::
    :toctree: generated/


cofi.utils
----------

.. autosummary ::
    :toctree: generated/
