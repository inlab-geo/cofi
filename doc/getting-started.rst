===============
Getting started
===============

.. attention::

    This page is still under construction. More precisely, I'm trying to figure out
    what best goes to "getting-started" and what goes to "tutorials".

Welcome! This tutorial provides basic usage and examples of CoFI. 


To discuss: objective or problem?
---------------------------------

We had some discussion over the term "objective", and are not sure whether moving to
"problem" would make it more intuitive to understand.

Reason: If we were to use the term "objective", we should distinguish it from "objective
function". If we further name "objective function" to be "misfit", then we should
distinguish it from "data misfit", as we assume "misfit" to also include regularisation.


To discuss: inversion options as a separate object?
---------------------------------------------------

option#1::

  from cofi import SomeSolver

  solver = SomeSolver(problem)
  solver.setIterationLimit(100)
  result = solver.solve()
  print(result.model)
  print(result.ok)


option#2::

  from cofi import BaseInversionOptions

  inversion = BaseInversionOptions()
  inversion.setMethod("optimisation")
  inversion.setTool("scipy.optimize.minimize")
  inversion.setIterationLimit(100)

  from cofi import Runner
  inversion_runner = Runner(problem, inversion)
  result = inversion_runner.run()
  print(result.model)
  print(result.ok)


Pre-defined inversion problem
-----------------------------

To use a pre-defined problem from inversion-test-suite::

  from inversion_test_suite import ExampleProblem

  problem = ExampleProblem.generate_basics()
  problem.setInitialModel(my_fancy_init_routine())


Self-defined inversion problem
------------------------------

To define a custom problem from scratch, there are 4 possible layers, depending the
level of flexibility you want.

Layer 0::
  
  from cofi import BaseProblem

  problem = BaseProblem()
  problem.setMisfit(objective_function)
  problem.setInitialModel(my_init_routine())

Layer 1::

  from cofi import BaseProblem

  problem = BaseProblem()
  problem.setDataMisfit(data_misfit_function)
  problem.setRegularisation(regularisation_function)
  problem.setInitialModel(my_init_routine())

Layer 2::

  from cofi import BaseProblem

  problem = BaseProblem()
  problem.setData("dataset.csv")
  problem.setForwardOperator(forward_function)
  problem.setDataMisfit("L2")
  problem.setRegularisation("L1")
  problem.setInitialModel(my_init_routine())

Layer 3:

.. code-block:: python
  
  from cofi import BaseProblem

  problem = BaseProblem()
  problem.setData("dataset.csv")
  problem.setForwardOperator("XRay Tomography")
  problem.setDataMisfit("L2")
  problem.setRegularisation("L1")
  problem.setInitialModel(my_init_routine())

