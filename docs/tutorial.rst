*********
Tutorials
*********

This page contains explanation about concepts in CoFI and the basic usage.

Furthermore, we provide guidance over how to define a more general and flexible
problem, as well as how to plug in your own solvers in the
:ref:`Advanced usage <advanced_usage>` section.


Basic concepts
==============

In the workflow of :code:`cofi`, there are three main
components: :code:`BaseProblem`, :code:`InversionOptions`, and :code:`Inversion`.

- :code:`BaseProblem` defines three things: 1) the forward problem; 2) model parameter 
  space (the unknowns); and 3) other information about the objective you'd like to 
  reach. Depending on the inversion approaches you'd like to use, the last one can be 
  an objective function, or a log likelihood function, etc.
- :code:`InversionOptions` describes details about how one wants to run the inversion, including the
  inversion approach, backend tool and solver-specific parameters.
- :code:`Inversion` can be seen as an inversion engine that takes in the above two as information,
  and will produce an :code:`InversionResult` upon running.
  
For each of the above components, there's a :code:`summary()` method to check the current status.
  
So a common workflow includes 4 steps:

1. define :code:`BaseProblem`. This can be done:

- either: through a series of set functions

.. code::

  inv_problem = BaseProblem()
  inv_problem.set_objective(some_function_here)
  inv_problem.set_initial_model(a_starting_point)

- or: by subclassing :code:`BaseProblem`

.. code::

  class MyOwnProblem(BaseProblem):
      def __init__(self, initial_model, whatever_I_want_to_pass_in):
          self.initial_model = initial_model
          self.whatever_I_want_to_pass_in = whatever_I_want_to_pass_in
      def objective(self, model):
          return some_objective_function_value

2. define :code:`InversionOptions`. Some useful methods include:

- :code:`set_solving_method()` and :code:`suggest_tools()`. Once you've set a solving method (from "least squares"
  and "optimisation", more will be supported), you can use :code:`suggest_tools()` to see a list of backend tools
  to choose from.
      
.. admonition:: Ways to select inversion tools
  :class: attention

  We are working on enabling different ways to select the backend tool for different
  classes of audience.

3. start an :code:`Inversion`. This step is common:

   .. code::

    inv = Inversion(inv_problem, inv_options)
    result = inv.run()
   
4. analyse the result, workflow and redo your experiments with different :code:`InversionOptions`


.. _advanced_usage:

Advanced usage
==============

Define your own solver
----------------------


