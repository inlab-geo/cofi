CoFI Tutorials
==============

This page contains explanations about concepts in CoFI and basic usage.

Furthermore, we provide guidance over how to define a more general flexible
problem, as well as how to plug in your own solvers in the "advanced usage"
section.

Basic concepts
--------------

In the workflow of `cofi`, there are three main 
components: `BaseProblem`, `InversionOptions`, and `Inversion`.

- `BaseProblem` defines three things: 1) the forward problem; 2) the inversion parameter (model) space; and 3) the objective function to be optimised
- `InversionOptions` describes details about how one wants to run the inversion, including the
  inversion approach, backend tool and solver-specific parameters.
- `Inversion` can be seen as an inversion engine that takes in the above two as information,
  and will produce an `InversionResult` upon running.
  
For each of the above components, there's a `summary()` method to check the current status.
  
So a common workflow includes 4 steps:
1. define `BaseProblem`. This can be done:
- either: through a series of set functions

.. code::
  inv_problem = BaseProblem()
  inv_problem.set_objective(some_function_here)
  inv_problem.set_initial_model(a_starting_point)
- or: by subclassing `BaseProblem`

.. code::
  class MyOwnProblem(BaseProblem):
      def __init__(self, initial_model, whatever_I_want_to_pass_in):
          self.initial_model = initial_model
          self.whatever_I_want_to_pass_in = whatever_I_want_to_pass_in
      def objective(self, model):
          return some_objective_function_value

2. define `InversionOptions`. Some useful methods include:

- `set_solving_method()` and `suggest_tools()`. Once you've set a solving method (from "least squares"
and "optimisation", more will be supported), you can use `suggest_tools()` to see a list of backend tools
to choose from.
      
3. start an `Inversion`. This step is common:

   .. code::
    inv = Inversion(inv_problem, inv_options)
    result = inv.run()
   
4. analyse the result, workflow and redo your experiments with different `InversionOptions`


Advanced usage
--------------