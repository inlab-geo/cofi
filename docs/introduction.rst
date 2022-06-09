*************
Intro to CoFI
*************

With a mission to bridge the gap between the domain expertise and the 
inference expertise, ``cofi`` provides an interface across a 
wide range of inference algorithms from different sources, underpinned by a rich set 
of domain relevant `examples <cofi-examples/generated/index.html>`_.

This page contains explanation about the basic concepts of this package.

In the workflow of :code:`cofi`, there are three main
components: :code:`BaseProblem`, :code:`InversionOptions`, and :code:`Inversion`.

- :code:`BaseProblem` defines three things: 1) the forward problem; 2) model parameter 
  space (the unknowns); and 3) options specific to type of inverse problem you are 
  trying to solve, such as the definition of an objective function for optimisation.
- :code:`InversionOptions` describes details about how one wants to run the inversion, including the
  inversion approach, backend tool and solver-specific parameters.
- :code:`Inversion` can be seen as an inversion engine that takes in the above two as information,
  and will produce an :code:`InversionResult` upon running.
  
For each of the above components, there's a :code:`summary()` method to check the current status.
  
So a common workflow includes 4 steps:

1. we begin by defining the :class:`BaseProblem`. This can be done through a series of set functions

.. code::

  inv_problem = BaseProblem()
  inv_problem.set_objective(some_function_here)
  inv_problem.set_initial_model(a_starting_point)

2. define :class:`InversionOptions`. 
   `Some useful methods <api/generated/cofi.InversionOptions.html>`_ include:

- :code:`set_solving_method()` and :code:`suggest_tools()`. Once you've set a solving method (from "least squares"
  and "optimisation", more will be supported), you can use :code:`suggest_tools()` to see a list of backend tools
  to choose from.
      
.. admonition:: Ways to suggest inversion tools
  :class: seealso

  We are working on enabling different ways to select the backend tool for different
  classes of audience, as discussed in the `roadmap <roadmap.html#suggesting-system>`_.

1. start an :class:`Inversion`. This step is common:

   .. code::

    inv = Inversion(inv_problem, inv_options)
    result = inv.run()
   
2. analyse the result, workflow and redo your experiments with different instances of
   :class:`InversionOptions`.

.. hint::

  Congrats! You are on board. Click `here <installation.html>`_ if you haven't 
  installed CoFI locally yet. Otherwise, continue with 
  `tutorials <tutorials/index.html>`_ for a step-by-step guide, or 
  `example gallery <cofi-examples/generated/index.html>` if you are eager to learn
  through examples.
