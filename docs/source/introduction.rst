*************
Intro to CoFI
*************

With a mission to bridge the gap between the domain expertise and the 
inference expertise, ``cofi`` provides an interface across a 
wide range of inference algorithms from different sources, underpinned by a rich set 
of domain relevant `examples <examples/generated/index.html>`_.

This page contains explanation about the basic concepts of this package.

In the workflow of :code:`cofi`, there are three main
components: :code:`BaseProblem`, :code:`InversionOptions`, and :code:`Inversion`.

- :code:`BaseProblem` defines the inverse problem including any user supplied quantities such as data
  vector, number of model parameters and measure of fit between model predictions and data.

  .. code::

    inv_problem = BaseProblem()
    inv_problem.set_objective(some_function_here)     # if needed
    inv_problem.set_jacobian(some_function_here)      # if needed
    inv_problem.set_initial_model(a_starting_point)   # if needed
    # more could be set here
    # choose depending on the problem and how you want to solve it

- :code:`InversionOptions` describes details about how one wants to run the inversion, including the backend
  tool and solver-specific parameters. It is based on the concept of a method and tool.

  .. code::

    inv_options = InversionOptions()
    inv_options.suggest_solving_methods()
    inv_options.set_solving_method("matrix solvers")
    inv_options.suggest_tools()
    inv_options.set_tool("scipy.linalg.lstsq")
    inv_options.summary()

- :code:`Inversion` can be seen as an inversion engine that takes in the above two as information,
  and will produce an :code:`InversionResult` upon running.

  .. code::
    
    inv = Inversion(inv_problem, inv_options)
    result = inv.run()

Internally CoFI decides the nature of the problem from the quantities set by the user and performs
internal checks to ensure it has all that it needs to solve a problem.

For each of the above components, there's an associated :code:`summary()` method to check the 
current status.


.. hint::

  Congrats! You are on board. Click `here <installation.html>`_ if you haven't 
  installed CoFI locally yet. Otherwise, continue with 
  `tutorials <tutorials/generated/index.html>`_ for a step-by-step guide, or 
  `example gallery <examples/generated/index.html>`_ if you are eager to learn
  through examples.
