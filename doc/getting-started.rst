===============
Getting started
===============

.. attention::

    This page is still under construction. More precisely, I'm trying to figure out
    what best goes to "getting-started" and what goes to "tutorials".

Welcome! This tutorial provides basic usage and examples of CoFI. 


To use a pre-defined problem from inversion-test-suite::

  from inversion_test_suite import ExampleProblem

  problem = ExampleProblem.generate_basics()
  problem.setInitialModel(my_fancy_init_routine())


To define a custom problem from scratch, there are 4 possible layers, depending the
level of flexibility you want.

Layer 0::
  
  from cofi import BaseProblem

  problem = BaseProblem()
  problem.setMisfit(objective_function)
  problem.setInitialModel(my_init_routine())

