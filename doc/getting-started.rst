===============
Getting started
===============

Welcome! This tutorial provides basic usage and examples of CoFI. 


Installation
============

The simplest way to install CoFI is using ```pip```::

  pip install cofi

Alternatively you can build from source by following instructions in the Github `repository`_.
Then run::

  pip install .

or::

  pip install -e .

for development purpose.

Note that building from source involves more pre-requisites, including C/C++/Fortran compilers
and CMake.


A Minimal Example
=================

In order to do blabla, we can define the following objective::

  from cofi.cofi_objective import BaseObjective

  # blablabla
  # here are some code
  # TODO

Then, it's possible to pass the above example into several inverse solvers::

  from cofi.optimisers import *

  # biubiubiu
  # TODO
  # more explanation below #TODO

In principal, you are able to utilise the whole group of common use examples, or to customize
your own ones, in both the problems side and inverse solverse side.

Defining Your Own Problem
=========================

To define your own objective or forward solving workflow, extend the ```BaseObjective``` class.
For instance::

  from cofi.cofi_objective import BaseObjective

  class MyCurveFittingProblem(BaseObjective):
      def __init__(self, X, Y, forward, distance):
          # mamamimi hong
          # TODO

      def misfit(self, model: Model):
          # return something
          # TODO

Note that ```__init__``` and ```misfit``` are two functions that are generally required by
most inverse solvers. However for some other approaches, more functions may be required and
```misfit``` may not be necessary. Please check out API reference for details on what needs
to be implemented for the type of solvers you'd like to use.

Plugging In Your Own Solver
===========================

It's also easy to plug in your own inverse solver through the commonly defined interface.
Similarly, do this by extending the ```BaseSolver``` class.
For instance::

  from cofi import BaseSolver

  class MyDirectSearch(BaseSolver):
      def __init__(self, objective: BaseObjective):
          # mamamimi hong
          # TODO

      def solve(self, ) -> Model:
          # calculate something
          # TODO

For a more complete manual, please check out the API section.
  
