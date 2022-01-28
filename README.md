# CoFI (Common Framework for Inference)

[![Build](https://github.com/inlab-geo/cofi/actions/workflows/build_wheels.yml/badge.svg?branch=main)](https://github.com/inlab-geo/cofi/actions/workflows/build_wheels.yml)
[![Documentation Status](https://readthedocs.org/projects/cofi/badge/?version=latest)](https://cofi.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/inlab-geo/cofi/branch/main/graph/badge.svg?token=T8R9VKM4D7)](https://codecov.io/gh/inlab-geo/cofi)
[![Slack](https://img.shields.io/badge/Slack-inlab-4A154B?logo=slack)](https://inlab-geo.slack.com)


# Introduction

These are my main notes on the initial implementation of CoFI. Surely this will all be thrown away and done properly when someone has time to devote to doing it properly, but nevertheless this initial implementation will hopefully serve as a good initial scaffold on which to think clearly about CoFI design and implementation choices.

In the remainder of this document, I will discuss briefly the high level design choices made in this initial implementaton, and in various parts I discuss possible alternatives.


# What is CoFI?

CoFI is a framework that will help bring together people doing physical modelling (i.e. solving forward problems) and people working in inversion methods so that each can benefit from the others work without needing to be an expert in the other area. It is also anticipated that CoFI would be a good teaching platform, for students to experiement and learn about inversion and/or physcial modelling.

So there are four user groups that CoFI will target:
  1. Physical modellers (forward problem solvers)
  2. Inverse modellers
  3. Teachers
  4. Students

# Design 

## Language choice

Within each of the target groups, there will be a lot of heterogeneity in terms of the level of experience, the 
languages favoured (Matlab, python, Fortran, C++), and the libraries used. It is not possible to cater to all of these, so some pragmatic choices have to be made up front about what will be covered. Users, in whatever group, will have to interact with CoFI via either:
  * a simple YML file that defines their experiment/computation
  * code in a compiled language (Fortran, C, C++) that fits a particular specification (more on this in a moment)
  * code in python either through a python script or a jupyter notebook

It is too much work initially to try and support other languages or runtimes, such as (for example) Matlab.

## Simple install and examples

Given the heterogeneity of users, it is important that CoFI is as easy as possible to install and get started with.
The patience of users for complex installation, configuration, or use of CoFI will probably be the main limited
resource affecting CoFI's success. Success thus means:
  * Having a simple/clean installation
  * Having well documented examples that cover all the common cases of what people want to do
  * Having a lot of error handling code within CoFI itself to catch user errors and provide helpful error messages

### Installing CoFI

I have NOT made any attempt to do any neat packaging/install of CoFI. But I think it is crucial to CoFI's success.
So some thought (and work) needs to be put into packaging for CoFI. This probably means packaging CoFI up as a 
`conda` or `apt`/`deb` package.

In the absence of an installer, for testing the reference implementation, you will need the following installed:

  * python3.6 or later
  * All the typical python scientific libraries (scipy, numpy, pandas, etc etc). I just install everthing in 
   the jupyter/datascience-notebook docker image. 
  * C and fortran compilers (e.g. g++ gfortran)
  * blas and lapack (e.g. libblas-dev liblapack-dev). These are not part of CoFI but needed for one of the examples
  * you need the plotnine python package (just for plotting in the examples)
  * You'll also need to be set up to run a jupyter notebook if you want to step through some of the examples

I have included a Dockerfile in the CoFI repo which is what I have been using to develop and test in. You can refer
to that if you want to recreate the build environment. You can pull the actual image at docker.io/peterrickwood/cofi

## Forward model interface: `cofi_misfit` and `cofi_init`

In we want people to be able to bring along their own forward code and easily plug that in to CoFI, 
we need to define some standard functions that CoFI will use to interface with their code.
We discussed this at length in early May, and decided that the cleanest approach was to require
the user to define a *misfit* function: given a model, calculate the error of that model. This 
is cleanest because it means we do not need to deal directly with the users forward model or with the
error/loss function. This is a good thing, because we can cover the case where a user just has some
simple misfit calculation (e.g. squared loss between predicted and observed), but also the case where
the misfit represents a probability density for inclusion in somethign like MCMC.

So we require that the user define a function `cofi_misfit` that calculates a misfit (a single float)
based on an arbitrary number of model parameters. This is very similar to the interface that is used by 
`scipy.optimize`, with the main difference being that rather than requiring a single parameter vector
(as required in `scipy.optimze`), we will allow an arbitrary number of parameters, each of arbitrary shape,
to `cofi_misfit`. So you can have `cofi_misfit(A, v, x)` where your model is made up of A (a matrix), v (a vector)),
and x (a scalar). 

See the examples to see how this works out in practice.

We also require a function `cofi_init` to be defined. This is called **once**, before anything else, and 
gives the user code a chance to do any required setup/processing prior to inversion. This might involve 
reading in files, doing preliminary calculations, etc.

So, the general setup for CoFI is:

  * User defines `cofi_init(...)` and `cofi_misfit(...)` routines, in either python or compiled
  * User defines their experiment in either YML or via python api calls (examples of which later)
  * CoFI performs the experiment by calling `cofi_init` once and then calling `cofi_misfit` many times.


## Inverse model interface

We want to make it as easy as possible for people to include inverse solvers into CoFI.

I dont have any examples of this so far. A python example would be easy, but a compiled-language 
solver would require some more thinking and work. 

## YAML structure

Here is the YAML structure I have landed on for the moment. There are 4 main sections: 
  * model specifies the model structure
  * init_info specifies any miscellaneous information needed for the experiment
  * fwd_code specifies where the forward/misfit code lives
  * method specifies what inverse solver to use, and any required arguments

Here is an example

```
# The model structure, consisting of a list of parameters, of any shape.
# Parameters can have an initial value, or can have a specified distribution (i.e. a pdf)
# specifying the distribution of values the parameter can take. Any distributiion
# in scipy.stats can be used for the pdf.
model: &idmodel 
  parameters:
    - name: x
      bounds: uniform 1 10  # uniform between 1 and 11
      value: 10             # Initial value 10
    - name:
      bounds: norm 0 1      # normal with mean 0 stddev 1
      value: 0              # initial value 0

# General configuration information can be specified as arbitrary key/value pairs
# This information is passeed into cofi_init() on first call, and allows
# the user to keep config information and some parameters out of their code
# and in their yaml config, if they want
init_info:
    - item1: 2.0
    - item2: 42.0
    - item3: 60065
    .....

# User supplied forward code and misfit calculation 
fwd_code: &idfwd
  name: rfc        # the name of the module
  location: path   # the path to where the code lives (either python or compiled code)

method:
  # The name of the inverter. This must be something that is packaged with CoFI
  # We can allow users to plug in their own inversion code, but need to work out how 
  # best to do this. 
  name: DumbDescent 
  # Every inverter takes *at least* two mandatory arguments:
  #     * 'model' specifies the structure of the model being inverted (at this stage I
  #       havent thought about how to deal with reversible jump MCMC type models where
  #       the dimension of the model space is itself part of the search space)
  #     * 'forward' links to the forward code to be used in the inversion. This *must*
  #       implement cofi_misfit() and cofi_init() functions.
  #
  # Additional arguments will be specific to the inverter. In this case, the inverter takes
  # 'step' and 'time' arguments, but different inverters will take different arguments,
  # and you would specify them here.
  args:
      # These first 2 arguments are common to every inverse solver
      model: *idmodel
      forward: *idfwd
      # The remaining ones are arbitrary key/value pairs specific to the solver being used
      time: 30  # run for 30 seconds
      step: 0.2 # size of step in model search space
 
```
          


# Examples

There is a separate notebook for each of the following examples:

  * Running a CoFI inversion from a YAML file, using a compiled forward problem (receiver function) 
    and one of CoFI's in-built inverse solvers
  * Running a CoFI inversion from python, using a compiler forward problem (receiver function) and
    one of CoFI's in-built inverse solvers
  * Running a CoFI inversion from YAML, using a python-defined forward problem (rosenbrock) and
    one of CoFI's in-built inverse solvers
  * Running a CoFI inversion from python, using a python-defined forward problem (rosenbrock) and
    one of CoFI's in-built inverse solvers

At the moment, CoFI only has a single, very simple, inverse-solver/optimizer, which I have implemented
just to test the interfaces. It is called `DumbDescent` and is used in all the above examples.  

# Things still to do

OK, so I hope what we have is useful as a start, but there are still 3 things that I havent tackled
that really need to be addressed:
  1. *How will pepole plug in their own inverse solvers?* Discussed above, but at least 1 example of
     each type (python or compiled language) are required ASAP so we can better appeciate the work 
     required.
  2. *Flesh out yaml spec and interface* More examples should be implemented so we can flesh out 
     the YAML specification, and test how well the `cofi_init`/`cofi_misfit` interface allows us
     to cover those examples
  3. *Gradient-based inversion/optimization* I've only covered direct search/optimization so far.
     I am not clear on the requirements for situations where the user wants to use the Hessian or
     Jacobian, so I haven't tried to tackle that in the examples so far. I guess this is really
     just a special case of 2.
  4. If possible, solve the packaging/installation problem, add detailed error checking/messaging,
     and some help for people authoring their yaml and auto-annotating their wrapper code.


