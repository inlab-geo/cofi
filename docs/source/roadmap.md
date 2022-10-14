# CoFI Roadmap

This page contains both short-term and long-term goals of the CoFI project, and is a 
live snapshot of what we plan to invest our time and effort in.


## Solver libraries

Expanding the ***breadth*** of our supported solver libraries is one of the top 
priorities. Our initial targets are listed in the solver's tree you've already seen 
in the [home page](index.rst).


## Example gallery

This is another prioritised task to be done. Here's a list of examples we'd like to add
within the next few months:

- Shaw problem
- Earthquake location 
- X-ray tomography
- Seismic wave travel time tomography
- Receiver function
- Forward simulation with PyGIMLI (2D ERT, 2D DCIP, etc.)
- Forward simulation with SimPEG (gravity, electromagnetics, etc.)
- (to be continued)


## User interface

We aim to find a relatively stable public API that guarantees user experience and
extensibility. The current status of CoFI includes three objects that represent
inference problem (`BaseProblem`), inference options (`InversionOptions`) as well as a 
"running engine" (`Inversion`). Additionally, the actual library wrappers 
(subclasses of `BaseSolver`) are exposed to more advanced users for them to make it 
easier to plug in their own solver or do some hacking.

The details of the APIs mentioned above will be tested by first group of users and 
adjusted according to their feedbacks.


### Suggesting system

For the time being, CoFI's `InversionOptions` has a minimal suggesting system that
categorises all the backend tools with a layer of solving method. However, such a 
flattened list of solving methods does not reflect much about the relevance and
difference between different tools.

Thoughts about a better suggesting system is under an initial stage, and we want to
suit three types of audience: people who are certain about which solving approach they
want to use, people who just wants to use CoFI's wrapper on a certain tool, and others
who are not sure about how to solve their inversion problem so need more advice.


## Documentation

The following documentation pages are planned to be written or expanded:

- ***Tutorials*** section, where guides are step-by-step and compact enough to present users
  with common use cases and best practices with CoFI.
- ***Solvers library*** section - this will be a brand new part of CoFI's documentation, 
  starting from the solver's tree you've seen in [home page](index.rst) to expand the
  details of each inference methods. This part will focus on theories instead of 
  technical details.
- Frequently asked questions (***FAQs***)


## Education materials

***Workshops and talks*** are to be arranged ideally more often in 2023, once we have a 
relatively stable API and richer set of examples established. These will be in the
field of both geoscience and more general scientific open source community.

Additionally, a set of ***course materials*** on inference theory is going to be 
developed with CoFI as its main tool of code demonstration.
