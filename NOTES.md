These are personal notes from the meeting between Juerg, Malcolm, and me (Peter Rickwood) on 3/5/2021 at ANU, where we did some of the initial sketching of what CoFI might look like.

# Overall design choice

CoFI is intended to connect are forward solvers and inverse solvers.

Malcolm proposed, and Juerg and I agreed that this made sense, that the simplest way to do this is to actually connect inverse solvers with objective-functions. Objective-functions can themselves call forward solvers (either an existing one or one that the person has written themselves in fortran or C). So from an inverse point of view, we need a single objective function that can be called with different parameter values, and which will return a misfit/error (with higher being worse). This could be a squared error, or a negative log liklihood, or anything really, so long as higher is worse.

# People who want to bring their forward problem

Theer are 3 ways:

1. Bring a compiled language (e.g. C or Fortran) and be able to compile that into a shared object
2. Get COFI to do the compilation
3. Write their own python methods to interact with CoFI


# Yml objects

Discussed the following key objects that would be specified in yml

Model:
	model_params:
		A:
			type: vector float32
			initial_values: 1,2,3
		VP:
			type: vector float32
			initial_values: 2,3,4


Observation:
	aux_data
	observed
	prredicted

ForwardSolver:
	library_name
	hessian
	jacobian

Objective:
	??

Inverse Solver:







