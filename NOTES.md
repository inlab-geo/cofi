These are personal notes started at the meeting between Juerg, Malcolm, and me (Peter Rickwood) on 3/5/2021 at ANU, where we did some of the initial sketching of what CoFI might look like.

# Overall design choice

CoFI is intended to connect are forward solvers and inverse solvers.

Malcolm proposed, and Juerg and I agreed that this made sense, that the simplest way to do this is to actually connect inverse solvers with objective-functions. Objective-functions can themselves call forward solvers (either an existing one or one that the person has written themselves in fortran or C). So from an inverse point of view, we need a single objective function that can be called with different parameter values, and which will return a misfit/error (with higher being worse). This could be a squared error, or a negative log liklihood, or anything really, so long as higher is worse.

# People who want to bring their forward problem

There are 3 ways:

1. Bring code in a compiled language (e.g. C or Fortran) and be able to interface with CoFI
2. Get COFI to do the compilation
3. Write their own python methods to interact with CoFI


## Way number 1: Bring code in a compiled language

The main difficulty that has to be tackled here is making the final compile/link step as easy as possible. We would like a CoFI
user to be able to do something like this:
	1. Bring their own C/C++/Fortran code along, and compile it more or less as they normally would when working outside CoFI
	2. Write a minimal C or Fortran 'wrapper' that exposes the `cofi_misfit` and `cofi_init` functions required by CoFI
	3. Tell CoFI where the compiled code is (from 1), and where the CoFI interface code is (from 2), and get CoFI to do the rest 
	   (i.e. compile and combine the CoFI and the user code, and generate the necessary python interfaces)

Unfortunately, getting CoFI to do step 3 well, with good error handling and useful error messages if the user does something wrong,
will be a significant undertaking. So for the moment, I am assuming that the user is capable of doing steps 2 and 3 if we 
make it relatively simple for them. So we require that the user follows the following steps to work with CoFI (for compiled languages):

	1. Compile their own C/C++/Fortran code into a static library (e.g. `rfc.a`). This won't really require them to do anything
	   different from what they usually do, except they would usually compile to execuutable whereas we want them to
           compile to a static library. This is hopefully not too difficult a change. I include an example with the RFC code.
	   So far they don't need to worry about CoFI at all: they just need to compile to a static `.a` file.
	2. They need to write some minimal CoFI wrapper code (defining the `cofi_init` and `cofi_misfit`) functions. This is not
           a lot of work (see the example for rfc), but does require them to provide comment hints, as described here 
	   under the 'Easy and Smart Way` here: (https://numpy.org/doc/stable/f2py/f2py.getting-started.html). While this is
           not much work, it will present a barrier to some users. 	
	3. They need to compile and link their non-CoFI code (from 1) with their CoFI wrapper code (from 2). This *should*
           just be an easy 1-line build step (e.g. `f2py -m rfc -c rfcofi.f90 -llapack code/rfc.a`), but it does 
           require them to get the linking right (i.e. to specify `-llapack` and any other build flags needed to 
           successfully load and run their code).

I do not see how steps 1 and 2 can be avoided. Step 1 is necessary because we cannot replicate an entire build and
link system within CoFI, given the diversity of code that pepole might want to use to interface with CoFI. Step 2 is 
necessary because the user has to do some minimal amount of work to define a misfit function, and we cannot hope to 
automate this. What we *might* be able to automate is insertion of `F2PY` hints required in the interface code, and the 
actual build stepi (step 3), but these will require a deal of work to get right, handle errors gracefully, and provide the
user with useful error messages.


## Way number 2: Get COFI to do the compilation

This is discussed already a little above. The aim would be to reduce the requirement on the user to just compiling
their own code, and writing a minimal wrapper for `cofi_init()` and `cofi_misfit()`

## Way number 3: Write their own python methods to interact with CoFI



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
	predicted

ForwardSolver:
	library_name
	hessian
	jacobian

Objective:
	??

Inverse Solver:







