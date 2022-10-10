# Change Log

<!--next-version-placeholder-->

## v0.1.2.dev12 (10/10/2022)

### CoFI Solvers

- `cofi.simple_newton`
  - hide options of line search (until line search is implemented)
  - prevent `initial_model` from being modified inplace

## v0.1.2.dev11 (10/10/2022)

### CoFI Core

- Minor fix (removing debug prints)
- [#72](https://github.com/inlab-geo/cofi/issues/72) `set_data_misfit` error message fix

### CoFI Utils

- Add `util` to `cofi` namespace by importing it

### CoFI Solvers

- `matrix-based solvers` -> `matrix solvers`

## v0.1.2.dev10 (03/10/2022)

### CoFI Core

- [#56](https://github.com/inlab-geo/cofi/issues/56) Modify BaseProblem.data_misfit to include data covariance matrix
- [#70](https://github.com/inlab-geo/cofi/issues/70) Words renaming optimise -> optimize, etc.

### CoFI Utils

- [#54](https://github.com/inlab-geo/cofi/issues/54) Utility functions using findiff to generate the difference matrices

### CoFI Solvers

- [#68](https://github.com/inlab-geo/cofi/issues/#68) Optimise special cases in linear system solver


## v0.1.2.dev9 (13/09/2022)

### CoFI Core

- Fixed potential problem in auto generated "times vector" functions when 
  input matrix might be of type `numpy.matrix`
- Enabled possibility for parallelism with emcee, by making user defined 
  functions pickleable
- [#53](https://github.com/inlab-geo/cofi/issues/53) Add set_regularisation(reg, reg_matrix, lamda)
- [#57](https://github.com/inlab-geo/cofi/issues/57) Create our own exception class
- [#59](https://github.com/inlab-geo/cofi/issues/59) Optimize import cofi by not importing cofi.solvers
- [#61](https://github.com/inlab-geo/cofi/issues/61) Remove lambda function from BaseProblem to avoid error in multiprocessing

### CoFI Solvers

- [#55](https://github.com/inlab-geo/cofi/issues/55) Linear solvers with Tikhonov regularisations


## v0.1.2.dev8 (13/07/2022)

- Bug fix in `_FunctionWrapper`, for functions with extra arguments like
  `BaseProblem.hessian_times_vector(m, v)` and `BaseProblem.jacobian_times_vector(m, v)`

## v0.1.2.dev7 (15/06/2022)

- Bug fix in `BaseSolver.model_covariance`

## v0.1.2.dev6 (15/06/2022)

- Bug fix in `BaseSolver._assign_options`
- `BaseProblem.model_covariance_inv` and `BaseProblem.model_covariance`

## v0.1.2.dev5 (09/06/2022)

- `BaseProblem.set_data_covariance` and more general linear system solver

## v0.1.2.dev4 (07/06/2022)

- Bugs fix in `EmceeSolver` and result summary

## v0.1.2.dev3 (06/06/2022)

- Bug fixed in `BaseProblem.set_regularisation`

## v0.1.2.dev2 (03/06/2022)

- Added `emcee` as new solver, with the following new APIs
  - `BaseProblem.set_log_prior`
  - `BaseProblem.set_log_likelihood`,
  - `BaseProblem.set_log_posterior`,
  - `BaseProblem.set_log_posterior_with_blobs`,
  - `BaseProblem.set_blobs_dtype`
- Process sampler output by converting to `arviz.InferenceData`, with the new API:
  - class `SamplingResult`
  - `SamplingResult.to_arviz()`
- Removed `BaseProblem.set_dataset(x,y)`, added `BaseProblem.set_data(y)`
- Added args and kwargs for all setting functions in `BaseProblem`
  - `_FunctionWrapper`
- Relaxed python version `>=3.8` to `>=3.6`
- Docs improvement, updated with emcee

## v0.1.2.dev1 (16/05/2022)

- bug fixes

## v0.1.2.dev0 (13/05/2022)

- `InversionRunner` has been changed into `Inversion`
- Added a references list for each backend tool
- How much information in `BaseProblem` is used for the inversion run
  now displayed through `Inversion.summary()`
- `numpy` and `scipy` versions relaxed
- Set objective function to be equal to data misfit if regularisation
  is not set
- Better error message when building failed
