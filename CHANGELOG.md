# Change Log

<!--next-version-placeholder-->

## v0.2.3 (31/07/2023)

### CoFI Utils

- A new class, `cofi.utils.EnsembleOfInversions`, has been introduced to replace `cofi.utils.run_multiple_inversions`

## v0.2.2 (17/07/2023)

### CoFI Utils

- `cofi.utils.run_multiple_inversions`, sequential and parallel options
- Make `cofi.utils._reg_base.CompositeRegularization` pickleable


## v0.2.1 (07/07/2023)

### CoFI Tools

- Better `cofi.simple_newton` solver (more numerically stable; addition of stopping criteria)


## v0.2.0 (05/06/2023)

### CoFI Core

- Rewrite `BaseProblem.set_regularization`

### CoFI Utils

- Rewrite and implement regularization utils:
  - `cofi.utils.BaseRegularization`
  - `cofi.utils.LpNormRegularization`
  - `cofi.utils.QuadraticReg`
  - `cofi.utils.ModelCovariance`
  - `cofi.utils.GaussianPrior`


## v0.1.3.dev2 (05/04/2023)

### CoFI Core

- Bug fix: avoid evaluating log_likelihood if prior is -np.inf
- Enable properties set at `BaseProblem` constructor, e.g. `cofi.BaseProblem(forward=my_fwd, model_shape=my_shape)`

### CoFI Tools

- Bug fix in `numpy.linalg.lstsq`

## v0.1.3.dev1 (15/03/2023)

### CoFI Core

- Use try block for functools.update_wrapper

## v0.1.3.dev0 (15/03/2023)

### CoFI Solvers -> CoFI Tools

- [#110](https://github.com/inlab-geo/cofi/issues/110) `BaseSolver` -> `BaseInferenceTool`

### CoFI Core

- `_base_problem._FunctionWrapper` improvements

## v0.1.2.dev25 (06/02/2023)

### CoFI Solvers

- Bug fixes in `BaseSolver`

### CoFI Utils

- [#108](https://github.com/inlab-geo/cofi/issues/108) Utility regularization for flattening and smoothing in 1D cases


## v0.1.2.dev24 (14/12/2022)

### Infrastructure

- [#84](https://github.com/inlab-geo/cofi/issues/84) Use versioningit in build process

## v0.1.2.dev23 (14/12/2022)

### CoFI Core

- [#91](https://github.com/inlab-geo/cofi/issues/91) Raise warning when people set solver params that are not in optional list
- [#97](https://github.com/inlab-geo/cofi/issues/97) Make walkers_start_pos a property of InversionOptions instead of BaseProblem
- [#98](https://github.com/inlab-geo/cofi/issues/98) Typo, wording fixes; shorten error messages

## v0.1.2.dev22 (23/11/2022)

### CoFI Core

- [#90](https://github.com/inlab-geo/cofi/issues/90) Replaced `BaseProblem.suggest_solvers` with `BaseProblem.suggest_tools`
- [#89](https://github.com/inlab-geo/cofi/issues/89) Avoid importing third party modules on `import cofi`

### CoFI Solvers

- [#92](https://github.com/inlab-geo/cofi/issues/92) List pytorch optim algorithms dynamically


## v0.1.2.dev21 (27/10/2022)

### CoFI Solvers

- torch.optim
  - return number of function evaluations
  - accept callback function
  - return better losses list
  - add this to docs tree

## v0.1.2.dev20 (25/10/2022)

### CoFI Solvers

- Internal bug fix in PyTorch optimizers: adding "success" key in returned dictionary

## v0.1.2.dev19 (25/10/2022)

### CoFI Solvers

- In solvers table: `pytorch` -> `torch.optim`

## v0.1.2.dev18 (25/10/2022)

### CoFI Solvers

- Adding PyTorch.optim algorithms


## v0.1.2.dev17 (19/10/2022)

### CoFI Solvers

- Simple newton
  - Fix dimension issue
  - return number of function evaluations


## v0.1.2.dev16 (18/10/2022)

### CoFI Core

- [#63](https://github.com/inlab-geo/cofi/issues/63) Minor restructure of `BaseSolver._assign_options()`
- Wording change in `BaseProblem.summary()`

## v0.1.2.dev15 (14/10/2022)

### CoFI Core

- Further explanation in `BaseProblem.summary()`

## v0.1.2.dev14 (14/10/2022)

### CoFI Core

- Made CoFI pure Python, requires >=3.7

## v0.1.2.dev13 (13/10/2022)

### CoFI Core

- Fix `BaseProblem.hessian_times_vector` and `BaseProblem.jacobian_times_vector` that
  are generated from provided hessian / jacobian functions, by squeezing the results
  to ensure 1D dimensions

### CoFI Solvers

- Fix `InversionResult` keys to include underscores (so that attributes can be accessed
  easily)

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
