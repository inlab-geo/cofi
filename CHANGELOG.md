# Change Log

<!--next-version-placeholder-->

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
