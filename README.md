

# <img src="docs/_static/latte_art_cropped.png" width="5%" padding="0" margin="0"/> CoFI (Common Framework for Inference)


[![PyPI version](https://img.shields.io/pypi/v/cofi?logo=pypi&style=flat-square&color=bde0fe)](https://pypi.org/project/cofi/)
[![build](https://img.shields.io/github/workflow/status/inlab-geo/cofi/Build?logo=githubactions&style=flat-square&color=ccd5ae)](https://github.com/inlab-geo/cofi/actions/workflows/build_wheels.yml)
[![Documentation Status](https://img.shields.io/readthedocs/cofi?logo=readthedocs&style=flat-square&color=faedcd)](https://cofi.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://img.shields.io/codecov/c/github/inlab-geo/cofi?logo=pytest&style=flat-square&token=T8R9VKM4D7&color=f7d6e0)](https://codecov.io/gh/inlab-geo/cofi)
[![Slack](https://img.shields.io/badge/Slack-inlab-4A154B?logo=slack&style=flat-square&color=cdb4db)](https://inlab-geo.slack.com)
<!-- [![Wheels](https://img.shields.io/pypi/wheel/cofi)](https://pypi.org/project/cofi/) -->


# Introduction

CoFI (Common Framework for Inference) is an open-source initiative for interfacing between generic inference algorithms and specific geoscience problems.

With a mission to bridge the gap between the domain expertise and the inference expertise, this Python package provides an interface across a wide range of inference algorithms from different sources, as well as ways of defining inverse problems with ***examples*** included [here](https://github.com/inlab-geo/cofi-examples).

> This project and [documentation](https://cofi.readthedocs.io/en/latest/) are under initial development stage. Please feel free to contact us for feedback or issues!

```mermaid
graph TD;

    cofi(CoFI - Common Framework for Inference):::cls_cofi;
    parameter_estimation(Parameter estimation):::cls_parameter_estimation;
    linear(Linear):::cls_parameter_estimation;
    non_linear(Non linear):::cls_parameter_estimation;
    linear_system_solvers(Linear system solvers):::cls_parameter_estimation;
    linear_solverlist(scipy.linalg.lstsq <br> ...):::cls_solvers;
    optimisation(Optimisation):::cls_parameter_estimation;
    opt_solverlist(scipy.minimize <br> PETSc <br> Rapid Optimization Library<br>...):::cls_solvers;
    ensemble_methods(Ensemble methods):::cls_ensemble_methods;
    direct_search(Direct Search):::cls_ensemble_methods;
    amc(Monte Carlo):::cls_ensemble_methods;
    amc_solverlist(Neighbourhood Algorithm <br> ...):::cls_solvers;
    ng(Deterministic):::cls_ensemble_methods;
    ng_solverlist(Nested grids <br> ...):::cls_solvers;
    bs(Bayesian Sampling):::cls_ensemble_methods;
    mcmc(McMC samplers):::cls_ensemble_methods;
    mcmc_solverlist(emcee <br> pyMC <br> ...):::cls_solvers;
    rjmcmc(Reversible jump McMC):::cls_ensemble_methods;
    rjmcmc_solverlist(RJ-mcmc):::cls_solvers;

    cofi --> parameter_estimation;
    parameter_estimation --> linear;
    linear --> linear_system_solvers;
    linear_system_solvers -.- linear_solverlist;
    parameter_estimation --> non_linear;
    non_linear --> optimisation;
    optimisation -.- opt_solverlist;

    cofi --> ensemble_methods;
    ensemble_methods --> direct_search;
    direct_search --> amc;
    amc -.- amc_solverlist;
    direct_search --> ng;
    ng -.- ng_solverlist;
    ensemble_methods --> bs;  
    bs --> mcmc;
    mcmc -.- mcmc_solverlist;
    bs --> rjmcmc;
    rjmcmc -.- rjmcmc_solverlist;

classDef cls_cofi fill:#f0ead2, stroke-width:0;
classDef cls_parameter_estimation fill:#e1eff6, stroke-width:0;
classDef cls_ensemble_methods fill:#e9edc9, stroke-width:0;
classDef cls_solvers fill:#eae4e9, stroke-width:0;
```


## Installation

It's optional, but recommended to use a virtual environment:

```console
conda create -n cofi_env python=3.8 scipy
conda activate cofi_env
```

Install `cofi` with:

```console
pip install cofi
```

## Basic Usage

CoFI API has flexible ways of defining an inversion problem. For instance:

```python
from cofi import BaseProblem

inv_problem = BaseProblem()
inv_problem.set_objective(my_objective_func)
```

Once a problem is defined, `cofi` can tell you what inference solvers you can use based on what level of
information you've provided:

```python
inv_problem.suggest_solvers()   # a list will be printed
```

Run an inversion with these lines:

```python
from cofi import InversionOptions, Inversion

inv_options = InversionOptions()
inv_options.set_solving_method("optimisation")
inv_options.set_iteration_limit(100)

inv = Inversion(inv_problem, inv_options)
result = inv.run()
print(result.ok)
print(result.model)
```

And now an inversion is completed! Check out our [example gallery](https://cofi.readthedocs.io/en/latest/cofi-examples/generated/index.html)
and [tutorial](https://cofi.readthedocs.io/en/latest/tutorial.html) pages for more advanced usages.

## Contributing

Interested in contributing? Please check out our [contributor's guide](https://cofi.readthedocs.io/en/latest/contribute.html).


## License

This project is distributed under a 2-clause BSD license. A copy of this license is 
provided with distributions of the software.
