

# <img src="https://raw.githubusercontent.com/inlab-geo/cofi/main/docs/_static/latte_art_cropped.png" width="5%" style="vertical-align:bottom"/> CoFI (Common Framework for Inference)


[![PyPI version](https://img.shields.io/pypi/v/cofi?logo=pypi&style=flat-square&color=bde0fe)](https://pypi.org/project/cofi/)
[![Documentation Status](https://img.shields.io/readthedocs/cofi?logo=readthedocs&style=flat-square&color=faedcd)](https://cofi.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://img.shields.io/codecov/c/github/inlab-geo/cofi?logo=pytest&style=flat-square&token=T8R9VKM4D7&color=f7d6e0)](https://codecov.io/gh/inlab-geo/cofi)
[![Slack](https://img.shields.io/badge/Slack-inlab-4A154B?logo=slack&style=flat-square&color=cdb4db)](https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg)
<!-- [![Wheels](https://img.shields.io/pypi/wheel/cofi)](https://pypi.org/project/cofi/) -->


## Introduction

CoFI (Common Framework for Inference) is an open-source initiative for interfacing between generic inference algorithms and specific geoscience problems.

With a mission to bridge the gap between the domain expertise and the inference expertise, CoFI provides an interface across a wide range of inference algorithms from different sources, underpinned by a rich set of domain relevant [examples](https://github.com/inlab-geo/cofi-examples).

> This project and [documentation](https://cofi.readthedocs.io/en/latest/) are under initial development stage. Please feel free to contact us for feedback or issues!


## Installation

```console
$ pip install cofi
```

Check CoFI documentation - 
[installation page](https://cofi.readthedocs.io/en/latest/installation.html) 
for details on dependencies and setting up with virtual environments.

## Basic Usage

CoFI API has flexible ways of defining an inversion problem. For instance:

```python
from cofi import BaseProblem

inv_problem = BaseProblem()
inv_problem.set_objective(my_objective_func)
inv_problem.set_initial_model(my_starting_point)
```

Once a problem is defined, `cofi` can tell you what inference solvers you can use based on what level of
information you've provided:

```python
inv_problem.suggest_solvers()   # a tree will be printed
```

Run an inversion with these lines:

```python
from cofi import InversionOptions, Inversion

inv_options = InversionOptions()
inv_options.set_solving_method("optimization")
inv_options.set_params(options={"maxiter":100})

inv = Inversion(inv_problem, inv_options)
result = inv.run()
print(result.success)
print(result.model)
```

And now an inversion is completed! Check out our [example gallery](https://cofi.readthedocs.io/en/latest/cofi-examples/tools/sphinx_gallery/generated/index.html)
and [tutorial](https://cofi.readthedocs.io/en/latest/tutorial.html) pages for more 
real-world or advanced use cases.

## Contributing

Interested in contributing? Please check out our [contributor's guide](https://cofi.readthedocs.io/en/latest/contribute.html).


## Licence

This project is distributed under a 2-clause BSD licence. A copy of this licence is 
provided with distributions of the software.
