

# <img src="https://raw.githubusercontent.com/inlab-geo/cofi/main/docs/source/_static/latte_art_cropped.png" width="5%" style="vertical-align:bottom"/> CoFI (Common Framework for Inference)

[![PyPI version](https://img.shields.io/pypi/v/cofi?logo=pypi&style=flat-square&color=cae9ff&labelColor=f8f9fa)](https://pypi.org/project/cofi/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/cofi.svg?logo=condaforge&style=flat-square&color=cce3de&labelColor=f8f9fa&logoColor=344e41)](https://anaconda.org/conda-forge/cofi)
[![Documentation Status](https://img.shields.io/readthedocs/cofi?logo=readthedocs&style=flat-square&color=fed9b7&labelColor=f8f9fa&logoColor=eaac8b)](https://cofi.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://img.shields.io/codecov/c/github/inlab-geo/cofi?logo=pytest&style=flat-square&token=T8R9VKM4D7&color=ffcad4&labelColor=f8f9fa&logoColor=ff99c8)](https://codecov.io/gh/inlab-geo/cofi)
[![Slack](https://img.shields.io/badge/Slack-InLab_community-4A154B?logo=slack&style=flat-square&color=cdb4db&labelColor=f8f9fa&logoColor=9c89b8)](https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg)
<!-- [![Wheels](https://img.shields.io/pypi/wheel/cofi)](https://pypi.org/project/cofi/) -->

> Related repositories by [InLab](https://inlab.edu.au/community/):
> - [CoFI Examples](https://github.com/inlab-geo/cofi-examples)
> - [Espresso](https://github.com/inlab-geo/espresso)

## Introduction

CoFI (Common Framework for Inference) is an open source initiative for interfacing between generic inference algorithms and specific geoscience problems.

With a mission to bridge the gap between the domain expertise and the inference expertise, CoFI provides an interface across a wide range of inference algorithms from different sources, underpinned by a rich set of domain relevant [examples](https://cofi.readthedocs.io/en/latest/examples/generated/index.html).

Read [the documentation](https://cofi.readthedocs.io/en/latest/), and let us know your feedback or any issues!

## Installation

From PyPI:

```console
$ pip install cofi
```

Or alternatively, from conda-forge:

```console
$ conda install -c conda-forge cofi
```

Check CoFI documentation - 
[installation page](https://cofi.readthedocs.io/en/latest/installation.html) 
for details on dependencies and setting up with virtual environments.

## Basic Usage

```mermaid
graph TD;
    base_problem_details(inv_problem = BaseProblem#40;#41;\ninv_problem.set_objective#40;DEFINE ME#41;\ninv_problem.set_jacobian#40;DEFINE ME#41;\ninv_problem.set_initial_model#40;DEFINE ME#41;):::cls_code_block
    inversion_options_details(inv_options = InversionOptions#40;#41;\ninv_options.set_tool#40;#34;scipy.linalg.lstsq#34;#41;):::cls_code_block
    inversion_details(inv=Inversion#40;inv_problem, inv_options#41;\nresult = inv.run#40;#41;):::cls_code_block

    subgraph base_problem ["Base Problem"]
        base_problem_details
    end

    subgraph inversion_options ["Inversion Options"]
        inversion_options_details
    end

    subgraph inversion ["Inversion"]
        inversion_details
    end

    base_problem --> inversion;
    inversion_options --> inversion;

    classDef cls_base_problem fill: oldlace, stroke-width: 0;
    classDef cls_inversion_options fill: oldlace, stroke-width: 0;
    classDef cls_inversion fill: lavender, stroke-width: 0;
    classDef cls_code_block fill: lightgrey, stroke-width: 0, text-align: left;

    class base_problem cls_base_problem;
    class inversion_options cls_inversion_options;
    class inversion cls_inversion;
```

CoFI API has flexible ways of defining an inversion problem. For instance:

```python
import cofi

inv_problem = cofi.BaseProblem()
inv_problem.set_objective(my_objective_func)
inv_problem.set_initial_model(my_starting_point)
```

Once a problem is defined, `cofi` can tell you what inference tools you can use based on what level of
information you've provided:

```python
inv_problem.suggest_tools()   # a tree will be printed
```

Run an inversion with these lines:

```python
inv_options = cofi.InversionOptions()
inv_options.set_tool("torch.optim")
inv_options.set_params(options={"num_iterations": 50, "algorithm": "Adam"})

inv = cofi.Inversion(inv_problem, inv_options)
result = inv.run()
print(result.success)
print(result.model)
```

And now an inversion is completed! Check out our [example gallery](https://cofi.readthedocs.io/en/latest/examples/generated/index.html)
and [tutorial](https://cofi.readthedocs.io/en/latest/tutorials/generated/index.html) pages for more 
real-world or advanced use cases.

## Contributing

Interested in contributing? Please check out our [contributor's guide](https://cofi.readthedocs.io/en/latest/contribute.html).


## Licence

This project is distributed under a 2-clause BSD licence. A copy of this licence is 
provided with distributions of the software.
