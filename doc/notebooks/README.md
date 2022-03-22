# CoFI Tutorials

This folder walks through concepts of CoFI and provide minimum examples to help understand them.

## Sections
1. [basic concepts](1_basic_concepts.ipynb)
2. [CoFI example - linear regression](2_cofi_example_linear_reg.ipynb)
3. [CoFI example - xray tomography](3_cofi_example_xrt.ipynb)
4. [CoFI example - receiver function](3_cofi_example_rfc.ipynb)

## Installation
Note that the first section is purely about the concepts, so you don't have to install `cofi` for section 1 (basic concepts).

If you'd like to install `cofi`, then the following steps may be a good reference:

(It's recommended to install it into a virtual environment either managed by `conda` or `venv`, etc.)

```bash
conda create -n cofi_env -y python=3.9 pip jupyterlab numpy matplotlib
conda activate cofi_env
```

And then:
```bash
pip install cofi
```

## Deletion
After you've finished this tutorial, you can safely delete cofi or the whole environment by:

```bash
pip uninstall cofi_test -y
```
or

```bash
conda deactivate
conda env remove -n cofi_test
```
