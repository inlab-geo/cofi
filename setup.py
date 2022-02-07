# rm -rf _skbuild; pip install -e .

import sys

# import numpy

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

setup(
    name="cofi",
    version="0.1.0",
    description="Common Framework for Inference",
    author="InLab",
    packages=[
        "cofi",
        "cofi.utils",
        "cofi.cofi_objective",
        "cofi.cofi_objective.examples",
        "cofi.linear_reg",
        "cofi.optimizers",
        "cofi.samplers",
    ],
    install_requires=[
        "cython>=0.29.27", 
        "numpy>=1.22.2", 
        "scipy>=1.8.0", 
        "pyyaml>=6.0", 
        "pybind11[global]>=2.9.1",
    ],
)
