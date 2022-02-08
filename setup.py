# rm -rf _skbuild; pip install -e .

import sys


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
    # name="cofi",
    # version="0.1.0",

    # -------- BELOW FOR TEST_PYPI ------
    name="cofi_test",
    version="0.1.2",
    # -------- END TEST CONFIG --------

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
        "numpy>=1.22.2", 
        "scipy>=1.8.0", 
        "pyyaml>=6.0",
    ],
    extras_require={
        "petsc": ["petsc4py>=3.16.0"],
    },
)
