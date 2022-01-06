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
    packages=["cofi", "cofi.cofi_solvers", "cofi.cofi_objective"],
    install_requires=[
        'cython',
        'numpy',
        'scipy',
        'pyyaml',
    ],
)
