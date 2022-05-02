# rm -rf _skbuild; pip install -e .

import sys
import pathlib

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

# get version number
_ROOT = pathlib.Path(__file__).parent
with open(str(_ROOT / "cofi" / "_version.py")) as f:
    for line in f:
        if line.startswith("__version__ ="):
            _, _, version = line.partition("=")
            VERSION = version.strip(" \n'\"")
            break
    else:
        raise RuntimeError("unable to read the version from ./_version.py")


setup(
    name="cofi",
    version=VERSION,
    description="Common Framework for Inference",
    author="InLab",
    packages=[
        "cofi",
        "cofi.inv_problems",
        "cofi.solvers",
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
