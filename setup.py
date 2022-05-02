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
with open(str(_ROOT / "src" / "cofi" / "_version.py")) as f:
    for line in f:
        if line.startswith("__version__ ="):
            _, _, version = line.partition("=")
            VERSION = version.strip(" \n'\"")
            break
    else:
        raise RuntimeError("unable to read the version from src/cofi/_version.py")


# read the contents of the README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cofi",
    version=VERSION,
    author="InLab",
    description="Common Framework for Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords=["inversion", "inference", "python package", "geoscience", "geophysics"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: C",
        "Programming Language :: Fortran",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "petsc": ["petsc4py>=3.16.0"],
        "doc": [
            "sphinx", 
            "sphinx-book-theme", 
            "sphinx-panels", 
            "nbsphinx",
            "sphinx-togglebutton",
            "sphinx-autobuild"
        ],
        "test": ["pytest", "matplotlib", "coverage[toml]"],
    },
)
