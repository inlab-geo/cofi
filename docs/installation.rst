============
Installation
============

Dependencies
------------

CoFI requires Python 3.8+, and the following dependencies:

- numpy>=1.20
- scipy>=1.0.0
- pyyaml>=6.0

PyPI
----

It's optional, but recommended to use a virtual environment::

  conda create -n cofi_env python=3.8 scipy
  conda activate cofi_env

Install CoFI with::

  pip install cofi


conda-forge
-----------

(WIP)


Install from source
-------------------

If you'd like to build from source, clone the repository::

  git clone https://github.com/inlab-geo/cofi.git
  cd cofi
  conda env create -f envs/environment.yml
  conda activate cofi_env
  pip install .

or::

  pip install -e .

for an edittable installation.
