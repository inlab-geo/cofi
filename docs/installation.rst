============
Installation
============

Pre-requisite
-------------

CoFI requires Python 3.6+, and the following dependencies:

- numpy>=1.18
- scipy>=1.0.0

Install
-------

.. tabbed:: PyPI

  It's optional, but recommended to use a virtual environment::

    conda create -n cofi_env python=3.8 scipy
    conda activate cofi_env

  Install CoFI with::

    pip install cofi

.. tabbed:: conda-forge

  Uploading to conda-forge is still work in progress. 
  
  It won't be long!

.. tabbed:: from source

  If you'd like to build from source, clone the repository::

    git clone https://github.com/inlab-geo/cofi.git
    cd cofi
    conda env create -f envs/environment.yml
    conda activate cofi_env
    pip install .

  or::

    pip install -e .

  for an editable installation.


.. hint::

  CoFI time!
  Check out our step-by-step `tutorials <tutorial.html>`_ or 
  `examples <cofi-examples/generated/index.html>`_ to get started.
