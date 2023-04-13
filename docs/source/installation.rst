============
Installation
============

Pre-requisites
--------------

CoFI requires Python 3.7+, and the following dependencies:

- numpy>=1.18
- scipy>=1.0.0
- emcee>=3.1.0
- arviz>=0.9.0
- findiff>=0.7.0
- torch>=1.10

Install
-------

.. tab-set::

  .. tab-item:: PyPI

    .. code-block:: console

      $ pip install cofi

  .. tab-item:: conda-forge

    .. code-block:: console

      $ conda install -c conda-forge cofi

  .. tab-item:: from source

    If you'd like to build from source, clone the repository

    .. code-block:: console

      $ git clone https://github.com/inlab-geo/cofi.git
      $ cd cofi

    And use either one of the following command to install

    .. code-block:: console

      $ pip install .
      $ pip install -e .      # (alternatively) developer mode



.. admonition:: Virtual environment setup (*strongly recommended*)
  :class: attention, dropdown

  It's optional, but recommended to use a virtual environment.

  .. tab-set::

    .. tab-item:: venv

      Ensure you have `python>=3.7`.

      To create:

      .. code-block:: console

        python -m venv <path-to-new-env>/cofi_env

      To activate:
      
      .. code-block:: console

        source <path-to-new-env>/cofi_env/bin/activate

      To exit:
      
      .. code-block:: console

        deactivate

      To remove:

      .. code-block:: console

        rm -rf <path-to-new-env>/cofi_env

    .. tab-item:: virtualenv

      To create:

      .. code-block:: console

        virtualenv <path-to-new-env>/cofi_env -p=3.10

      To activate:

      .. code-block:: console

        source <path-to-new-env>/cofi_env/bin/activate

      To exit:

      .. code-block:: console

        deactivate

      To remove:

      .. code-block:: console

        rm -rf <path-to-new-env>/cofi_env

    .. tab-item::  conda / mamba

      To create:

      .. code-block:: console

        conda create -n cofi_env python=3.10

      To activate:

      .. code-block:: console

        conda activate cofi_env

      To exit:

      .. code-block:: console

        conda deactivate

      To remove:
      
      .. code-block:: console

        conda env remove -n cofi_env


.. hint::

  CoFI time!
  Check out our step-by-step `tutorials <tutorials/generated/index.html>`_ or 
  `examples <examples/generated/index.html>`_ to get started.
