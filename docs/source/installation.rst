============
Installation
============

Install CoFI
------------

**Note**: When installing CoFI we recommend using Python version 3.12 for stability

**Step 1**: (*Optional*) Set up a virtual environment.

We strongly recommend installing CoFI within a 
`virtual environment <https://docs.python.org/3/tutorial/venv.html>`_. 
This ensures that CoFI can install the various modules that it needs without the 
risk of breaking anything else on your system. There are a number of tools that can 
facilitate this, including `venv`, `virtualenv`, `conda` and `mamba`.

.. dropdown:: Expand for how to manage virtual environments
  :icon: package

  .. tab-set::

    .. tab-item:: venv

      Ensure you have `python>=3.10`. Then, you can create a new virtual environment by 
      running the command:

      .. code-block:: console

        $ python -m venv <path-to-new-env>/cofi_env

      where :code:`<path-to-new-env>` is your prefered location for storing information 
      about this environment, and :code:`<env-name>` is your preferred name for the 
      virtual environmment. For example,

      .. code-block:: console

        $ python -m venv ~/my_envs/cofi_env 

      will create a virtual environment named :code:`cofi_env` and store everything 
      within a sub-directory of your home-space named :code:`my_envs`.

      To 'activate' or 'switch on' the virtual environment, run the command
    
      .. code-block:: console

        $ source <path-to-new-env>/<env-name>/bin/activate

      At this point you effectively have a 'clean' Python installation. You can now 
      install and use CoFI, following the instructions at step 2. When you are 
      finished, you can run the command
      
      .. code-block:: console

        $ deactivate

      and your system will return to its default state. If you want to use CoFI again, 
      simply re-run the 'activate' step above; you do not need to repeat the 
      installation process. Alternatively, you can remove CoFI and the virtual 
      environment from your system by running

      .. code-block:: console

        $ rm -rf <path-to-new-env>/<env-name>

    .. tab-item:: virtualenv

      You can create a new virtual environment (using Python version 3.10) by running 
      the command

      .. code-block:: console

        $ virtualenv <path-to-new-env>/<env-name> -p=3.10
      
      where :code:`<path-to-new-env>` is your prefered location for storing information 
      about this environment, and :code:`<env-name>` is your preferred name for the 
      virtual environmment. For example,

      .. code-block:: console

        $ virtualenv ~/my_envs/cofi_env -p=3.10

      will create a virtual environment named :code:`cofi_env` and store everything 
      within a sub-directory of your home-space named :code:`my_envs`.

      To 'activate' or 'switch on' the virtual environment, run the command

      .. code-block:: console

        $ source <path-to-new-env>/<env-name>/bin/activate

      At this point you effectively have a 'clean' Python installation. You can now 
      install and use CoFI, following the instructions at step 2. When you are 
      finished, you can run the command

      .. code-block:: console

        $ deactivate

      and your system will return to its default state. If you want to use CoFI again, 
      simply re-run the 'activate' step above; you do not need to repeat the 
      installation process. Alternatively, you can remove CoFI and the virtual 
      environment from your system by running

      .. code-block:: console

        $ rm -rf <path-to-new-env>/<env-name>

    .. tab-item::  conda / mamba

      You can create a new virtual environment (using Python version 3.10) by running 
      the command

      .. code-block:: console

        $ conda create -n <env-name> python=3.10

      where :code:`<env-name>` is your preferred name for the virtual environmment. 
      For example,

      .. code-block:: console

        $ conda create -n cofi_env python=3.10

      will create a virtual environment named :code:`cofi_env`.
      
      To 'activate' or 'switch on' the virtual environment, run the command

      .. code-block:: console

        $ conda activate <env-name>

      At this point you effectively have a 'clean' Python installation. You can now 
      install and use CoFI, following the instructions at step 2. When you are 
      finished, you can run the command
      
      .. code-block:: console

        $ conda deactivate

      and your system will return to its default state. If you want to use CoFI again, 
      simply re-run the 'activate' step above; you do not need to repeat the 
      installation process. Alternatively, you can remove CoFI and the virtual 
      environment from your system by running
      
      .. code-block:: console

        $ conda env remove -n <env-name>



**Step 2**: Install CoFI

.. tab-set::

  .. tab-item:: pip

    CoFI is available on `PyPI <https://pypi.org/project/cofi/>`_, so for most users
    installation is as simple as:

    1. Install CoFI from PyPI

      .. code-block:: console 

        $ pip install --upgrade pip
        $ pip install cofi

    2. Clone the examples repository

      .. code-block:: console  

        $ git clone https://github.com/inlab-geo/cofi-examples.git
        $ cd cofi-examples

    3. Install the full dependency set for the examples

      .. code-block:: console 

        $ pip install -r envs/requirements.txt

    4. Optional: run the examples in JupyterLab

      .. code-block:: console 

        $ jupyter-lab

  .. tab-item:: Docker

    CoFI is also available through the pre-built INLAB Docker image.

    1. Ensure Docker is installed

    2. Download the pre-built CoFI environment

       .. code-block:: console

          $ docker pull inlabgeo/inlab:latest

    3. Start the Docker container

       .. code-block:: console

          $ docker run -p 8888:8888 inlabgeo/inlab:latest

    4. Open JupyterLab in your browser

       Go to:

       .. code-block:: console

          http://127.0.0.1:8888

    5. Retrieve the access token from the terminal

       It will look similar to:

       .. code-block:: console

          http://27ddc0449990:8888/lab?token=fa9af2f42563a048c58316c690e416a01956238ba580e963

       Copy the token after ``token=`` and paste it into the
       ``Password or token`` field in JupyterLab.

  .. tab-item:: conda / mamba

    CoFI is also available on `conda-forge <https://anaconda.org/conda-forge/cofi>`_, 
    so you can install it with ``mamba`` or ``conda``:

    .. code-block:: console

      $ conda install -c conda-forge cofi

  .. tab-item:: From source

    You can build CoFI from source. You are most likely to want to do this if you 
    want to work in 'developer mode', and make changes to CoFI's source code.

    .. code-block:: console

      $ git clone https://github.com/inlab-geo/cofi.git
      $ cd cofi
      $ pip install -e .
    
    The :code:`-e` flag ensures that the module is installed in editable mode; you can 
    omit this if you do not intend to make any changes.


.. admonition:: CoFI time!
  :class: tip

  Check out our step-by-step `tutorials <tutorials/generated/index.html>`_ or 
  `examples <examples/generated/index.html>`_ to get started.


Container support
-----------------

CoFI and the InLab geoscience software ecosystem are now available in the form of 
a handy container image. We kindly refer you to the 
`inlab-containers guide <https://github.com/inlab-geo/inlab-containers#getting-started>`_
to get started.
