===============
Troubleshooting
===============

Mac users should first update or reinstall Xcode, or at least the Xcode
Command Line Tools, so that scientific Python packages can be built from
source correctly.

.. code-block:: console

  $ xcode-select --install


.. tab-set::

  .. tab-item:: pgcore / pyGIMLi

    pyGIMLi depends on ``pgcore``, which provides the underlying C++ core used by
    pyGIMLi. Unlike many pure Python packages, ``pgcore`` may need to be built
    locally because suitable wheels are not always available for every platform
    or Python version.

    Use this option if ``pgcore`` or ``pyGIMLi`` does not install correctly from
    pip, or if you need to build against a specific Python version.

    **1. Install build prerequisites**

    You will need:

    - A C++17-capable compiler
    - CMake
    - CastXML
    - Boost with Python bindings
    - Your target Python interpreter

    .. important::

       Boost must be built with Python bindings for the exact Python version
       you are using. System Boost packages are often built against a different
       Python version.

    Example Boost build for Python 3.12:

    .. code-block:: console

      $ ./bootstrap.sh --with-python=python3.12
      $ ./b2 --with-python python=3.12

    Replace ``python3.12`` / ``3.12`` with your target Python version if needed.

    **2. Clone the gimli source repository**

    .. code-block:: console

      $ git clone https://github.com/gimli-org/gimli.git
      $ cd gimli

    **3. Create and enter a build directory**

    .. code-block:: console

      $ mkdir build
      $ cd build

    **4. Configure the build with CMake**

    Point CMake to your target Python interpreter and CastXML executable.

    .. code-block:: console

      $ cmake .. \
          -DPYVERSION=3.12 \
          -DPython_EXECUTABLE=/path/to/python3.12 \
          -DCASTXML_EXECUTABLE=/path/to/castxml

    If using a virtual environment, check the active Python path with:

    .. code-block:: console

      $ which python
      $ python --version

    Then use that path in the CMake command, for example:

    .. code-block:: console

      $ cmake .. \
          -DPYVERSION=3.12 \
          -DPython_EXECUTABLE=/path/to/cofi_env/bin/python \
          -DCASTXML_EXECUTABLE=/path/to/castxml

    **5. Build the C++ core**

    .. code-block:: console

      $ make gimli

    **6. Build the Python bindings**

    .. code-block:: console

      $ make pygimli

    **7. Make the local build visible to Python**

    Replace ``/path/to/gimli/build`` with the actual path to your build directory.

    .. code-block:: console

      $ export PYTHONPATH=/path/to/gimli/build:$PYTHONPATH

    On Linux, also add the compiled libraries to ``LD_LIBRARY_PATH``:

    .. code-block:: console

      $ export LD_LIBRARY_PATH=/path/to/gimli/build/lib:$LD_LIBRARY_PATH

    On macOS, use ``DYLD_LIBRARY_PATH`` instead:

    .. code-block:: console

      $ export DYLD_LIBRARY_PATH=/path/to/gimli/build/lib:$DYLD_LIBRARY_PATH

    **8. Test that Python can import the local build**

    .. code-block:: console

      $ python -c "import pygimli; print(pygimli.__version__)"

    **9. Check which pyGIMLi installation Python is using**

    This is useful if Python imports a different version than expected.

    .. code-block:: console

      $ python -c "import pygimli; print(pygimli.__file__)"

    If CMake cannot find Python, pass the Python executable explicitly:

    .. code-block:: console

      $ cmake .. -DPython_EXECUTABLE=/path/to/your/python

    If CMake cannot find CastXML, pass the CastXML executable explicitly:

    .. code-block:: console

      $ cmake .. -DCASTXML_EXECUTABLE=/path/to/castxml

    If Boost Python errors occur, confirm that Boost was built against the same
    Python version used for this build.

    For more information, please refer to the
    `gimli source repository <https://github.com/gimli-org/gimli>`_.