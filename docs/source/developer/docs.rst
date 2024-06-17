Editing documentation
^^^^^^^^^^^^^^^^^^^^^

It's very easy to edit or write documentation for CoFI. Start by cloning our GitHub
repository and setting up the environment, following instructions above - 
:ref:`fork & clone <fork_clone>` and :ref:`environment setup <env_setup>`.
Then head straight to the parts that you want to change, based on the mapping table
below:

.. list-table:: Table: documentation page mapping to file path
   :widths: 40 60
   :header-rows: 1

   * - Documentation page
     - File location
   * - `Home <index.html>`_
     - docs/index.rst
   * - `Installation <installation.html>`_
     - docs/installation.rst
   * - `Tutorials (front page) <tutorials/generated/index.html>`_
     - docs/tutorials/scripts/README.rst
   * - `Tutorials (tutorials content) <tutorials/generated/index.html>`_
     - `cofi-examples <https://github.com/inlab-geo/cofi-examples>`_ tutorials/tutorial_name.ipynb
   * - `Example gallery (front page) <examples/generated/index.html>`_
     - docs/examples/scripts/README.rst
   * - `Exmaple gallery (examples content) <examples/generated/index.html>`_
     - `cofi-examples <https://github.com/inlab-geo/cofi-examples>`_ examples/example_name/example_name.ipynb
   * - `Frequently asked questions <faq.html>`_
     - docs/faq.rst
   * - `List of functions and classes (API) <api/index.html>`_
     - docs/api/index.rst
   * - `API reference for BaseProblem <api/generated/cofi.BaseProblem.html>`_
     - src/cofi/base_problem.py
   * - `API reference for InversionOptions <api/generated/cofi.InversionOptions.html>`_
     - src/cofi/inversion_options.py
   * - `API reference for Inversion <api/generated/cofi.Inversion.html>`_
     - src/cofi/inversion.py
   * - `API refernece for InversionResult <api/generated/cofi.InversionResult.html>`_
     - src/cofi/inversion.py
   * - `API reference for BaseInferenceTool <api/generated/cofi.tools.BaseInferenceTool.html>`_
     - src/cofi/tools/base_inference_tool.py
   * - `Change Log <changelog.html>`_
     - CHANGELOG.md
   * - `Contribute to CoFI <contribute.html>`_
     - dos/contribute.rst

To change the **configuration** of this documentation, go change the content in file 
``docs/conf.py``.

To adjust the **styling** of pages, modify things in ``docs/_static/style.css`` and 
``docs/_templates``.

To **test** the changes, go to ``docs`` directory, run ``make html`` and open the file
``docs/_build/html/index.html`` in your browser to see the changes.

.. admonition:: reStructuredText
  :class: seealso

  All of the documentation (except for the ChangeLog part), including API references,
  use the `reStructuredText <https://en.wikipedia.org/wiki/ReStructuredText>`_ format. 
  This is a textual file format that tends to be more powerful compared to markdown.

  For the purpose of CoFI documentation, a good resource for reStructuredText syntax is
  the `sample project <https://sphinx-themes.org/sample-sites/sphinx-book-theme/>`_ (we
  link to the Book theme, the one this documentation uses), with the 
  `sources here <https://github.com/sphinx-themes/sphinx-themes.org/tree/master/sample-docs>`_ 
  to refer to.


Build locally
-------------

To generate and check documentation locally, 

.. code:: console

    $ cd docs
    $ make html
    $ python -m http.server -d build/html

Put ``localhost:8000`` into your browser address bar to read the generated 
documentation.
