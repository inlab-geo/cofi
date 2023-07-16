******************
Contribute to CoFI
******************

Great to see you here! There are many ways to contribute to CoFI.


Reporting issues
================

Bug reports and feature requests are welcome. Please search before lodging an issue at
our Github `repository`_.


Code contribution
=================

Thanks for making ``cofi`` better through coding! You don't have to know all the details
in order to contribute. If you feel confused and are unsure where to start, don't
hesitate to contact us via `GitHub issues <https://github.com/inlab-geo/cofi/issues/new/choose>`_
or `Slack`_.

General Workflow
----------------

Here is a general flowchart for code contribution, including preparation, editing and
submitting stages:

.. mermaid::

  %%{init: {'theme':'base'}}%%
    flowchart LR
      subgraph PREPARATION [ ]
        direction TB
        fork(fork repository)-->clone(create local clone)
        clone-->env_setup(environment setup)
      end
      subgraph EDIT [ ]
        direction TB
        code(start coding)-->commit(commit as needed)
        commit-->push(push to your own fork)
      end
      subgraph SUBMIT [ ]
        direction TB
        pr(create pull request)-->modify(edit based on our comments)
        modify-->commit_push(commit and push)
        commit_push-->merge(we merge it once ready)
        pr-->merge
      end
      PREPARATION-->EDIT
      EDIT-->SUBMIT

.. _fork_clone:

Fork and clone respository
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Navigate to the GitHub repository (`cofi <https://github.com/inlab-geo/cofi>`_,
   or `cofi-examples <https://github.com/inlab-geo/cofi-examples>`_ if you'd like to
   add or edit things in the `example gallery <examples/generated/index.html>`_)
2. Click the "Fork" button on top right of the page (followed by a confirmation page
   with a "Create fork" button)
3. Now you'll be redirected to your fork of ``cofi``, which is like a branch out in 
   your namespace. (And later you will see it merges back when your pull request is
   approved)

   .. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base', 'gitGraph': {'showCommitLabel': false}} }%%
      gitGraph
        commit
        commit
        branch your_own_fork
        checkout your_own_fork
        commit
        commit
        checkout main
        merge your_own_fork
        commit
        commit

4. The fork now stays remotely on GitHub servers, what's next is to "clone" it into
   your computer locally:

   .. code-block:: console

     $ git clone https://github.com/YOUR_GITHUB_ACCOUNT/cofi.git
     $ git remote add upstream https://github.com/inlab-geo/cofi.git
     $ git fetch upstream

   replacing ``YOUR_GITHUB_ACCOUNT`` with your own account.
5. If you are working on documentation, then remember to update the submodule linked to
   `cofi-examples <https://github.com/inlab-geo/cofi-examples>`_:

   .. code-block:: console

      $ cd cofi
      $ git submodule update --init


.. _env_setup:

Environment setup
^^^^^^^^^^^^^^^^^

The environment setup is different depending on your purpose:

- If you are going to work on :ref:`adding new forward examples <new_forward>`, then make 
  sure you have CoFI `installed <installation.html>`_ in the usual way.
- If you are going to work on :ref:`adding/linking new inversion tool <new_inversion>`, 
  or looking to :ref:`add features or fix bugs <cofi_core>` in the library core, then 
  try to prepare your environment to have dependencies listed in this 
  `environment_dev.yml <https://github.com/inlab-geo/cofi/blob/main/envs/environment_dev.yml>`_
  file. It's easy to set this up using ``conda`` under your local clone:

  .. code-block:: console

    $ conda env create -f envs/environment_dev.yml
    $ conda activate cofi_dev
    $ pip install -e .
- If you'd like to :ref:`edit the documentation <doc>`, then get the dependencies listed in this
  `environment.yml <https://github.com/inlab-geo/cofi/blob/main/docs/environment.yml>`_
  file. Similarly, set up this with ``conda``:

  .. code-block:: console

    $ conda env create -f docs/environment.yml
    $ conda activate readthedocs
    $ pip install -e .


Coding / editing
^^^^^^^^^^^^^^^^

Quick reference for working with the codebase:

:To install: ``pip install -e .``
:To test: ``coverage run -m pytest``
:To auto-format: ``black .`` or ``black --check .`` to check without changing

Additionally, we have some guidance on the following scenarios:

- :ref:`adding new forward examples <new_forward>`
- :ref:`adding/linking new inversion tool <new_inversion>`
- :ref:`add features or fix bugs <cofi_core>`
- :ref:`edit the documentation <doc>`

Again, don't hesitate to ask us whenever you feel confused. Contact us
via `GitHub issues <https://github.com/inlab-geo/cofi/issues/new/choose>`_
or `Slack`_.


.. _commit_push_pr:

Commit, push and pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The git `commit <https://git-scm.com/docs/git-commit>`_ operation captures the staged 
changes of the project.

The git `add <https://git-scm.com/docs/git-add>`_ command is how you add files to 
the so-called "staging" area.

Therefore, a typical pattern of commiting a change is:

.. code-block:: console

  $ git add path1/file1 path2/file2
  $ git commit -m "my commit message"

Please note that we aim to use 
`Angular style <https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines>`_ 
commit messages throughout our projects. Simply speaking, we categorise our commits by
a short prefix (from ``feat``, ``fix``, ``docs``, ``style``, ``refactor``, ``perf``, 
``test`` and ``chore``).

Once your changes are committed, push the commits into your remote fork:

.. code-block:: console
  
  $ git push

Open the remote repository under your GitHub account, you should be able to see the
new commits pushed.

Now that you've finished the coding and editing work, look for the "Contribute" button 
-> "Open pull request", write a description and continue as prompted.

Once your pull request is submitted, we are able to see it and will work our best to 
review and provide feedback as soon as we can. Thanks for all the efforts along the way
of contributing! 🎉🎉🎉


Coding in CoFI
--------------

.. _new_forward:

New forward example
^^^^^^^^^^^^^^^^^^^

CoFI doesn't have any forward solvers in the package itself. Instead, we manage
all of our forward code as a part of the example gallery maintained in the
`cofi-examples <https://github.com/inlab-geo/cofi-examples>`_ respository.

Follow the instructions
`here <https://github.com/inlab-geo/cofi-examples#contribution>`_ for details on
how to contribute to the example repository.

Our `tutorials <tutorials/generated/index.html>`_ page is a good place to start learning about how to
plug in an inversion problem in ``cofi``. Furthermore, there are examples with increasing 
complexity presented in the `example gallery <examples/generated/index.html>`_ 
page for you to learn from.


.. _new_inversion:

New inversion tool
^^^^^^^^^^^^^^^^^^

Thank you for your attempt in enriching ``cofi``'s library pool.

To get started, run the helper script:

.. code-block:: console

  $ python scripts/new_inference_tool.py <new_tool_name>

To define and plug in your own inference tool backend, you minimally have to create a
subclass of :class:`tools.BaseInferenceTool` and implement two methods: 
``__init__`` and ``__call__``. Additionally, add the name and class reference to our
inference tools tree under ``src/cofi/tools/__init__.py`` so that our dispatch routine can
find the class from the name specified in an :class:`InversionOptions` instance.

Documentation 
`API reference - BaseInferenceTool <api/generated/cofi.tools.BaseInferenceTool.html>`_ provides
further details and examples.

Follow the :ref:`environment setup section <env_setup>` to set up the package
and :ref:`commit, push and pull request section <commit_push_pr>` to raise a pull 
request.

We would also appreciate it if you write tests that ensure a good coverage under the
file path ``tests``.

.. admonition:: Checklist
  :class: tip, dropdown

  1. Have you added a new file with a proper name under ``src/cofi/tools/``?
  2. Have you declared the tool class as a subclass of :class:`tools.BaseInferenceTool`?
  3. Have you implemented ``__init__`` and ``__call__`` methods minimally? 
  4. If you'd like us to do input validation, have you defined class variables
     ``required_in_problems``, ``optional_in_problem``, ``required_in_options`` and
     ``optional_in_options``?
  5. If you'd like us to display the tool related information properly, have you 
     defined class variables ``short_description`` and ``documentation_links``?
  6. Have you imported and added the tool subclass name to ``src/cofi/tools/__init__.py``?
  7. Have you added tool name and class reference to the ``inference_tools_table`` in file
     ``src/cofi/tools/__init__.py``?
  8. Have you written tests for your new inference tool under ``tests/cofi_tools``?


.. _cofi_core:

Feature or bug fixes in ``cofi`` core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we provide a mapping table to the parts of code related to each existing feature.

.. list-table:: Table: feature mapping to code file
   :widths: 60 40
   :header-rows: 1

   * - Functionality
     - Code file path
   * - :class:`BaseProblem`
     - src/cofi/base_problem.py
   * - :class:`InversionOptions`
     - src/cofi/inversion_options.py
   * - :class:`Inversion`
     - src/cofi/inversion.py
   * - :class:`InversionResult`
     - src/cofi/inversion.py
   * - inference tools tree
     - src/cofi/tools/__init__.py
   * - inference tool dispatch function
     - src/cofi/inversion.py
   * - :class:`BaseInferenceTool`
     - src/cofi/tools/base_inference_tool.py
   * - validation for :class:`BaseProblem` and :class:`InversionOptions` objects
     - src/cofi/tools/base_inference_tool.py

.. src/cofi
.. ├── __init__.py
.. ├── _version.py
.. ├── base_problem.py
.. ├── inversion.py
.. ├── inversion_options.py
.. └── tools
..     ├── __init__.py
..     ├── base_inference_tool.py
..     ├── cofi_simple_newton.py
..     ├── emcee.py
..     └── pytorch_optim.py
..     └── scipy_lstsq.py
..     └── scipy_opt_lstsq.py
..     └── scipy_opt_min.py

.. admonition:: Checklist on adding a new set method in ``BaseProblem``
  :class: tip, dropdown

  Except for tests, all changes should take place in ``src/cofi/base_problem.py``.

  1. add method ``set_something(self, something)``
  2. add property/method ``something(self)``
  3. add method ``something_defined(self) -> bool``
  4. add ``something`` to list ``BaseProblem.all_components``
  5. write tests in ``tests/test_base_problem.py`` ("test_non_set", etc.)


.. _doc:

Documentation
^^^^^^^^^^^^^

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

.. ├── README.html
.. ├── api
.. │   ├── generated
.. │   │   ├── cofi.BaseProblem.html
.. │   │   ├── cofi.Inversion.html
.. │   │   ├── cofi.InversionOptions.html
.. │   │   ├── cofi.InversionResult.html
.. │   │   └── cofi.tools.BaseInferenceTool.html
.. │   └── index.html
.. ├── changelog.html
.. ├── cofi-examples
.. │   ├── README.html
.. │   ├── generated
.. │   │   ├── gravity_density.html
.. │   │   ├── index.html
.. │   │   ├── linear_regression.html
.. │   │   └── sg_execution_times.html
.. │   ├── index.html
.. │   ├── examples
.. │   │   ├── gravity_density.html
.. │   │   ├── gravity_density_lab.html
.. │   │   ├── linear_regression.html
.. │   │   └── linear_regression_lab.html
.. │   └── scripts
.. │       └── README.html
.. ├── contribute.html
.. ├── faq.html
.. ├── genindex.html
.. ├── index.html
.. ├── installation.html
.. ├── objects.inv
.. ├── py-modindex.html
.. ├── reports
.. │   ├── gravity_density_lab.log
.. │   └── linear_regression_lab.log
.. ├── search.html
.. ├── searchindex.js
.. └── tutorial.html

