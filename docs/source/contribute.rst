*****************
How to Contribute
*****************

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

Here is a general flowchart for code contribution, including preparation, editing and
submitting stages:

.. mermaid:: _static/github_workflow.mmd

.. _fork_clone:

1. Fork and clone respository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to hide / show
  :open:

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

2. Environment setup
^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to hide / show
  :open:

  The environment setup is different depending on your purpose:

  - If you are going to work on `adding new forward examples <developer/new_example.html>`_, then make 
    sure you have CoFI `installed <installation.html>`_ in the usual way.
  - If you are going to work on `adding/linking new inversion tool <developer/new_tool.html>`_, 
    or looking to `add features or fix bugs <developer/cofi_core.html>`_ in the library core, then 
    try to prepare your environment to have dependencies listed in this 
    `environment_dev.yml <https://github.com/inlab-geo/cofi/blob/main/envs/environment_dev.yml>`_
    file. It's easy to set this up using ``conda`` under your local clone:

    .. code-block:: console

      $ conda env create -f envs/environment_dev.yml
      $ conda activate cofi_dev
      $ pip install -e .
  - If you'd like to `edit the documentation <developer/docs.html>`_, then get the dependencies listed in this
    `requirements.txt <https://github.com/inlab-geo/cofi/blob/main/docs/requirements.txt>`_
    file. Set up this with ``pip``:

    .. code-block:: console

      $ pip install -r docs/requirements.txt  # in a virtual environment
      $ pip install -e .

.. code_editing:

3. Coding / editing
^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to hide / show
  :open:

  Quick reference for working with the codebase:

  :To install: ``pip install -e .``
  :To test: ``coverage run -m pytest``
  :To auto-format: ``black .`` or ``black --check .`` to check without changing

  Additionally, we have some guidance on the following scenarios:

  - :doc:`developer/new_example`
  - :doc:`developer/new_tool`
  - :doc:`developer/cofi_core`
  - :doc:`developer/docs`
  - :doc:`developer/release`

  Again, don't hesitate to ask us whenever you feel confused. Contact us
  via `GitHub issues <https://github.com/inlab-geo/cofi/issues/new/choose>`_
  or `Slack`_.

.. _test_coverage:

4. Testing your code
^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to hide / show
  :open:

  When you submit a pull request, an automatic testing job will be triggered on GitHub.

  If you'd like to test your changes locally, 

  1. Follow :ref:`instructions here to set up environment <env_setup>` if you haven't 
     done so yet.
  2. Run all the tests with
    
     .. code:: console

      $ pytest tests
    
  3. Check the test coverage with

     .. code:: console

      $ coverage -m pytest tests; coverage report; coverage xml

     We require contributors to add your tests to ensure 100% test coverage.


.. _commit_push_pr:

5. Commit, push and pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Click to hide / show
  :open:

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
