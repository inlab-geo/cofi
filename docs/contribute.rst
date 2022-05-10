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
or `Slack <https://inlab-geo.slack.com>`_.

General Workflow
----------------

Here is a general flowchart for code contribution, including preparation, editting and
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

Fork and clone respository
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Navigate to the GitHub repository (`cofi <https://github.com/inlab-geo/cofi>`_,
   or `cofi-examples <https://github.com/inlab-geo/cofi-examples>`_ if you'd like to
   add or edit things in the `example gallery <cofi-examples/generated/index.html>`_)
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
   your computer locally::

     git clone https://github.com/YOUR_GITHUB_ACCOUNT/cofi.git

   replacing ``YOUR_GITHUB_ACCOUNT`` with your own account.


Environment setup
^^^^^^^^^^^^^^^^^

The environment setup is different depending on your purpose:

- If you are going to work on :ref:`adding new forward examples <new_forward>`, then make 
  sure you have CoFI `installed <installation.html>`_ in the usual way.
- If you are going to work on :ref:`adding/linking new inversion solver <new_inversion>`, 
  or looking to :ref:`add features or fix bugs <cofi_core>` in the library core, then 
  try to prepare your environment to have dependencies listed in this 
  `environment_dev.yml <https://github.com/inlab-geo/cofi/blob/main/envs/environment_dev.yml>`_
  file. It's easy to set this up using ``conda`` under your local clone::
    conda env create -f envs/environment_dev.yml
- If you'd like to :ref:`edit the documentation <doc>`, then get the dependencies listed in this
  `environment.yml <https://github.com/inlab-geo/cofi/blob/main/docs/environment.yml>`_
  file. Similarly, set up this with ``conda``::
    conda env create -f docs/environment.yml


Coding / editting
^^^^^^^^^^^^^^^^^

We have some guidance on the following scenarios:

- :ref:`adding new forward examples <new_forward>`
- :ref:`adding/linking new inversion solver <new_inversion>`
- :ref:`add features or fix bugs <cofi_core>`
- :ref:`edit the documentation <doc>`

Again, don't hesitate to ask us whenever you feel confused. Contact us
via `GitHub issues <https://github.com/inlab-geo/cofi/issues/new/choose>`_
or `Slack <https://inlab-geo.slack.com>`_.


Commit, push and pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The git `commit <https://git-scm.com/docs/git-commit>`_ operation captures the staged 
changes of the project.

The git `add <https://git-scm.com/docs/git-add>`_ command is how you add files to 
the so-called "staging" area.

Therefore, a typical pattern of commiting a change is::

  git add path1/file1 path2/file2
  git commit -m "my commit message"

Please note that we aim to use 
`Angular style <https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines>`_ 
commit messages throughout our projects. Simply speaking, we categorise our commits by
a short prefix (from ``feat``, ``fix``, ``docs``, ``style``, ``refactor``, ``perf``, 
``test`` and ``chore``).

Once your changes are committed, push the commits into your remote fork::
  
  git push

Open the remote repository under your GitHub account, you should be able to see the
new commits pushed.

Now that you've finished the coding and editting work, look for the "Contribute" button 
-> "Open pull request", write a description and continue as prompted.

Once your pull request is submitted, we are able to see it and will work our best to 
review and provide feedback as soon as we can. Thanks for all the efforts along the way
of contributing!


Coding in CoFI
--------------

.. _new_forward:

New forward example
"""""""""""""""""""

CoFI doesn't have any forward solvers in the package itself. Instead, we manage
all of our forward code as a part of the example gallery maintained in the
`cofi-examples <https://github.com/inlab-geo/cofi-examples>`_ respository.

Follow the instructions
`here <https://github.com/inlab-geo/cofi-examples#contribution>`_ for details on
how to contribute to the example repository.

Our `tutorials <tutorial.html>`_ page is a good place to start learning about how to
plug in an inversion problem in ``cofi``. Furthermore, there are examples with increasing 
complexity presented in the `example gallery <cofi-examples/generated/index.html>`_ 
page for you to learn from.


.. _new_inversion:

New inversion solver
^^^^^^^^^^^^^^^^^^^^

If you'd like to link a forward problem defined with our API to your own solver,
please follow the instructions in our `tutorials <tutorial.html>`_ and 
`API reference - BaseSolver <api/generated/cofi.solvers.BaseSolver.html>`_.

If you'd like to further share your inversion code and link that to `cofi`, first of
all we'd like to thenk you for your attempt in enriching ``cofi``'s library pool. 
Please follow along this section to set up the package and raise a pull request.


.. _cofi_core:

Feature or bug fixes in ``cofi`` core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _doc:

Documentation
^^^^^^^^^^^^^



.. Patches may include but not limited to:

.. * Adding new **forward example** to our example gallery, following instructions 
..   `here <https://github.com/inlab-geo/cofi-examples#contribution>`_
.. * Adding new **inversion solver**, following instructions in `our tutorial - Advanced Usage <tutorial.html#advanced-usage>`_
.. * Fixing bugs
.. * Improving documentation
.. * \...


