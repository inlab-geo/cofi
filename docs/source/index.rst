.. CoFI documentation master file, created by
   sphinx-quickstart on Wed Nov 24 12:24:06 2021.

.. title:: Home

================
Welcome to CoFI!
================

.. The following defines the structure of the document. It is hidden so it doesn't
   render into the html, but it cannot be removed!
   We assume that each 'index.rst' document defines its own toctree that can be incorporated.

.. toctree::
    :caption: User guides
    :hidden: 
    :maxdepth: 1

    introduction.rst
    installation.rst
    tutorials/generated/index.rst
    examples/generated/index.rst
    gallery/generated/index.rst

.. toctree::
    :caption: User reference
    :hidden: 
    :maxdepth: 1

    api/index.rst
    faq.rst
    changelog.md
    licence.rst

.. toctree::
    :caption: Developer guides
    :hidden: 
    :maxdepth: 1

    contribute.rst
    developer/new_example
    developer/new_tool
    developer/cofi_core
    developer/docs


Welcome to the CoFI Documentation! CoFI, the Common Framework for Inference, is an 
open-source tool that bridges the gap between domain expertise and inference, within 
and beyond the field of geoscience.

Whether you're a student, an educator, or an industry professional, CoFI is your go-to 
tool. It simplifies the process of applying inference techniques, allowing for a wide 
range of applications from academic research and teaching to solving real-world 
geoscience problems in the industry. With its user-friendly, flexible and accessible
interface, CoFI empowers you to focus on what truly matters - the science.

.. mermaid:: _static/cofi_tree.mmd


.. code-block:: python
    :linenos:
    :class: toggle

    # CoFI API has flexible ways of defining an inversion problem. For instance:
    import cofi

    inv_problem = cofi.BaseProblem()
    inv_problem.set_objective(my_objective_func)
    inv_problem.set_initial_model(my_starting_point)

    # Once a problem is defined, `cofi` can tell you what inference tools you can 
    # use based on what level of information you've provided:
    inv_problem.suggest_tools()   # a tree will be printed

    # Run an inversion with these lines:
    inv_options = cofi.InversionOptions()
    inv_options.set_tool("torch.optim")
    inv_options.set_params(options={"num_iterations": 50, "algorithm": "Adam"})

    inv = cofi.Inversion(inv_problem, inv_options)
    result = inv.run()
    print(result.success)
    print(result.model)


If this looks useful, let's get started!

.. grid:: 1 3 3 3
    :gutter: 3
    :padding: 2

    .. grid-item-card::
        :link: installation.html
        :text-align: center
        :class-card: card-border

        *☕️ Installation*
        ^^^^^^^^^^^^^^^^^
        Get a CoFI from here
    
    .. grid-item-card::
        :link: tutorials/generated/index.html
        :text-align: center
        :class-card: card-border

        *📖 Tutorials*
        ^^^^^^^^^^^^^^
        Step-by-step guide on how to use CoFI

    .. grid-item-card::
        :link: examples/generated/index.html
        :text-align: center
        :class-card: card-border

        *🗂️ Examples* 
        ^^^^^^^^^^^^^
        Adapt a real-world application to your needs

    .. grid-item-card:: 
        :link: api/index.html
        :text-align: center
        :class-card: card-border

        *📗 API Reference*
        ^^^^^^^^^^^^^^^^^^
        Dive into the details

    .. grid-item-card::
        :link: contribute.html
        :text-align: center
        :class-card: card-border

        *🏗️ Development*
        ^^^^^^^^^^^^^^^^
        Report issues or contribute with your code
    
    .. grid-item-card::
        :link: https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg
        :text-align: center
        :class-card: card-border

        *💬 Have questions?*
        ^^^^^^^^^^^^^^^^^^^^
        Accept this invitation to join the conversation on Slack




CoFI is currently supported and coordinated by `InLab <https://inlab.au>`_.
