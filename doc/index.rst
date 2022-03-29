.. CoFI documentation master file, created by
   sphinx-quickstart on Wed Nov 24 12:24:06 2021.

.. title:: Home

================================
Welcome to CoFI's documentation!
================================

.. What CoFI is

CoFI (**Co**\ mmon **F**\ ramework for **I**\ nference) is an open-source 
initiative for interfacing between generic inference algorithms and specific 
geoscience problems.

With a mission to bridge the gap between the domain expertise and the 
inference expertise, this Python package provides an interface across a 
wide range of inference algorithms from different sources, as well as ways 
of defining inverse problems with examples included.

.. A small code example

Applying inversion techniques on a defined problem yeilds useful Python
object in return:

.. container:: toggle, toggle-hidden

    .. doctest:: basic

        >>> from cofi.cofi_objective import XRayTomographyObjective
        >>> xrt_problem = XRayTomographyObjective("data.csv")
        >>> from cofi.optimisers import LeastSquareSolver
        >>> solver = LeastSquareSolver(xrt_problem)
        >>> result = solver.solve(tool="numpy.linalg.lstsq")
        >>> result.ok
        True
        >>> result.solver_tool
        'numpy.linalg.lstsq'
        >>> len(result.model)
        2500


.. attention::

    This package is still under initial development stage, so public APIs are 
    not expected to be stable. Please stay updated and don't hesitate to raise
    feedback or issues through `GitHub issues <https://github.com/inlab-geo/cofi/issues/new/choose>`_ 
    or `Slack workspace <https://inlab-geo.slack.com>`_.


.. seealso::

    This site includes basic usage, tutorials & API documentation of CoFI (the
    Python package). For more information on **InLab**, which is what this 
    project is led by, 
    please check out `the InLab website <http://www.inlab.edu.au/>`_.


.. panels::
    :header: text-center text-large
    :card: border-1 m-1 text-center

    **Getting started**
    ^^^^^^^^^^^^^^^^^^^

    New to CoFI?

    .. link-button:: getting-started
        :type: ref
        :text: Start here
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Need support?**
    ^^^^^^^^^^^^^^

    Ask in our Slack workspace

    .. link-button:: https://inlab-geo.slack.com
        :type: url
        :text: Join the conversation
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Reference documentation**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A list of our functions and classes

    .. link-button:: api
        :type: ref
        :text: API reference
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Contribute to CoFI**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Features or bug fixes are always welcomed!

    .. link-button:: contribute
        :type: ref
        :text: Developer notes
        :classes: btn-outline-primary btn-block stretched-link


Table of contents
-----------------

.. toctree::
    :caption: Basic usage
    :maxdepth: 1

    installation.rst
    notebooks/index.rst
    getting-started.rst
    faq.rst

.. toctree::
    :caption: Reference
    :maxdepth: 1

    api/index.rst

.. toctree::
    :caption: Developer notes
    :maxdepth: 1

    contribute.rst

.. Getting started
.. ---------------

.. Typical API calls and some examples are explained in the getting-started page:

.. .. toctree::
..     :maxdepth: 2

..     getting-started



.. API
.. ---

.. Want to find out how to use a problem type or inverse solver? Check our auto-generated
.. API documentation:

.. .. toctree::
..     :maxdepth: 1
..     :glob:

..     api/*


.. Contribute to CoFI
.. ------------------

.. Found bugs? Wanna improve something? See the contribute page:

.. .. toctree::
..     :maxdepth: 1
    
..     contribute


.. FAQ
.. ---

.. Got a question?

.. .. toctree::
..     :maxdepth: 1

..     faq
