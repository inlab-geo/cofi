New inversion tool
^^^^^^^^^^^^^^^^^^

Thank you for your attempt in enriching ``cofi``'s library pool.

Below is a checklist for quick reference throughout the process of adding a new 
inversion tool. Read on the rest of this page for detailed instructions.

.. admonition:: Checklist
   :class: tip

   #. Run the helper script ``tools/new_inference_tool.py`` to generate a new tool file.
   #. Implement ``__init__`` and ``__call__`` methods at least.
   #. Define class variables ``required_in_problems``, ``optional_in_problem``,
      ``required_in_options`` and ``optional_in_options`` for input validation.
   #. Define class variables ``short_description`` and ``documentation_links`` for
      displaying tool related information.
   #. Import and add the tool subclass name to ``src/cofi/tools/__init__.py``.
   #. Add tool name and class reference to the ``inference_tools_table`` in file
      ``src/cofi/tools/__init__.py``.
   #. Fill in the last few lines of the tool file so that your new tool is registered
      in the inference tools tree.
   #. Write tests for your new inference tool under ``tests/cofi_tools``.
   #. Prepare a relevant example under CoFI examples and raise a pull request in the
      ``cofi-examples`` repository.


1. Prerequisites
----------------

Follow the :ref:`environment setup section <env_setup>` to set up the package
and :ref:`commit, push and pull request section <commit_push_pr>` to raise a pull 
request.

2. Generate a new inversion tool file
-------------------------------------

To get started, run the helper script:

.. code-block:: console

  $ python tools/new_inference_tool.py <new_tool_name>

3. Code up the new inversion tool
---------------------------------

To define and plug in your own inference tool backend, you minimally have to create a
subclass of :class:`tools.BaseInferenceTool` and implement two methods: 
``__init__`` and ``__call__``. Additionally, add the name and class reference to our
inference tools tree under ``src/cofi/tools/__init__.py`` so that our dispatch routine can
find the class from the name specified in an :class:`InversionOptions` instance.

Documentation 
`API reference - BaseInferenceTool <../api/generated/cofi.tools.BaseInferenceTool.html>`_ provides
further details and examples.

4. Register the new tool under CoFI tree
----------------------------------------

In the tool file, fill in the last few lines so that your new tool is registered in the
inference tools tree. The following is an example of how to register a new tool:

.. code-block:: python

   # CoFI -> Ensemble methods -> Bayesian sampling -> Trans-D McMC -> bayesbay -> VanillaSampler
   # description: Sampling the posterior by means of reversible-jump Markov chain Monte Carlo.
   # documentation: https://bayes-bay.readthedocs.io/en/latest/api/generated/bayesbay.samplers.VanillaSampler.html

Feel free to browse the existing tools in the ``cofi`` library for reference.
`InLab Explorer <https://inlab.au/inlab-explorer/>`_ provides a visual representation of
eligible branches in the inference tools tree.

4. Write tests
--------------

We need you to write tests that ensure a good coverage under the file path ``tests``.
Place the test file under ``tests/cofi_tools`` and name it as ``test_<new_tool_name>.py``.
You can refer to the existing test files for guidance, and copy the test template from
``tests/cofi_tools/_template.py`` as a starting point.

5. Add a relevant example
-------------------------

Once the above has been done, please add a relevant example under CoFI examples and raise a
pull request in the ``cofi-examples`` repository. You may refer to the
`Contributor Guide for CoFI Examples <https://github.com/inlab-geo/cofi-examples/blob/main/CONTRIBUTING.md#to-add-a-domain-specific-eg-geoscience-example>`_
to get started.
