New inversion tool
^^^^^^^^^^^^^^^^^^

Thank you for your attempt in enriching ``cofi``'s library pool.

To get started, run the helper script:

.. code-block:: console

  $ python tools/new_inference_tool.py <new_tool_name>

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
