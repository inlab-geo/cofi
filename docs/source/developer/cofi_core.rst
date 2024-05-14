Editing CoFI core
^^^^^^^^^^^^^^^^^

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

.. admonition:: Checklist on adding a new set method in ``BaseProblem``
  :class: tip, dropdown

  Except for tests, all changes should take place in ``src/cofi/base_problem.py``.

  1. add method ``set_something(self, something)``
  2. add property/method ``something(self)``
  3. add method ``something_defined(self) -> bool``
  4. add ``something`` to list ``BaseProblem.all_components``
  5. write tests in ``tests/test_base_problem.py`` ("test_non_set", etc.)
