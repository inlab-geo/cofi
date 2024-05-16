# CoFI Core Editing

## Description

Please provide a clear and concise description of the changes you've made to the CoFI core.

## Affected Functionality

Please indicate which functionality is affected by these changes:

- [ ] `BaseProblem` (src/cofi/base_problem.py)
- [ ] `InversionOptions` (src/cofi/inversion_options.py)
- [ ] `Inversion` (src/cofi/inversion.py)
- [ ] `InversionResult` (src/cofi/inversion.py)
- [ ] Inference tools tree (src/cofi/tools/__init__.py)
- [ ] Inference tool dispatch function (src/cofi/inversion.py)
- [ ] `BaseInferenceTool` (src/cofi/tools/base_inference_tool.py)
- [ ] Validation for `BaseProblem` and `InversionOptions` objects (src/cofi/tools/base_inference_tool.py)

## Checklist for Adding a New set Method in `BaseProblem`

If you have added a new set method in `BaseProblem`, please confirm the following:

- [ ] The method `set_something(self, something)` has been added
- [ ] The property/method `something(self)` has been added
- [ ] The method `something_defined(self) -> bool` has been added
- [ ] `something` has been added to list `BaseProblem.all_components`
- [ ] Tests have been written in `tests/test_base_problem.py` ("test_non_set", etc.)

## Additional Information

Please provide any additional information or context about the pull request here.
