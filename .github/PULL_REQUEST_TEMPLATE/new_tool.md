# CoFI New Inversion Tool

## Description

Please provide a clear and concise description of the new inversion tool you've added to CoFI.

## Checklist

Please confirm that you have completed the following steps:

- [ ] Run the helper script `tools/new_inference_tool.py` to generate a new tool file
- [ ] Implement `__init__` and `__call__` methods
- [ ] Define class variables `required_in_problems`, `optional_in_problem`, `required_in_options`, and `optional_in_options` for input validation
- [ ] Define class variables `short_description` and `documentation_links` for displaying tool related information
- [ ] Import and add the tool subclass name to `src/cofi/tools/__init__.py`
- [ ] Add tool name and class reference to the `inference_tools_table` in file `src/cofi/tools/__init__.py`
- [ ] Fill in the last few lines of the tool file so that your new tool is registered in the inference tools tree
- [ ] Write tests for your new inference tool under `tests/cofi_tools`
- [ ] Prepare a relevant example under CoFI examples and raise a pull request in the `cofi-examples` repository
      https://github.com/inlab-geo/cofi-examples

If unsure, refer to the detailed instructions here: https://cofi.readthedocs.io/en/latest/developer/new_tool.html.

## Additional Information

Please provide any additional information or context about the pull request here.
