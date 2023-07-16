import numpy as np

from . import BaseInferenceTool, error_handler


# Add TemplateTool class into src/cofi/tools/__init__.py
# FIXME 1. "from _template_tool import TemplateTool"
# FIXME 2. add "TemplateTool" to "__all__" list
# FIXME 3. add "TemplateTool" to "inference_tools_table" dictionary
# FIXME Remove above comments lines after completed

class TemplateTool(BaseInferenceTool):
    r"""Wrapper for the tool <FILL IN HERE>

    FIXME Any extra information about the tool
    """
    documentation_links = []        # FIXME required
    short_description = []          # FIXME required

    @classmethod
    def required_in_problem(cls) -> set:        # FIXME implementation required
        raise NotImplementedError
    
    @classmethod
    def optional_in_problem(cls) -> dict:       # FIXME implementation required
        raise NotImplementedError

    @classmethod
    def required_in_options(cls) -> set:        # FIXME implementation required
        raise NotImplementedError

    @classmethod
    def optional_in_options(cls) -> dict:       # FIXME implementation required
        raise NotImplementedError

    @classmethod
    def available_algorithms(cls) -> set:       # FIXME optional (delete it if not needed)
        raise NotImplementedError
    
    def __init__(self, inv_problem, inv_options):       # FIXME implementation required
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
    
    def __call__(self) -> dict:                         # FIXME implementation required
        raw_results = self._call_backend_tool()
        res = {
            "success": "TODO",
            "model": "TODO",
            # FIXME add more information if there's more in raw_results
        }
        return res
    
    @error_handler(
        when="FIXME (e.g. when solving / calling ...)",
        context="FIXME (e.g. in the process of solving / preparing)",
    )
    def _call_backend_tool(self):                       # FIXME implementation required
        raise NotImplementedError
