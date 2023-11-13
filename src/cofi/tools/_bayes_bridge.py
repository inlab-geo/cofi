import numpy as np
import bayesbridge as bb

from . import BaseInferenceTool, error_handler


class BayesBridge(BaseInferenceTool):
    r"""Wrapper for the tool Bayes Bridge, a trans-dimensional reversible jump Bayesian
    sampling framework

    FIXME Any extra information about the tool
    """
    documentation_links = []        # FIXME required
    short_description = []          # FIXME required

    @classmethod
    def required_in_problem(cls) -> set:        # FIXME implementation required
        return {"log_posterior"}
    
    @classmethod
    def optional_in_problem(cls) -> dict:       # FIXME implementation required
        return dict()

    @classmethod
    def required_in_options(cls) -> set:        # FIXME implementation required
        return {"perturbations"}

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


# CoFI -> Ensemble methods -> Bayesian sampling -> Trans-D McMC -> bayesbridge -> Reversible Jump Bayesian Sampling
# description: Reversible Jump Bayesian Inference with trans-dimensional and hierarchical features.
# documentation: TBD
