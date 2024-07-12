from numpy import argmin


from . import BaseInferenceTool, error_handler
from ._neighpyI import NeighpyI
from ._neighpyII import NeighpyII


class Neighpy(BaseInferenceTool):
    r"""Wrapper for the tool :code:`neighpy`, a Python implementation of the Neighbourhood Algorithm.

    The Neighbourhood Algorithm is a direct search method that uses a set of points in the parameter space to explore the objective function. It is a derivative-free method that is particularly useful for high-dimensional problems.

    Split into two phases, the direct search phase and the appraisal phase, this wrapper runs both phases.
    """

    documentation_links = ["https://neighpy.readthedocs.io/en/latest/index.html"]
    short_description = [
        "Wrapper for Neighpy, a Python implementation of the Neighbourhood Algorithm"
    ]

    @classmethod
    def required_in_problem(cls) -> set:
        # nothing required in problem from NeighpyII
        return NeighpyI.required_in_problem()

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {**NeighpyI.optional_in_problem(), **NeighpyII.optional_in_problem()}

    @classmethod
    def required_in_options(cls) -> set:
        # only need n_resample and n_walkers from NeighpyII
        # as we will use the output of the direct search phase
        # as the initial ensemble
        return NeighpyI.required_in_options() | {"n_resample", "n_walkers"}

    @classmethod
    def optional_in_options(cls) -> dict:
        return {**NeighpyI.optional_in_options(), **NeighpyII.optional_in_options()}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._params["ndim"] = len(self._params["bounds"])
        self._components_used = list(self.required_in_problem())

    def __call__(self) -> dict:
        raw_results = self._call_backend_tool()
        _best = raw_results["direct_search_samples"][
            argmin(raw_results["direct_search_objectives"])
        ]
        res = {"success": True, "model": _best}
        return {**res, **raw_results}

    @error_handler(
        when="at some point in the Neighbourhood Algorithm",
        context="",
    )
    def _call_backend_tool(self):
        searcher = NeighpyI._initialise_searcher(self)
        direct_search_samples, direct_search_objectives = NeighpyI._call_searcher(
            searcher, parallel=not self._params["serial"]
        )

        # a bit hacky but this is how we pass the results to the appraiser
        self._params["initial_ensemble"] = direct_search_samples
        self._params["log_ppd"] = -direct_search_objectives

        appraiser = NeighpyII._initalise_appraiser(self)
        appraisal_samples = NeighpyII._call_appriaser(appraiser)

        result = {
            "direct_search_samples": direct_search_samples,
            "direct_search_objectives": direct_search_objectives,
            "appraisal_samples": appraisal_samples,
        }

        return result


# CoFI -> Ensemble methods -> Direct search -> Monte Carlo -> Neighpy -> Neighbourhood Algorithm
# description: Wrapper for the tool Neighpy, implementing the Neighbourhood Algorithm
# documentation: https://neighpy.readthedocs.io/en/latest/
