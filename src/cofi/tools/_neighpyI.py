from numpy import argmin
from numpy.typing import NDArray
from typing import Any

from . import BaseInferenceTool, error_handler


class NeighpyI(BaseInferenceTool):
    r"""Wrapper for the tool :code:`neighpy`, a Python implementation of the Neighbourhood Algorithm.

    The Neighbourhood Algorithm is a direct search method that uses a set of points in the parameter space to explore the objective function. It is a derivative-free method that is particularly useful for high-dimensional problems.

    This wrapper only implements the 1st phase of the NA i.e. the direct search phase.
    """

    documentation_links = ["https://neighpy.readthedocs.io/en/latest/index.html"]
    short_description = [
        "Wrapper for Neighpy, a Python implementation of the Neighbourhood Algorithm"
    ]

    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {}

    @classmethod
    def required_in_options(cls) -> set:
        return {
            "bounds",
            "n_samples_per_iteration",
            "n_cells_to_resample",
            "n_initial_samples",
            "n_iterations",
        }

    @classmethod
    def optional_in_options(cls) -> dict:
        return {"serial": False}

    def __init__(self, inv_problem, inv_options) -> None:
        super().__init__(inv_problem, inv_options)
        self._params["ndim"] = len(self._params["bounds"])
        self._components_used = list(self.required_in_problem())

    def __call__(self) -> dict:
        raw_results = self._call_backend_tool()
        _best = raw_results["samples"][argmin(raw_results["objectives"])]
        res = {"success": True, "model": _best}
        return {**res, **raw_results}

    @error_handler(
        when="at some point in the Neighbourhood Algorithm Direct Search phase",
        context="",
    )
    def _call_backend_tool(self) -> dict:
        searcher = self._initialise_searcher()
        samples, objectives = self._call_searcher(
            searcher, parallel=not self._params["serial"]
        )
        return {
            "samples": samples,
            "objectives": objectives,
        }

    @staticmethod
    @error_handler(
        when="in calling neighpy.search.NASearcher instance",
        context="for the direct search phase",
    )
    def _call_searcher(searcher, parallel=True):
        searcher.run(parallel=parallel)
        return searcher.samples, searcher.objectives

    @error_handler(
        when="in creating neighpy.search.NASearcher object",
        context="at initialisation",
    )
    def _initialise_searcher(self):
        from neighpy import NASearcher

        return NASearcher(
            self.inv_problem.objective,
            ns=self._params["n_samples_per_iteration"],
            nr=self._params["n_cells_to_resample"],
            ni=self._params["n_initial_samples"],
            n=self._params["n_iterations"],
            bounds=self._params["bounds"],
        )
