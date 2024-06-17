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
    def _call_backend_tool(self) -> dict[str, NDArray[Any]]:
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


class NeighpyII(BaseInferenceTool):
    r"""Wrapper for the tool :code:`neighpy`, a Python implementation of the Neighbourhood Algorithm.

    The Neighbourhood Algorithm is a direct search method that uses a set of points in the parameter space to explore the objective function. It is a derivative-free method that is particularly useful for high-dimensional problems.

    This wrapper only runs the second phase of the NA i.e. the appraisal phase.
    This phase is used to refine the samples obtained from the first phase.
    It thus requires an initial set of samples with their corresponding log posteriors to be provided.
    This is initial set can be obtained either from the first phase of the NA or any other sampling method.
    """

    documentation_links = ["https://neighpy.readthedocs.io/en/latest/index.html"]
    short_description = [
        "Wrapper for Neighpy, a Python implementation of the Neighbourhood Algorithm"
    ]

    @classmethod
    def required_in_problem(cls) -> set:
        return {}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {}

    @classmethod
    def required_in_options(cls) -> set:
        return {
            "bounds",
            "initial_ensemble",
            "log_ppd",
            "n_resample",
            "n_walkers",
        }

    @classmethod
    def optional_in_options(cls) -> dict:
        return {}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._params["ndim"] = len(self._params["bounds"])
        if (
            self._params["initial_ensemble"].shape[0]
            != self._params["log_ppd"].shape[0]
        ):
            raise ValueError(
                "The number of samples in initial_ensemble does not match the number of samples in log_ppd"
            )
        if self._params["initial_ensemble"].shape[1] != self._params["ndim"]:
            raise ValueError(
                "The shape of initial_ensemble does not match the number of dimensions"
            )
        self._components_used = list(self.required_in_problem())

    def __call__(self) -> dict:
        raw_results = self._call_backend_tool()
        res = {"success": True}
        return {**res, **raw_results}

    @error_handler(
        when="at some point in the Neighbourhood Algorithm Appraisal phase",
        context="",
    )
    def _call_backend_tool(self):

        appraiser = self._initalise_appraiser()
        appraisal_samples = self._call_appriaser(appraiser)

        return {"new_samples": appraisal_samples}

    @staticmethod
    @error_handler(
        when="in calling neighpy.appraise.NAAppraiser instance",
        context="for the appraisal phase",
    )
    def _call_appriaser(appraiser):
        appraiser.run()
        return appraiser.samples

    @error_handler(
        when="in creating neighpy.appraise.NAAppraiser object",
        context="at initialisation.",
    )
    def _initalise_appraiser(self):
        from neighpy import NAAppraiser

        return NAAppraiser(
            initial_ensemble=self._params["initial_ensemble"],
            log_ppd=self._params["log_ppd"],
            n_resample=self._params["n_resample"],
            n_walkers=self._params["n_walkers"],
            bounds=self._params["bounds"],
        )


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
