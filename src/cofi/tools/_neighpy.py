from numpy import exp, argmin

from . import BaseInferenceTool, error_handler


class Neighpy(BaseInferenceTool):
    r"""Wrapper for the tool <FILL IN HERE>

    FIXME Any extra information about the tool
    """

    documentation_links = []  # FIXME required
    short_description = []  # FIXME required

    @classmethod
    def required_in_problem(cls) -> set:
        return set("objective")

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {}

    @classmethod
    def required_in_options(cls) -> set:
        return set(
            "bounds",
            "direct_search_ns",
            "direct_search_nr",
            "direct_search_ni",
            "direct_search_n",
            "appraisal_n_resample",
            "appraisal_n_walkers",
        )

    @classmethod
    def optional_in_options(cls) -> dict:
        return {}

    def __init__(self, inv_problem, inv_options):  # FIXME implementation required
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())

    def __call__(self) -> dict:  # FIXME implementation required

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
    def _call_backend_tool(self):  # FIXME implementation required
        searcher = self._initialise_searcher()
        direct_search_samples, direct_search_objectives = self._call_searcher(searcher)

        appraiser = self._initalise_appraiser(
            direct_search_samples, direct_search_objectives
        )
        appraisal_samples = appraiser.run()

        return {
            "direct_search_samples": direct_search_samples,
            "direct_search_objectives": direct_search_objectives,
            "appraisal_samples": appraisal_samples,
        }

    @error_handler(
        when="in calling neighpy.search.NASearcher instance",
        context="for the direct search phase",
    )
    def _call_searcher(searcher):
        searcher.run()
        return searcher.samples, searcher.objectives

    @error_handler(
        when="in calling neighpy.appraise.NAAppraiser instance",
        context="for the appraisal phase",
    )
    def _call_appriaser(appraiser):
        appraiser.run()
        return appraiser.samples

    @error_handler(
        when="in creating neighpy.search.NASearcher object",
        context="at initialisation",
    )
    def _initialise_searcher(self):
        from neighpy import NASearcher

        return NASearcher(
            self.inv_problem.objective,
            ns=self._params["direct_search_ns"],
            nr=self._params["direct_search_nr"],
            ni=self._params["direct_search_ni"],
            n=self._params["direct_search_n"],
            bounds=self._params["bounds"],
        )

    @error_handler(
        when="in creating neighpy.appraise.NAAppraiser object",
        context="at initialisation, after running NASearcher",
    )
    def _initalise_appraiser(self, samples, objectives):
        from neighpy import NAAppraiser

        self._appraiser = NAAppraiser(
            samples,
            exp(-objectives),
            bounds=self._params["bounds"],
            n_resample=self._params["appraisal_n_resample"],
            n_walkers=self._params["appraisal_n_walkers"],
        )
