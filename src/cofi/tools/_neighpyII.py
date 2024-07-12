from . import BaseInferenceTool, error_handler


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
        return set()

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
