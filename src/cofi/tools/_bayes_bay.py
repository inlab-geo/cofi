import functools

from . import BaseInferenceTool, error_handler


class BayesBay(BaseInferenceTool):
    r"""Wrapper for the tool BayesBay, a trans-dimensional reversible jump Bayesian
    sampling framework
    """

    documentation_links = [
        "https://bayes-bay.readthedocs.io/en/latest/api/generated/bayesbay.BaseBayesianInversion.html#bayesbay.BaseBayesianInversion"
    ]
    short_description = [
        "Wrapper for BayesBay, a trans-dimensional reversible jump Bayesian sampling"
        "framework"
    ]

    @classmethod
    def required_in_problem(cls) -> set:
        return set()

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {"log_likelihood": None}

    @classmethod
    def required_in_options(cls) -> set:
        return {"perturbation_funcs", "walkers_starting_states"}

    @classmethod
    def optional_in_options(cls) -> dict:
        return _inspect_default_options()

    @classmethod
    def available_algorithms(cls) -> set:
        import bayesbay as bb

        return {s for s in bb.samplers.__all__ if s != "Sampler"}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        self._initialize_bb_inversion()

    def __call__(self) -> dict:
        self._call_backend_tool()
        samples = self._bb_bayes_inversion.get_results()
        res = {
            "success": True,
            "models": samples,
            "sampler": self._bb_bayes_inversion,
        }
        return res

    @error_handler(
        when="when initializating bayesbay.BaseBayesianInversion",
        context="in the process of preparing",
    )
    def _initialize_bb_inversion(self):
        _log_like_defined = self.inv_problem.log_likelihood_defined
        assert not (
            not _log_like_defined and self._params["log_like_ratio_func"] is None
        ), "at least one of `log_likelihood` and `log_like_ratio_func` to be defined"

        import bayesbay as bb

        self._bb_bayes_inversion = bb.BaseBayesianInversion(
            walkers_starting_states=self._params["walkers_starting_states"],
            perturbation_funcs=self._params["perturbation_funcs"],
            log_like_func=(
                self.inv_problem.log_likelihood if _log_like_defined else None
            ),
            log_like_ratio_func=self._params["log_like_ratio_func"],
            n_chains=self._params["n_chains"],
            n_cpus=self._params["n_cpus"],
        )

    @error_handler(
        when="when calling bayesbay.BaseBayesianInversion.run method",
        context="in the process of solving",
    )
    def _call_backend_tool(self):
        self._bb_bayes_inversion.run(
            sampler=self._params["sampler"],
            n_iterations=self._params["n_iterations"],
            burnin_iterations=self._params["burnin_iterations"],
            save_every=self._params["save_every"],
            verbose=self._params["verbose"],
            print_every=self._params["print_every"],
        )


@functools.lru_cache(maxsize=None)
def _inspect_default_options():
    import inspect
    from bayesbay import BaseBayesianInversion

    _bb_inv_init_args = dict(inspect.signature(BaseBayesianInversion).parameters)
    optional_in_options: dict = {
        k: v.default
        for k, v in _bb_inv_init_args.items()
        if v.default is not inspect._empty and k != "log_likelihood_func"
    }
    _bb_inv_run_args = dict(inspect.signature(BaseBayesianInversion.run).parameters)
    optional_in_options.update({
        k: v.default
        for k, v in _bb_inv_run_args.items()
        if v.default is not inspect._empty
    })
    return optional_in_options


# CoFI -> Ensemble methods -> Bayesian sampling -> Trans-D McMC -> bayesbay -> VanillaSampler
# description: Sampling the posterior by means of reversible-jump Markov chain Monte Carlo.
# documentation: https://bayes-bay.readthedocs.io/en/latest/api/generated/bayesbay.samplers.VanillaSampler.html

# CoFI -> Ensemble methods -> Bayesian sampling -> Trans-D McMC -> bayesbay -> ParallelTempering
# description: Sampling the posterior by means of reversible-jump Markov chain Monte Carlo accelerated with parallel tempering.
# documentation: https://bayes-bay.readthedocs.io/en/latest/api/generated/bayesbay.samplers.ParallelTempering.html

# CoFI -> Ensemble methods -> Bayesian sampling -> Trans-D McMC -> bayesbay -> SimulatedAnnealing
# description: Sampling the posterior by means of reversible-jump Markov chain Monte Carlo accelerated with simulated annealing.
# documentation: https://bayes-bay.readthedocs.io/en/latest/api/generated/bayesbay.samplers.SimulatedAnnealing.html
