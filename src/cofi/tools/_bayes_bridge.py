import functools

from . import BaseInferenceTool, error_handler


class BayesBridge(BaseInferenceTool):
    r"""Wrapper for the tool Bayes Bridge, a trans-dimensional reversible jump Bayesian
    sampling framework

    FIXME Any extra information about the tool
    """
    documentation_links = []  # FIXME required
    short_description = []  # FIXME required

    @classmethod
    def required_in_problem(cls) -> set:
        return {"log_posterior"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return dict()

    @classmethod
    def required_in_options(cls) -> set:
        return {"perturbation_funcs", "walkers_starting_models"}

    @classmethod
    def optional_in_options(cls) -> dict: 
        return _inspect_default_options()

    @classmethod
    def available_algorithms(cls) -> set: 
        import bayesbridge as bb
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
        when="when initializating bayesbridge.BaseBayesianInversion", 
        context="in the process of preparing", 
    )
    def _initialize_bb_inversion(self):
        import bayesbridge as bb
        self._bb_bayes_inversion = bb.BaseBayesianInversion(
            walkers_starting_models=self._params["walkers_starting_models"], 
            perturbation_funcs=self._params["perturbation_funcs"], 
            log_posterior_func=self.inv_problem.log_posterior,
            n_chains=self._params["n_chains"], 
            n_cpus=self._params["n_cpus"], 
        )

    @error_handler(
        when="when calling bayesbridge.BaseBayesianInversion.run method",
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
    from bayesbridge import BaseBayesianInversion
    
    _bb_inv_init_args = dict(inspect.signature(BaseBayesianInversion).parameters)
    optional_in_options: dict = {
        k: v.default for k, v in _bb_inv_init_args.items() \
            if v.default is not inspect._empty
    }
    _bb_inv_run_args = dict(inspect.signature(BaseBayesianInversion.run).parameters)
    optional_in_options.update({
        k: v.default for k, v in _bb_inv_run_args.items() \
            if v.default is not inspect._empty
    })
    return optional_in_options



# CoFI -> Ensemble methods -> Bayesian sampling -> Trans-D McMC -> bayesbridge -> Reversible Jump Bayesian Sampling
# description: Reversible Jump Bayesian Inference with trans-dimensional and hierarchical features.
# documentation: TBD
