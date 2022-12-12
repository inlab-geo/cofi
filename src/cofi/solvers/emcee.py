import functools
import numpy as np

from . import BaseSolver, error_handler


class EmceeSolver(BaseSolver):
    documentation_links = [
        "https://emcee.readthedocs.io/en/stable/user/sampler/#emcee.EnsembleSampler",
        "https://emcee.readthedocs.io/en/stable/user/sampler/#emcee.EnsembleSampler.sample",
    ]
    short_description = (
        "an MIT licensed pure-Python implementation of Goodman & Weare’s Affine "
        "Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler"
    )

    @classmethod
    def required_in_problem(cls) -> set:
        return _init_class_methods()[0]

    @classmethod
    def optional_in_problem(cls) -> dict:
        return _init_class_methods()[1]

    @classmethod
    def required_in_options(cls) -> set:
        return _init_class_methods()[2]

    @classmethod
    def optional_in_options(cls) -> dict:
        return _init_class_methods()[3]

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._assign_args()
        self._initialize_sampler()

    def __call__(self) -> dict:
        self._run_mcmc()
        result = {
            "success": True,
            "sampler": self.sampler,
            "blob_names": self._params["blob_names"],
        }
        return result

    def _assign_args(self):
        # assign components in problem to args
        inv_problem = self.inv_problem
        self._params["blobs_dtype"] = None
        self._params["blob_names"] = None
        if inv_problem.log_posterior_with_blobs_defined:
            self._params["log_prob_fn"] = inv_problem.log_posterior_with_blobs
            if inv_problem.blobs_dtype_defined:
                self._params["blob_names"] = [
                    name for (name, _) in inv_problem.blobs_dtype
                ]
                # uncomment below once this has been fixed:
                # issue: https://github.com/arviz-devs/arviz/issues/2036
                # self._blobs_dtype = inv_problem._blobs_dtype
        else:
            self._params["log_prob_fn"] = inv_problem.log_posterior
        self._params["ndim"] = np.shape(self._params["initial_state"])[0]
        self._params["ndim"] = np.prod(inv_problem.model_shape)

    @error_handler(
        when="in creating emcee.EnsembleSampler object",
        context="before sampling",
    )
    def _initialize_sampler(self):
        from emcee import EnsembleSampler

        self.sampler = EnsembleSampler(
            nwalkers=self._params["nwalkers"],
            ndim=self._params["ndim"],
            log_prob_fn=self._params["log_prob_fn"],
            pool=self._params["pool"],
            moves=self._params["moves"],
            args=None,  # already handled by BaseProblem
            kwargs=None,  # already handled by BaseProblem
            backend=self._params["backend"],
            vectorize=self._params["vectorize"],
            blobs_dtype=self._params["blobs_dtype"],
            parameter_names=self._params["parameter_names"],
            a=self._params["a"],
            postargs=self._params["postargs"],
            threads=self._params["threads"],
            live_dangerously=self._params["live_dangerously"],
            runtime_sortingfn=self._params["runtime_sortingfn"],
        )

    @error_handler(
        when="when running sampling",
        context="in the process of sampling",
    )
    def _run_mcmc(self):
        self.sampler.reset()
        self.sampler.run_mcmc(
            initial_state=self._params["initial_state"],
            nsteps=self._params["nsteps"],
            log_prob0=self._params["log_prob0"],
            rstate0=self._params["rstate0"],
            blobs0=self._params["blobs0"],
            tune=self._params["tune"],
            skip_initial_state_check=self._params["skip_initial_state_check"],
            thin_by=self._params["thin_by"],
            thin=self._params["thin"],
            store=self._params["store"],
            progress=self._params["progress"],
            progress_kwargs=self._params["progress_kwargs"],
        )


@functools.lru_cache(maxsize=None)
def _init_class_methods():
    import inspect
    from emcee import EnsembleSampler

    _emcee_EnsembleSampler_args = dict(inspect.signature(EnsembleSampler).parameters)
    _emcee_EnsembleSampler_sample_args = dict(
        inspect.signature(EnsembleSampler.sample).parameters
    )
    required_in_problem = {"log_posterior"}
    optional_in_problem = dict()
    required_in_options = {"nwalkers", "nsteps", "initial_state"}
    optional_in_options = {
        k: v.default
        for k, v in _emcee_EnsembleSampler_args.items()
        if k not in {"nwalkers", "ndim", "log_prob_fn", "self", "args", "kwargs"}
    }
    optional_in_options.update(
        {
            k: v.default
            for k, v in _emcee_EnsembleSampler_sample_args.items()
            if k not in {"iterations", "self"}
        }
    )
    return (
        required_in_problem,
        optional_in_problem,
        required_in_options,
        optional_in_options,
    )
