import inspect
import numpy as np
from emcee import EnsembleSampler

from . import BaseSolver


class EmceeSolver(BaseSolver):
    documentation_links = [
        "https://emcee.readthedocs.io/en/stable/user/sampler/#emcee.EnsembleSampler",
        "https://emcee.readthedocs.io/en/stable/user/sampler/#emcee.EnsembleSampler.sample",
    ]
    short_description = (
        "an MIT licensed pure-Python implementation of Goodman & Weare’s Affine "
        "Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler"
    )

    _emcee_EnsembleSampler_args = dict(inspect.signature(EnsembleSampler).parameters)
    _emcee_EnsembleSampler_sample_args = dict(
        inspect.signature(EnsembleSampler.sample).parameters
    )
    required_in_problem = {"log_posterior", "model_shape", "walkers_starting_pos"}
    optional_in_problem = dict()
    required_in_options = {"nwalkers", "nsteps"}
    optional_in_options = {
        k: v.default
        for k, v in _emcee_EnsembleSampler_args.items()
        if k not in {"nwalkers", "ndim", "log_prob_fn", "self", "args", "kwargs"}
    }
    optional_in_options.update(
        {
            k: v.default
            for k, v in _emcee_EnsembleSampler_sample_args.items()
            if k not in {"initial_state", "iterations"}
        }
    )

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)
        self._assign_args()
        self.sampler = EnsembleSampler(
            nwalkers=self._nwalkers,
            ndim=self._ndim,
            log_prob_fn=self._log_prob_fn,
            pool=self._pool,
            moves=self._moves,
            args=None,          # already handled by BaseProblem
            kwargs=None,        # already handled by BaseProblem
            backend=self._backend,
            vectorize=self._vectorize,
            blobs_dtype=self._blobs_dtype,
            parameter_names=self._parameter_names,
            a=self._a,
            postargs=self._postargs,
            threads=self._threads,
            live_dangerously=self._live_dangerously,
            runtime_sortingfn=self._runtime_sortingfn,
        )

    def _assign_args(self):
        # assign options
        self._assign_options()
        # assign components in problem to args
        inv_problem = self.inv_problem
        self.components_used = list(self.required_in_problem)
        self._blobs_dtype = None
        self._blob_names = None
        if inv_problem.log_posterior_with_blobs_defined:
            self._log_prob_fn = inv_problem.log_posterior_with_blobs
            if inv_problem.blobs_dtype_defined:
                self._blob_names = [name for (name, _) in inv_problem.blobs_dtype]
                # uncomment below once this has been fixed:
                # issue: https://github.com/arviz-devs/arviz/issues/2036
                # self._blobs_dtype = inv_problem._blobs_dtype
        else:
            self._log_prob_fn = inv_problem.log_posterior
        self._ndim = np.prod(inv_problem.model_shape)
        self._initial_state = inv_problem.walkers_starting_pos

    def __call__(self) -> dict:
        self.sampler.reset()
        self.sampler.run_mcmc(
            initial_state=self._initial_state,
            nsteps=self._nsteps,
            log_prob0=self._log_prob0,
            rstate0=self._rstate0,
            blobs0=self._blobs0,
            tune=self._tune,
            skip_initial_state_check=self._skip_initial_state_check,
            thin_by=self._thin_by,
            thin=self._thin,
            store=self._store,
            progress=self._progress,
            progress_kwargs=self._progress_kwargs,
        )
        result = {
            "success": True,
            "sampler": self.sampler,
            "blob_names": self._blob_names
        }
        return result
