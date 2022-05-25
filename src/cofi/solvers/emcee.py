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
    _emcee_EnsembleSampler_sample_args = dict(inspect.signature(EnsembleSampler.sample).parameters)
    required_in_problem = {"log_posterior", "initial_model", "model_shape"}
    optional_in_problem = dict()
    required_in_options = {"nwalkers", "nsteps"}
    optional_in_options = {
        k: v.default
        for k, v in _emcee_EnsembleSampler_args.items()
        if k not in {"nwalkers", "ndim", "log_prob_fn", "self"}
    }
    optional_in_options.update({
        k: v.default
        for k, v in _emcee_EnsembleSampler_sample_args.items()
        if k not in {"initial_state", "iterations"}
    })

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
            args=self._args,
            kwargs=self._kwargs,
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
        self._log_prob_fn = inv_problem.log_posterior
        self._ndim = np.prod(inv_problem.model_shape)
        self._initial_state = inv_problem.initial_model + \
            1e-4 * np.random.randn(self._nwalkers, self._ndim)
        # TODO is the above a good choice? (generate starting points for users)

    def __call__(self) -> dict:
        self.sampler.reset()
        res_state = self.sampler.run_mcmc(
            initial_state=self._initial_state,
            nsteps = self._nsteps,
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
        }
        # result["acceptance_fraction"] = self.sampler.acceptance_fraction  # property
        # result["random_state"] = self.sampler.random_state                # property
        # result["get_autocorr_time"] = self.sampler.get_autocorr_time      # func
        # result["get_blobs"] = self.sampler.get_blobs                      # func
        # result["get_chain"] = self.sampler.get_chain                      # func
        # result["get_log_prob"] = self.sampler.get_log_prob                # func
        return result
