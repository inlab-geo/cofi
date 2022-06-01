import pytest
import numpy as np
import emcee

from cofi import SamplingResult, InversionResult


def test_validation():
    # 1 - InversionResult
    with pytest.raises(ValueError, match=r".*status not returned.*"):
        res = InversionResult(dict())
    # 2 - SamplingResult
    with pytest.raises(ValueError, match=r".*sampler not found.*"):
        res = SamplingResult({"success": True})

def test_success_status():
    # 1
    res = InversionResult({"success": True})
    assert res.success_or_not == "success"
    # 2
    res = InversionResult({"success": 1})
    assert res.success_or_not == "success"
    # 3
    res = InversionResult({"success": False})
    assert res.success_or_not == "failure"
    # 4
    res = InversionResult({"success": 0})
    assert res.success_or_not == "failure"

def test_to_arviz():
    # 1 - None sampler
    res = SamplingResult({"success": True, "sampler": None})
    with pytest.raises(ValueError):
        inf_data = res.to_arviz()
    # 2 - incorrect sampler type
    res = SamplingResult({"success": True, "sampler": np.array([1])})
    with pytest.raises(NotImplementedError):
        inf_data = res.to_arviz()
    # 3 - correct sampler (posterior)
    def dummy_pdf(model):
        if model < 0 or model > 1:
            return -np.inf
        return 0.0 # model lies within bounds -> return log(1)
    log_prob = dummy_pdf
    sampler = emcee.EnsembleSampler(2,1,log_prob)
    sampler.run_mcmc(np.array([[0.1],[0.3]]), 100)
    res = SamplingResult({"success": True, "sampler": sampler})
    idata = res.to_arviz()
    # 4 - correct sampler (prior + likelihood)
    def log_prob(model):
        pdf = dummy_pdf(model)
        return pdf+pdf, pdf, pdf
    sampler = emcee.EnsembleSampler(2,1,log_prob)
    sampler.run_mcmc(np.array([[0.1],[0.3]]), 100)
    res = SamplingResult({"success": True, "sampler": sampler, "blob_names": ["ll","lp"]})
    with pytest.warns(UserWarning, match=".*group is not defined.*"):
        idata = res.to_arviz()
    # 5 - correct sampler (prior + likelihood + blob_groups)
    idata = res.to_arviz(blob_groups=["log_likelihood", "prior"])

