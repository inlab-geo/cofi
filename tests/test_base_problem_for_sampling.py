import numpy

from cofi import BaseProblem


_dummy_dist = lambda m: -numpy.inf if (m[0]<0 or m[0]>1) else 0
_dummy_dist_with_blobs = lambda m: (_dummy_dist(m), m, 2*m)
_blobs_dtype = [("model", float), ("model_doubled", float)]

def test_prior_likelihood():
    inv_problem = BaseProblem()
    inv_problem.set_log_prior(_dummy_dist)
    inv_problem.set_log_likelihood(_dummy_dist)
    assert inv_problem.log_prior_defined
    assert inv_problem.log_likelihood_defined
    assert inv_problem.log_posterior_defined

def test_prior_likelihood_skip_ll():
    inv_problem = BaseProblem()
    inv_problem.set_log_prior(_dummy_dist)
    def _failing_dist(_): assert "this line shouldn't be reached"
    inv_problem.set_log_likelihood(_failing_dist)
    inv_problem.log_posterior(numpy.array([-1]))

def test_posterior():
    inv_problem = BaseProblem()
    inv_problem.set_log_posterior(_dummy_dist)
    assert not inv_problem.log_prior_defined
    assert not inv_problem.log_likelihood_defined
    assert inv_problem.log_posterior_defined

def test_posterior_with_blobs():
    inv_problem = BaseProblem()
    inv_problem.set_log_posterior_with_blobs(_dummy_dist_with_blobs, _blobs_dtype)
    assert not inv_problem.log_prior_defined
    assert not inv_problem.log_likelihood_defined
    assert inv_problem.log_posterior_defined
    assert inv_problem.log_posterior_with_blobs_defined
    assert inv_problem.blobs_dtype_defined
