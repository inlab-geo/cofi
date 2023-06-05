import pytest

from cofi import BaseProblem
from cofi._exceptions import NotDefinedError


@pytest.fixture
def empty_problem():
    return BaseProblem()


def test_default_not_defined_error(empty_problem):
    with pytest.raises(NotDefinedError):
        empty_problem.objective(1)
    with pytest.raises(NotDefinedError):
        empty_problem.gradient(1)
    with pytest.raises(NotDefinedError):
        empty_problem.hessian(1)
    with pytest.raises(NotDefinedError):
        empty_problem.hessian_times_vector(1, 2)
    with pytest.raises(NotDefinedError):
        empty_problem.residual(1)
    with pytest.raises(NotDefinedError):
        empty_problem.jacobian(1)
    with pytest.raises(NotDefinedError):
        empty_problem.jacobian_times_vector(1, 2)
    with pytest.raises(NotDefinedError):
        empty_problem.data_misfit(1)
    with pytest.raises(NotDefinedError):
        empty_problem.regularization(1)
    with pytest.raises(NotDefinedError):
        empty_problem.regularization_matrix(1)
    with pytest.raises(NotDefinedError):
        empty_problem.forward(1)
    with pytest.raises(NotDefinedError):
        empty_problem.data
    with pytest.raises(NotDefinedError):
        empty_problem.data_covariance
    with pytest.raises(NotDefinedError):
        empty_problem.data_covariance_inv
    with pytest.raises(NotDefinedError):
        empty_problem.initial_model
    with pytest.raises(NotDefinedError):
        empty_problem.model_shape
    with pytest.raises(NotDefinedError):
        empty_problem.bounds
    with pytest.raises(NotDefinedError):
        empty_problem.constraints
    with pytest.raises(NotDefinedError):
        empty_problem.log_posterior(1)
    with pytest.raises(NotDefinedError):
        empty_problem.log_prior(1)
    with pytest.raises(NotDefinedError):
        empty_problem.log_likelihood(1)
    with pytest.raises(NotDefinedError):
        empty_problem.log_posterior_with_blobs(1)
    with pytest.raises(NotDefinedError):
        empty_problem.blobs_dtype


def test_default_not_defined_properties(empty_problem):
    assert not empty_problem.objective_defined
    assert not empty_problem.gradient_defined
    assert not empty_problem.hessian_defined
    assert not empty_problem.hessian_times_vector_defined
    assert not empty_problem.residual_defined
    assert not empty_problem.jacobian_defined
    assert not empty_problem.jacobian_times_vector_defined
    assert not empty_problem.data_misfit_defined
    assert not empty_problem.regularization_defined
    assert not empty_problem.regularization_matrix_defined
    assert not empty_problem.forward_defined
    assert not empty_problem.data_defined
    assert not empty_problem.data_covariance_defined
    assert not empty_problem.data_covariance_inv_defined
    assert not empty_problem.initial_model_defined
    assert not empty_problem.model_shape_defined
    assert not empty_problem.bounds_defined
    assert not empty_problem.constraints_defined
    assert not empty_problem.log_posterior_defined
    assert not empty_problem.log_prior_defined
    assert not empty_problem.log_likelihood_defined
    assert not empty_problem.log_posterior_with_blobs_defined
    assert not empty_problem.blobs_dtype_defined


def test_default_defined_components(empty_problem):
    assert len(empty_problem.defined_components()) == 0


def test_default_summary(empty_problem):
    empty_problem.summary()
