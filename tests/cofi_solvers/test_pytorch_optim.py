import numpy as np
import pytest

from cofi.solvers import PyTorchOptim
from cofi import BaseProblem, InversionOptions, Inversion


def test_invalid_algorithm():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda: 3)
    inv_problem.set_gradient(lambda: 3)
    inv_problem.set_initial_model(1)
    inv_options = InversionOptions()
    inv_options.set_params(algorithm="Adamm", num_iterations=1)
    with pytest.raises(ValueError, match=r".*algorithm.*is invalid.*"):
        solver = PyTorchOptim(inv_problem, inv_options)

def test_run_simple_obj():
    inv_problem = BaseProblem()
    inv_problem.set_objective(lambda x: (x-3)**2)
    inv_problem.set_initial_model(30)
    inv_problem.set_gradient(lambda x: 2*x - 6)
    inv_problem.set_hessian(lambda x: 2)
    inv_options = InversionOptions()
    inv_options.set_params(algorithm="SGD", lr=0.2, num_iterations=10)
    solver = PyTorchOptim(inv_problem, inv_options)
    res = solver()
    assert pytest.approx(res["model"], abs=1) == 3

def test_run_lin_regression():
    # problem setup code
    nparams = 4
    G = lambda x: np.array([x**i for i in range(4)]).T
    _m_true = np.array([-6,-5,2,1])
    sample_size = 20
    x = np.random.choice(np.linspace(-3.5,2.5), size=sample_size)
    forward = lambda m: G(x).dot(m)
    y_obs = forward(_m_true) + np.random.normal(0,1,sample_size)
    objective = lambda m: np.sum(np.square(y_obs - forward(m)))
    objective_grad = lambda m: - 2 * np.expand_dims(y_obs - forward(m),1) * G(x)
    # BaseProblem definition
    inv_problem = BaseProblem()
    inv_problem.set_objective(objective)
    inv_problem.set_gradient(objective_grad)
    inv_problem.set_initial_model(np.zeros(nparams))
    # InversionOptions definition
    inv_options = InversionOptions()
    inv_options.set_params(algorithm="Adam", num_iterations=200, lr=0.1)
    # solve
    solver = PyTorchOptim(inv_problem, inv_options)
    res = solver()
    for i in range(nparams):
        assert pytest.approx(res["model"][i], abs=1) == _m_true[i]


# ----> original example script
if __name__ == "__main__":
    import torch

    # generate data with random Gaussian noise
    nparams = 4
    G = lambda x: np.array([x**i for i in range(4)]).T                        # x -> G
    _m_true = np.array([-6,-5,2,1])                                           # m
    sample_size = 20                                                          # N
    x = np.random.choice(np.linspace(-3.5,2.5), size=sample_size)             # x
    forward = lambda m: G(x).dot(m)                                              # m -> y_synthetic
    y_obs = forward(_m_true) + np.random.normal(0,1,sample_size)              # d
    objective = lambda m: np.sum(np.square(y_obs - forward(m)))               # m -> objective
    objective_grad = lambda m: - 2 * np.expand_dims(y_obs - forward(m),1) * G(x) # m -> gradient

    class MyLossFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, m, my_obj, my_grad):
            ctx.save_for_backward(torch.tensor(my_grad(m)))    
            return torch.tensor([my_obj(m)], requires_grad=True)
        @staticmethod
        def backward(ctx, _):
            grad = ctx.saved_tensors[0]
            return grad, None, None

    m_to_update = torch.tensor(np.zeros(nparams), dtype=float, requires_grad=True)
    opt = torch.optim.Adam([m_to_update], lr=0.01)

    def training_loop(x, optimizer, n=10000):
        loss_func = MyLossFunc.apply
        losses = []
        for _ in range(n):
            loss = loss_func(x, objective, objective_grad)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss)
        return losses

    losses = training_loop(m_to_update, opt)
    print(m_to_update)
