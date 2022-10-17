import torch

from . import BaseSolver


# class ForwardOperator(torch.nn.Module):
#     def __init__(self, x0, forward_func):
#         super().__init__()
#         self.weights = torch.nn.Parameter(x0)
#         self.forward_func = forward_func
    
#     def forward(self, x):
#         try:
#             return self.forward_func(x)
#         except:
#             return self.forward_func(x.numpy())


class Objective(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, my_objective, my_gradient):
        ctx.save_for_backward(m, my_gradient(m))
        return my_objective(m)

    @staticmethod
    def backward(ctx, grad_output):
        _, grad = ctx.saved_tensors
        return grad, None, None


class PyTorchOptim(BaseSolver):
    documentation_links = [
        "https://pytorch.org/docs/stable/optim.html#algorithms",
    ]
    short_description = (
        "PyTorch Optimizers under module `pytorch.optim`"
    )

    required_in_problem = {"objective", "gradient", "initial_model"}
    optional_in_problem = dict()
    required_in_options = {"algorithm", "num_iterations"}
    optional_in_options = {"verbose": True}

    available_algs = [
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "SparseAdam",
        "Adamax",
        "ASGD",
        "LBFGS",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD",
    ]

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self.components_used = list(self.required_in_problem)
        self._assign_options()

        # validate algorithm and instantiate optimizer
        if self._algorithm not in self.available_algs:
            raise ValueError(
                f"You've chosen an invalid algorithm {self._algorithm}. "
                f"Please choose from: {self.available_algs}."
            )

        # save problem info for later use
        self._obj = self.inv_problem.objective
        self._grad = self.inv_problem.gradient
        self._m = torch.tensor(self.inv_problem.initial_model, dtype=float, requires_grad=True)

        # instantiate torch optimizer
        self.torch_optimizer = getattr(torch.optim, self._algorithm)([self._m], lr=self.inv_options.hyper_params["lr"])

        # instantiate torch misfit function
        self.torch_objective = Objective.apply
    
    def __call__(self) -> dict:
        for i in range(self._num_iterations):
            self.torch_optimizer.zero_grad()
            obj = self.torch_objective(self._m, self._obj, self._grad)
            obj.backward()
            self.torch_optimizer.step()
            if self._verbose:
                print(f"Iteration #{i}, objective value: {obj}")
        return {"model": self._m.detach().numpy(), "objective_value": obj.detach().numpy()}
