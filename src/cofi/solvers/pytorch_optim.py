import torch

from . import BaseSolver


class CofiObjective(torch.autograd.Function):
    # https://pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html
    @staticmethod
    def forward(ctx, m, my_objective, my_gradient):
        # calculate and save gradient value
        grad = my_gradient(m)
        if not torch.is_tensor(grad):       # converting type only when not tensor
            grad = torch.tensor(grad)
        ctx.save_for_backward(grad)
        # calculate and return objective value
        obj_val = my_objective(m)
        if not torch.is_tensor(obj_val):    # converting type only when not tensor
            obj_val = torch.tensor(obj_val, requires_grad=True)
        return obj_val

    @staticmethod
    def backward(ctx, _):
        grad = ctx.saved_tensors[0]
        return grad, None, None


class PyTorchOptim(BaseSolver):
    documentation_links = [
        "https://pytorch.org/docs/stable/optim.html#algorithms",
    ]
    short_description = "PyTorch Optimizers under module `pytorch.optim`"

    required_in_problem = {"objective", "gradient", "initial_model"}
    optional_in_problem = dict()                            # TODO
    required_in_options = {"algorithm", "num_iterations"}
    optional_in_options = {"verbose": True, "lr":0.01}                 # TODO

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
        self._validate_algorithm()

        # validate algorithm and instantiate optimizer
        if self._params["algorithm"] not in self.available_algs:
            raise ValueError(
                f"You've chosen an invalid algorithm {self._algorithm}. "
                f"Please choose from: {self.available_algs}."
            )

        # save problem info for later use
        self._obj = self.inv_problem.objective
        self._grad = self.inv_problem.gradient
        self._m = torch.tensor(
            self.inv_problem.initial_model, dtype=float, requires_grad=True
        )

        # instantiate torch optimizer
        self.torch_optimizer = getattr(torch.optim, self._params["algorithm"])(
            [self._m], 
            lr = self._params["lr"],
            # TODO (the rest parameters)
        )

        # instantiate torch misfit function
        self.torch_objective = CofiObjective.apply

    def __call__(self) -> dict:
        losses = []
        for i in range(self._params["num_iterations"]):
            self.torch_optimizer.zero_grad()
            obj = self.torch_objective(self._m, self._obj, self._grad)
            obj.backward()
            self.torch_optimizer.step()
            losses.append(obj)
            if self._params["verbose"]:
                print(f"Iteration #{i}, objective value: {obj}")
        return {
            "model": self._m.detach().numpy(),
            "objective_value": obj.detach().numpy(),
            "losses": losses,
        }

    def _validate_algorithm(self):
        if self._params["algorithm"] not in self.available_algs:
            raise ValueError(
                f"the algorithm you've chosen ({self._params['algorithm']}) "
                f"is invalid. Please choose from the following: {self.available_algs}"
            )
