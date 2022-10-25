import torch

from . import BaseSolver


class CofiObjective(torch.autograd.Function):
    # https://pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html
    @staticmethod
    def forward(ctx, m, my_objective, my_gradient):
        # calculate and save gradient value
        grad = my_gradient(m)
        if not torch.is_tensor(grad):  # converting type only when not tensor
            grad = torch.tensor(grad)
        ctx.save_for_backward(grad)
        # calculate and return objective value
        obj_val = my_objective(m)
        if not torch.is_tensor(obj_val):  # converting type only when not tensor
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
    optional_in_problem = dict()
    required_in_options = {"algorithm", "num_iterations"}
    optional_in_options = {"verbose": True, "algorithm_params": dict()}  # TODO

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

        # save options (not "verbose") into self._params["algorithm_params"]
        for param in self.inv_options.hyper_params:
            if param != "verbose" and param not in self.required_in_options:
                self._params["algorithm_params"][param] = self.inv_options.hyper_params[
                    param
                ]

        # save problem info for later use
        self._obj = self.inv_problem.objective
        self._grad = self.inv_problem.gradient
        if torch.is_tensor(self.inv_problem.initial_model):
            self._m = (
                self.inv_problem.initial_model.double()
                .clone()
                .detach()
                .requires_grad_(True)
            )
        else:
            self._m = self._wrap_error_handler(
                torch.tensor,
                args=[self.inv_problem.initial_model],
                kwargs={"dtype": float, "requires_grad": True},
                when="in converting initial_model into PyTorch Tensor",
                context="before solving the optimization problem",
            )

        # instantiate torch optimizer
        self.torch_optimizer = self._wrap_error_handler(
            getattr(torch.optim, self._params["algorithm"]),
            args=[[self._m]],
            kwargs=self._params["algorithm_params"],
            when=f"in creating PyTorch Optimizer '{self._params['algorithm']}'",
            context="before solving the optimization problem",
        )

        # instantiate torch misfit function
        self.torch_objective = self._wrap_error_handler(
            getattr,
            args=[CofiObjective, "apply"],
            kwargs=dict(),
            when="in creating PyTorch custom Loss Function",
            context="before solving the optimization problem",
        )

    def _one_iteration(self, i, losses):
        def closure():
            self.torch_optimizer.zero_grad()
            self._last_loss = self.torch_objective(self._m, self._obj, self._grad)
            self._last_loss.backward()
            return self._last_loss

        self.torch_optimizer.step(closure)
        losses.append(self._last_loss)
        if self._params["verbose"]:
            print(f"Iteration #{i}, objective value: {self._last_loss}")

    def __call__(self) -> dict:
        losses = []
        for i in range(self._params["num_iterations"]):
            self._wrap_error_handler(
                self._one_iteration,
                args=[i, losses],
                kwargs=dict(),
                when="when performing optimization stepping",
                context="in the process of solving",
            )
        return {
            "model": self._m.detach().numpy(),
            "objective_value": self._last_loss.detach().numpy(),
            "losses": losses,
            "success": True,
        }

    def _validate_algorithm(self):
        if self._params["algorithm"] not in self.available_algs:
            raise ValueError(
                f"the algorithm you've chosen ({self._params['algorithm']}) "
                f"is invalid. Please choose from the following: {self.available_algs}"
            )
