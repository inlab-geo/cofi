import functools

from . import BaseInferenceTool, error_handler


class PyTorchOptim(BaseInferenceTool):
    documentation_links = [
        "https://pytorch.org/docs/stable/optim.html#algorithms",
    ]
    short_description = "PyTorch Optimizers under module `torch.optim`"

    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective", "gradient", "initial_model"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return dict()

    @classmethod
    def required_in_options(cls) -> set:
        return {"algorithm", "num_iterations"}

    @classmethod
    def optional_in_options(cls) -> dict:
        return {
            "verbose": True,
            "callback": None,
            "algorithm_params": dict(),
        }

    @classmethod
    @functools.lru_cache(maxsize=None)
    def available_algorithms(cls) -> set:
        import torch

        optim_dir = dir(torch.optim)
        algs = {name for name in optim_dir if name[0].isupper() and name != "Optimizer"}
        return algs

    def __init__(self, inv_problem, inv_options):
        # save extra options into inv_options.hyper_params["algorithm_params"]
        if "algorithm_params" not in inv_options.hyper_params:
            inv_options.hyper_params["algorithm_params"] = dict()
        for param in list(inv_options.hyper_params):
            if (
                param not in self.optional_in_options()
                and param not in self.required_in_options()
            ):
                inv_options.hyper_params["algorithm_params"][param] = (
                    inv_options.hyper_params.pop(param)
                )

        # initialisation, validation
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        self._validate_algorithm()

        # save problem info for later use
        self._obj = self.inv_problem.objective
        self._grad = self.inv_problem.gradient

        # initialize torch stuff
        self._initialize_torch_tensor()
        self._initialize_torch_optimizer()
        self._initialize_torch_objective()

        # initialize function evaluation counter
        self._nb_evaluations = 0

    def __call__(self) -> dict:
        import torch

        losses = []
        for i in range(self._params["num_iterations"]):
            self._one_iteration(i, losses)
        losses_out = torch.stack(losses)
        return {
            "model": self._m.detach().numpy(),
            "objective_value": self._last_loss.detach().numpy(),
            "losses": losses_out,
            "n_obj_evaluations": self._nb_evaluations,
            "n_grad_evaluations": self._nb_evaluations,
            "success": True,
        }

    def _validate_algorithm(self):
        if self._params["algorithm"] not in self.available_algorithms():
            raise ValueError(
                f"the algorithm you've chosen ({self._params['algorithm']}) "
                "is invalid. Please choose from the following: "
                f"{self.available_algorithms()}"
            )

    @error_handler(
        when="in converting initial_model into PyTorch Tensor with requires_grad=True",
        context="before solving the optimization problem",
    )
    def _initialize_torch_tensor(self):
        import torch

        if torch.is_tensor(self.inv_problem.initial_model):
            self._m = (
                self.inv_problem.initial_model.double()
                .clone()
                .detach()
                .requires_grad_(True)
            )
        else:
            self._m = torch.tensor(
                self.inv_problem.initial_model, dtype=float, requires_grad=True
            )

    @error_handler(
        when=f"in creating PyTorch Optimizer",
        context="before solving the optimization problem",
    )
    def _initialize_torch_optimizer(self):
        import torch

        self.torch_optimizer = getattr(torch.optim, self._params["algorithm"])(
            [self._m],
            **self._params["algorithm_params"],
        )

    @error_handler(
        when="in creating PyTorch custom Loss Function",
        context="before solving the optimization problem",
    )
    def _initialize_torch_objective(self):
        self.torch_objective = _CoFIObjective().apply

    @error_handler(
        when="when performing optimization stepping",
        context="in the process of solving",
    )
    def _one_iteration(self, i, losses):
        def closure():
            self.torch_optimizer.zero_grad()
            self._last_loss = self.torch_objective(self._m, self._obj, self._grad)
            self._last_loss.backward()
            self._nb_evaluations += 1
            return self._last_loss

        self.torch_optimizer.step(closure)
        losses.append(self._last_loss)
        if self._params["callback"] is not None:
            self._run_callback()
        if self._params["verbose"]:
            print(f"Iteration #{i}, objective value: {self._last_loss}")

    @error_handler(
        when="when running your callback function",
        context="in the process of solving",
    )
    def _run_callback(self):
        self._params["callback"](self._m)


def _CoFIObjective():
    import torch

    class CoFIObjective(torch.autograd.Function):
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

    return CoFIObjective


# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> Adadelta
# description: ADADELTA, an Adaptive Learning Rate Method.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> Adagrad
# description: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> Adam
# description: Adam, a Method for Stochastic Optimization.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> AdamW
# description: Decoupled Weight Decay Regularization.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> SparseAdam
# description: Lazy version of Adam algorithm suitable for sparse tensors.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> Adamax
# description: Adamax algorithm (a variant of Adam based on infinity norm).
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> ASGD
# description: Averaged Stochastic Gradient Descent.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> LBFGS
# description: L-BFGS algorithm, heavily inspired by minFunc.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> NAdam
# description: NAdam algorithm, incorporating Nesterov Momentum into Adam
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> RAdam
# description: RAdam algorithm, on the variance of the adaptive learning rate and beyond
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> RMSprop
# description: RMSprop, a form of stochastic gradient descent where the gradients are divided by a running average of their recent magnitude
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> Rprop
# description: The resilient backpropagation algorithm.
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop

# CoFI -> Parameter estimation -> Optimization -> Non linear -> torch.optim -> SGD
# description: The stochastic gradient descent (optionally with momentum).
# documentation: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
