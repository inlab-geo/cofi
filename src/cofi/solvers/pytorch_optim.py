import functools

from . import BaseSolver, error_handler


class PyTorchOptim(BaseSolver):
    documentation_links = [
        "https://pytorch.org/docs/stable/optim.html#algorithms",
    ]
    short_description = "PyTorch Optimizers under module `pytorch.optim`"

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
    def available_algorithms(cls) -> list:
        import torch

        optim_dir = dir(torch.optim)
        algs = [name for name in optim_dir if name[0].isupper() and name != "Optimizer"]
        return algs

    def __init__(self, inv_problem, inv_options):
        # save extra options into inv_options.hyper_params["algorithm_params"]
        if "algorithm_params" not in inv_options.hyper_params:
            inv_options.hyper_params["algorithm_params"] = dict()
        for param in list(inv_options.hyper_params):
            print(param)
            if (
                param not in self.optional_in_options()
                and param not in self.required_in_options()
            ):
                print(1)
                inv_options.hyper_params["algorithm_params"][
                    param
                ] = inv_options.hyper_params.pop(param)

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
