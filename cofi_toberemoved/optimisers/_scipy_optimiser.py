import warnings
from typing import Callable, Union
from math import inf

import numpy as np
from scipy.optimize import minimize, least_squares

from .. import Model, BaseObjective, BaseSolver, OptimiserMixin


# methods available in scipy (can be string or callable):
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

# 'jac' is only for:
# CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact
# and trust-constr

# 'hess' is only for:
# Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

# 'hessp' is only for:
# Newton-CG, trust-ncg, trust-krylov, trust-constr

# 'bounds' is only for:
# Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr

# 'constraints' is only for:
# COBYLA, SLSQP and trust-constr

# other arguments include: tol, options, callback


class ScipyOptimiserSolver(BaseSolver, OptimiserMixin):
    """Optimiser wrapper of scipy.optimizer.minimize

    Objective definition needs to implement the following functions:
    - misfit(model)
    - gradient(model), optional depending on method
    - hessian(model), optional depending on method
    - data_x()
    - data_y()
    - initial_model()

    More details on methods and functions to implement WIP... #TODO
    """

    def __init__(self, objective: BaseObjective):
        self._obj = objective
        self._x0 = objective.initial_model()
        self._t = objective.data_x()
        self._y = objective.data_y()

    def solve(
        self,
        method: Union[str, Callable] = None,
        gradient=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=None,
        tol=None,
        options=None,
        callback=None,
    ) -> Model:
        """Solve the "best" model using SciPy's optimization methods.
        It calls scipy.optimize.minimize() in the backend. The objective
        function is necessary for this method. Optionally, `gradient` and
        `hess` may also be used depending on the method selected.

        :param method: optimization algorithm, check :func:`scipy.optimize.minimize`
            for a detailed list and argumnets required for each method. Defaults to None
        :type method: Union[str, Callable], optional
        :param gradient: a function that calculates the gradient of your objective
            function from a given model (numpy.ndarray). Only for CG, BFGS, Newton-CG,
            L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and
            trust-constr. Defaults to the `gradient` function defined in objective.
        :type gradient: {callable, ‘2-point’, ‘3-point’, ‘cs’, bool}, optional
        :param hess: a function that calculates the Hessian of your objective
            function from a given model (numpy.ndarray). Only for Newton-CG, dogleg,
            trust-ncg, trust-krylov, trust-exact and trust-constr. Defaults to the
            `hessian` function defined in objective.
        :type hess: {callable, ‘2-point’, ‘3-point’, ‘cs’, HessianUpdateStrategy}, optional
        :param hessp: a function that calculates the Hessian of objective function
            times an arbitrary vector p. Only for Newton-CG, trust-ncg, trust-krylov,
            trust-constr. Only one of hessp or hess needs to be given (if hess is
            provided, then hessp will be ignored). Defaults to None
        :type hessp: callable, optional
        :param bounds: bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
            and trust-constr methods. Defaults to None
        :type bounds: sequence or :class:`scipy.optimize.Bounds`, optional
        :param constraints: Constraints definition (only for COBYLA, SLSQP and trust-constr).
            Check SciPy documentation for more details on types. Defaults to None
        :type constraints: {Constraint, dict} or List of {Constraint, dict}, optional
        :param tol: solver-specific termination tolerance. Defaults to None
        :type tol: float, optional
        :param options: solver options ('maxiter' and 'disp'). Defaults to None
        :type options: dict, optional
        :param callback: a function that gets called after each iteration. Check Scipy
            documentation for more details on the 'trust-constr' solver. Defaults to None
        :type callback: callable, optional
        :return: the optimization result - a model in the inversion context
        :rtype: cofi.cofi_objective.Model
        """
        if (
            method is None and hasattr(self, "method") and self.method is not None
        ):  # method set by set_method()
            method = self.method

        if gradient is None:
            gradient = self._obj.gradient
        if hess is None:
            hess = self._obj.hessian

        # If unimplementederror is raised, then set gradient and hess to None
        # Scipy optimisers choose methods based on whether gradient or hess are provided
        # Avoid cases where they are not implemented but are also not None
        try:
            gradient(self._x0)
        except:
            gradient = None
        try:
            hess(self._x0)
        except:
            hess = None

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        res = minimize(
            self._obj.misfit,
            self._x0,
            method=method,
            jac=gradient,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=self.options["tol"]
            if tol is None and hasattr(self, "options") and "tol" in self.options
            else tol,
            options=options,
            callback=self.options["callback"]
            if callback is None
            and hasattr(self, "options")
            and "callback" in self.options
            else callback,
        )

        model = Model(
            **dict(
                [("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res.x)]
            )
        )
        return model


class ScipyOptimiserLSSolver(BaseSolver, OptimiserMixin):
    """Optimiser wrapper of scipy.optimizer.least_squares

    Objective definition needs to implement the following functions:
    - residual(model)
    - jacobian(model), optional depending on method
    - data_x()
    - data_y()
    - initial_model()

    More details on methods and functions to implement WIP... #TODO
    """

    def __init__(self, objective: BaseObjective):
        self._obj = objective
        self._x0 = objective.initial_model()
        self._t = objective.data_x()
        self._y = objective.data_y()

    def solve(
        self,
        method: str = None,
        jac=None,
        bounds=(-inf, inf),
        ftol=1e-08,
        xtol=1e-08,
        gtol=1e-08,
        x_scale=1.0,
        loss="linear",
        f_scale=1.0,
        diff_step=None,
        tr_solver=None,
        tr_options={},
        jac_sparsity=None,
        max_nfev=None,
        verbose=0,
        args=(),
        kwargs={},
    ) -> Model:
        if method is None:
            if (
                hasattr(self, "method") and self.method is not None
            ):  # method set by set_method()
                method = self.method
            else:
                method = "trf"

        if jac is None:
            jac = self._obj.jacobian
            if jac is None:
                jac = "2-point"

        res = least_squares(
            self._obj.residual,
            self._x0,
            method=method,
            jac=jac,
            bounds=bounds,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            loss=loss,
            f_scale=f_scale,
            diff_step=diff_step,
            tr_solver=tr_solver,
            tr_options=tr_options,
            jac_sparsity=jac_sparsity,
            max_nfev=max_nfev,
            verbose=verbose,
            args=args,
            kwargs=kwargs,
        )

        model = Model(
            **dict(
                [("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res.x)]
            )
        )
        return model
