from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import BaseObjective, Model

import warnings
from typing import Callable, Union
import numpy as np
from scipy.optimize import minimize


# methods available in scipy (can be string or callable):
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

# 'jac' is only for:
# CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

# 'hess' is only for:
# Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

# 'hessp' is only for:
# Newton-CG, trust-ncg, trust-krylov, trust-constr

# 'bounds' is only for:
# Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr

# 'constraints' is only for:
# COBYLA, SLSQP and trust-constr

# other arguments include: tol, options, callback


class ScipyOptimizerSolver(BaseSolver):
    def __init__(self, objective: BaseObjective):
        self._obj = objective
        self._x0 = objective.m0
        self._t = objective.x
        self._y = objective.y

    def solve(
        self,
        method: Union[str, Callable]=None,
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

        :param method: optimization algorithm, check 
            `SciPy's documentation on minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            for a detailed list and argumnets required for each method. Defaults to None
        :type method: Union[str, Callable], optional
        :param gradient: a function that calculates the gradient of your objective 
            function from a given model (numpy.ndarray). Only for CG, BFGS, Newton-CG, 
            L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and 
            trust-constr. Defaults to None
        :type gradient: {callable, ‘2-point’, ‘3-point’, ‘cs’, bool}, optional
        :param hess: a function that calculates the Hessian of your objective 
            function from a given model (numpy.ndarray). Only for Newton-CG, dogleg, 
            trust-ncg, trust-krylov, trust-exact and trust-constr. Defaults to None
        :type hess: {callable, ‘2-point’, ‘3-point’, ‘cs’, HessianUpdateStrategy}, optional
        :param hessp: a function that calculates the Hessian of objective function 
            times an arbitrary vector p. Only for Newton-CG, trust-ncg, trust-krylov, 
            trust-constr. Only one of hessp or hess needs to be given (if hess is 
            provided, then hessp will be ignored). Defaults to None
        :type hessp: callable, optional
        :param bounds: bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, 
            and trust-constr methods. Defaults to None
        :type bounds: sequence or `scipy.optimize.Bounds <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds>`_, optional
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
        if gradient is None:
            gradient = self._obj.gradient
        if hess is None:
            hess = self._obj.hessian

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        res = minimize(
            self._obj.objective,
            self._x0,
            method=method,
            jac=gradient,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options,
            callback=callback,
        )

        model = Model(
            **dict(
                [("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res.x)]
            )
        )
        return model
