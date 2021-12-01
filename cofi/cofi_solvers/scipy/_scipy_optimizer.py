from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import BaseObjective, Model

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

# 'buonds' is only for:
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
        method: Union[str, Callable] = None,
        jac=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=None,
        tol=None,
        options=None,
        callback=None,
    ) -> Model:
        if jac is None:
            jac = self._obj.jacobian
        if hess is None:
            hess = self._obj.hessian

        res = minimize(
            self._obj.objective,
            self._x0,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options,
            callback=callback,
        )
        res_x = res.x
