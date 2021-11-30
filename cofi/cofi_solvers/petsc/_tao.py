from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import BaseObjective, Model

import numpy as np
from petsc4py import PETSc


# some of the methods available in PETSc 
# https://petsc.org/main/docs/manualpages/Tao/TaoSetType.html
methods=["nm","lmvm","nls","ntr","cg","blmvm","tron"]

class TAOSolver(BaseSolver):
    def __init__(self, objective: BaseObjective):
        self.obj = objective
        self.tao_app_ctx = _TAOAppCtx(objective)
        self.n_params = self.tao_app_ctx.nm


    def solve(self, method: str) -> Model:
        # create TAO Solver
        tao = PETSc.TAO().create(PETSc.COMM_SELF)
        tao.setType(method)
        tao.setFromOptions()

        # solve the problem
        tao.setObjectiveGradient(self.tao_app_ctx.formObjGrad)
        tao.setObjective(self.tao_app_ctx.formObjective)
        tao.setGradient(self.tao_app_ctx.formGradient)
        tao.setHessian(self.tao_app_ctx.formHessian, self.H)
        self.x.setValues(range(0, self.n_params), self.tao_app_ctx.x0)
        tao.setInitial(self.x)
        tao.solve(self.x)
        print('------------------', method, '------------------')
        self.x.view()
        tao.destroy()


    def pre_solve(self):
        # create solution vector
        self.x = PETSc.Vec().create(PETSc.COMM_SELF)
        self.x.setSizes(self.n_params)
        self.x.setFromOptions()
        self.x.setValues(range(0, self.n_params), self.tao_app_ctx.x0)

        # create Hessian matrix
        self.H = PETSc.Mat().create(PETSc.COMM_SELF)
        self.H.setSizes([self.n_params, self.n_params])
        self.H.setFromOptions()
        self.H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.H.setUp()


class _TAOAppCtx:
    """Application context struct in C, required in TAO.
        This class is initialised and converted from an objective.
    """

    def __init__(self, objective: BaseObjective):
        self._obj = objective
        self.y = objective.y
        self.t = objective.x
        self.x0 = objective.m0
        self.nm = objective.n_params

    def formObjective(self, tao, x):
        return self._obj.objective(x)

    def formGradient(self, tao, x, G):
        G.setArray(self._obj.gradient(x))
        G.assemble()

    def formObjGrad(self, tao, x, G):
        self.formGradient(tao, x, G)
        return self.formObjective(tao, x)

    def formHessian(self, tao, x, H, HP):
        n_params = self._obj.n_params
        H.setValues(range(0, n_params), range(0, n_params), self._obj.hessian(x))
        H.assemble()
