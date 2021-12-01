from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import BaseObjective, Model

import numpy as np
from petsc4py import PETSc
from typing import Union, List


# ref doc for TAO: https://petsc.org/release/docs/manual/tao/

# methods available in PETSc 
# https://petsc.org/main/docs/manualpages/Tao/TaoSetType.html
valid_methods_unconstrained_min = ["nls", "ntr", "ntl", "lmvm", "cg", "nm", "tron", "gpcg", "blmvm", "pounders"]
valid_methods_brgn = ["brgn"]


class TAOSolver(BaseSolver):
    def __init__(self, objective: BaseObjective):
        self.obj = objective
        self.tao_app_ctx = _TAOAppCtx(objective)
        self.n_params = self.tao_app_ctx.nm
        self.n_points = self.tao_app_ctx.t.shape[0]


    def solve(self, method: str, extra_options: Union[List[str], str] =None, verbose=1) -> Model:
        # access PETSc options database
        if extra_options:
            OptDB = PETSc.Options()
            if isinstance(extra_options, list):
                for option in extra_options:
                    OptDB.insertString(option)
            else:
                OptDB.insertString(extra_options)
        
        self._pre_solve(method)
        user = self.tao_app_ctx

        if method in valid_methods_unconstrained_min:
            # create TAO Solver
            tao = PETSc.TAO().create(PETSc.COMM_SELF)
            tao.setType(method)
            tao.setFromOptions()

            # solve the problem
            tao.setObjectiveGradient(user.formObjGrad)
            tao.setObjective(user.formObjective)
            tao.setGradient(user.formGradient)
            tao.setHessian(user.formHessian, self.H)
            self.x.setValues(range(0, self.n_params), user.x0)
            tao.setInitial(self.x)
            tao.solve(self.x)

        elif method in valid_methods_brgn:
            # create TAO solver
            tao = PETSc.TAO().create(PETSc.COMM_SELF)
            tao.setType(method)
            tao.setFromOptions()

            # solve the problem
            tao.setResidual(user.evaluateFunction, self.f)
            tao.setJacobianResidual(user.evaluateJacobian, self.J, self.Jp)
            self.x.setValues(range(0, self.n_params), user.x0)
            tao.solve(self.x)
        
        if verbose:
            print('------------------', method, '------------------')
            self.x.view()
        tao.destroy()


    def _pre_solve(self, method: str):
        if method in valid_methods_unconstrained_min:
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
        
        elif method in valid_methods_brgn:
            # create solution vector
            self.x = PETSc.Vec().create(PETSc.COMM_SELF)
            self.x.setSizes(self.n_params)
            self.x.setFromOptions()
            self.x.setValues(range(0, self.n_params), self.tao_app_ctx.x0)

            # create residual vector etc as PETSc vectors and matrices
            self.f = PETSc.Vec().create(PETSc.COMM_SELF)
            self.f.setSizes(self.n_points)
            self.f.setFromOptions()
            self.f.set(0)

            self.J = PETSc.Mat().createDense([self.n_points, self.n_params])
            self.J.setFromOptions()
            self.J.setUp()

            self.Jp = PETSc.Mat().createDense([self.n_points, self.n_params])
            self.Jp.setFromOptions()
            self.Jp.setUp() 

        # TODO - there are other TaoType not implemented yet here
        # else:
        #     raise ValueError(f"Method {method} is not valid. Check https://petsc.org/release/docs/manualpages/Tao/TaoType.html#TaoType for available methods.")


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
        if self._obj.gradient:
            G.setArray(self._obj.gradient(x))
            G.assemble()
        else:
            self.formGradient = None

    def formObjGrad(self, tao, x, G):
        if self._obj.gradient:
            self.formGradient(tao, x, G)
            return self.formObjective(tao, x)
        else:
            self.formObjGrad = None

    def formHessian(self, tao, x, H, HP):
        if self._obj.hessian:
            H.setValues(range(0, self.nm), range(0, self.nm), self._obj.hessian(x))
            H.assemble()
        else:
            self.formHessian = None

    def evaluateJacobian(self, tao, x, J, Jp):
        if self._obj.jacobian:
            jac = self._obj.jacobian(x)
            J.setValues(range(0, self.t.shape[0]), range(0, self.nm), jac)
            J.assemble()
            Jp.setValues(range(0, self.t.shape[0]), range(0, self.nm), jac)
            Jp.assemble()
        else:
            self.evaludateJacobian = None

    def evaluateFunction(self, tao, x, f):
        if self._obj.residuals:
            f.setArray(self._obj.residuals(x))
            f.assemble()
        else:
            self.evaluateFunction = None

