from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import BaseObjective, Model

import numpy as np
from petsc4py import PETSc
from typing import Union, List


# ref doc for TAO: https://petsc.org/release/docs/manual/tao/

# methods available in PETSc
# https://petsc.org/main/docs/manualpages/Tao/TaoSetType.html
valid_methods_unconstrained_min = [
    "nls",
    "ntr",
    "ntl",
    "lmvm",
    "cg",
    "nm",
    "tron",
    "gpcg",
    "blmvm",
    "pounders",
]
valid_methods_brgn = ["brgn"]


class TAOSolver(BaseSolver):
    """Optimizer wrapper of TAO

    Objective definition needs to implement the following functions:
    - misfit(model)
    - residuals(model), optional depending on method
    - jacobian(model), optional depending on method
    - gradient(model), optional depending on method
    - hessian(model), optional depending on method
    - data_x()
    - data_y()
    - initial_model()
    - n_params()

    More details on methods and functions to implement WIP... #TODO
    """

    def __init__(self, objective: BaseObjective, mpi=False):
        self.obj = objective
        self._use_mpi = mpi
        if mpi:
            self._comm = PETSc.COMM_WORLD
            self.tao_app_ctx = _TAOAppCtxMPI(objective, self._comm)
        else:
            self._comm = PETSc.COMM_SELF
            self.tao_app_ctx = _TAOAppCtx(objective)

        self.n_params = self.tao_app_ctx.nm
        self.n_points = self.tao_app_ctx.t.shape[0]

    def solve(
        self, method: str, extra_options: Union[List[str], str] = None, verbose=1
    ) -> Model:
        if extra_options:
            self.set_options(extra_options)

        self._pre_solve(method)
        user = self.tao_app_ctx

        if method in valid_methods_unconstrained_min:
            # create TAO Solver
            tao = PETSc.TAO().create(self._comm)
            tao.setType(method)
            tao.setFromOptions()

            # solve the problem
            tao.setObjectiveGradient(user.formObjGrad)
            tao.setObjective(user.formObjective)
            tao.setGradient(user.formGradient)
            tao.setHessian(user.formHessian, self.H)
            tao.setInitial(self.x)
            tao.solve(self.x)

        elif method in valid_methods_brgn:
            # create TAO solver
            tao = PETSc.TAO().create(self._comm)
            tao.setType(method)
            tao.setFromOptions()

            # solve the problem
            tao.setResidual(user.evaluateFunction, self.f)
            tao.setJacobianResidual(user.evaluateJacobian, self.J, self.Jp)
            tao.solve(self.x)

        if verbose:
            print("------------------", method, "------------------")
            if self._use_mpi:
                print("MPI enabled.")
            self.x.view()

        params = self.x.getArray()
        tao.destroy()

        model = Model(
            **dict(
                [("p" + str(index[0]), val) for (index, val) in np.ndenumerate(params)]
            )
        )
        return model


    def set_options(options: Union[List[str], str]):
        # access PETSc options database
        OptDB = PETSc.Options()
        if isinstance(options, list):
            for option in options:
                if not isinstance(option, str):
                    raise ValueError("options of TAOSolver needs to be of type `str` or `List[str]`")
                OptDB.insertString(option)
        elif isinstance(options, str):
            OptDB.insertString(options) 
        else:
            raise ValueError("options of TAOSolver needs to be of type `str` or `List[str]`")

    def _pre_solve(self, method: str):
        if self._use_mpi:
            self._pre_solve_mpi(method)
        else:
            self._pre_solve_non_mpi(method)

    def _pre_solve_non_mpi(self, method: str):
        if method in valid_methods_unconstrained_min:
            # create solution model vector
            self.x = PETSc.Vec().create(self._comm)
            self.x.setSizes(self.n_params)
            self.x.setFromOptions()
            self.x.setValues(range(0, self.n_params), self.tao_app_ctx.x0)

            # create Hessian matrix
            self.H = PETSc.Mat().create(self._comm)
            self.H.setSizes([self.n_params, self.n_params])
            self.H.setFromOptions()
            self.H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self.H.setUp()

        elif method in valid_methods_brgn:
            # create solution vector
            self.x = PETSc.Vec().create(self._comm)
            self.x.setSizes(self.n_params)
            self.x.setFromOptions()
            self.x.setValues(range(0, self.n_params), self.tao_app_ctx.x0)

            # create residual vector etc as PETSc vectors and matrices
            self.f = PETSc.Vec().create(self._comm)
            self.f.setSizes(self.n_points)
            self.f.setFromOptions()
            self.f.set(0)

            self.J = PETSc.Mat().createDense([self.n_points, self.n_params])
            self.J.setFromOptions()
            self.J.setUp()

            self.Jp = PETSc.Mat().createDense([self.n_points, self.n_params])
            self.Jp.setFromOptions()
            self.Jp.setUp()

    def _pre_solve_mpi(self, method: str):
        if method in valid_methods_unconstrained_min:
            pass
            
        elif method in valid_methods_brgn:
            # create solution model vector
            self.x = PETSc.Vec().create(self._comm)
            self.x.setSizes(self.n_params)
            self.x.setType('mpi')
            self.x.setFromOptions()
            self.x.setUp()
            n, m = self.x.getOwnershipRange()
            self.x.setValues(range(n, m), self.tao_app_ctx.x0[n:m])
            self.x.assemble()

            # create time, observation, predicction vectors and Jacobian matrix as mpi types
            self.t = PETSc.Vec().create(self._comm)
            self.t.setSizes(self.n_points)
            self.t.setType('mpi')
            self.t.setFromOptions()
            n, = self.t.getOwnershipRange()
            self.t.setValues(range(n, m), self.tao_app_ctx.t[n:m])
            self.t.assemble()

            self.y = PETSc.Vec().create(self._comm)
            self.y.setSizes(self.n_points)
            self.y.setType('mpi')
            self.y.setFromOptions()
            n, m = self.t.getOwnershipRange()
            self.y.setValues(range(n, m), self.tao_app_ctx.y[n:m])
            self.y.assemble()

            self.f = PETSc.Vec().create(self._comm)
            self.f.setSizes(self.n_points)
            self.f.setType('mpi')
            self.f.setFromOptions()

            self.J = PETSc.Mat().create(self._comm)
            self.J.setType('mpidense')
            self.J.setSizes([self.n_points, self.n_params])
            self.J.setFromOptions()
            self.J.setUp()

            self.Jp = PETSc.Mat().create(self._comm)
            self.Jp.setType('mpidense')
            self.Jp.setSizes([self.n_points, self.n_params])
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
        if objective.data_y is None or objective.data_x is None or objective.initial_model is None:
            raise ValueError("Data x, y and initial model are required for TAO solver")

        self.y = objective.data_y()
        self.t = objective.data_x()
        self.x0 = objective.initial_model()
        self.nm = objective.params_size()

    def formObjective(self, tao, x):
        return self._obj.misfit(x)

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


class _TAOAppCtxMPI:
    def __init__(self, objective: BaseObjective, comm):
        self._obj = objective
        self._comm = comm

        if objective.data_y is None or objective.data_x is None or objective.initial_model is None:
            raise ValueError("Data x, y and initial model are required for TAO solver")

        self.y = objective.data_y()
        self.t = objective.data_x()
        self.x0 = objective.initial_model()
        self.nm = objective.params_size()

    def formSequentialModelVector(self,x):
        self.xseq = PETSc.Vec().create(PETSc.COMM_SELF)
        self.xseq.setType('seq')
        scatter, self.xseq = PETSc.Scatter.toAll(x)
        scatter.begin(x, self.xseq)
        scatter.end(x, self.xseq)

    def evaluateJacobian(self, tao, x, J, Jp):
        self.formSequentialModelVector(x)
        n, m = self.t.getOwnershipRange()
        nx = x.getSize()

        for j in range(n, m):
            tt = self.t.getValue(j)
            for i in range(int(nx/2)):
                x1 = self.xseq.getValue(i*2)
                x2 = self.xseq.getValue(i*2+1)

                val = np.exp(-x2*tt)
                J.setValue(j, i*2, val)
                Jp.setValue(j, i*2, val)
                val = -x1*tt*np.exp(-x2*tt)
                J.setValue(j, i*2+1, val)
                Jp.setvalue(j, i*2+1, val)

    

