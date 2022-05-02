from typing import Union, List
import traceback
import logging
import sys
import warnings

import numpy as np

try:
    import petsc4py
    from petsc4py import PETSc
except:
    warnings.warn("Please install petsc4py if you'd like to use its solvers")

from .. import Model, BaseObjective, BaseSolver, OptimiserMixin


# ref doc for TAO: https://petsc.org/release/docs/manual/tao/

# methods available in PETSc
# https://petsc.org/main/docs/manualpages/Tao/TaoSetType.html
_methods_need_objective = [
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
_methods_need_objective_mpi = [
    # "nm",
    # "lmvm",
    # "cg",
    # "blmvm"
]
_methods_need_residual = ["brgn"]
_methods_need_residual_mpi = ["brgn"]


class TAOSolver(BaseSolver, OptimiserMixin):
    """Optimiser wrapper of TAO

    Objective definition needs to implement the following functions:
    - misfit(model)
    - residual(model), optional depending on method
    - jacobian(model), optional depending on method
    - gradient(model), optional depending on method
    - hessian(model), optional depending on method
    - data_x()
    - data_y()
    - initial_model()
    - params_size()

    Methods can be classified into 2 sets that:
    1. require you to specify objective function, (optionally) with the gradient and hessian
    2. require you to specify residual vector, (optioanlly) with the jacobian

    More details on methods and functions to implement WIP... #TODO

    """

    def __init__(self, objective: BaseObjective, mpi=False):
        try:
            petsc4py.init(sys.argv)
        except:
            raise Exception("Please install petsc4py if you'd like to use its solvers")

        self.obj = objective
        self._use_mpi = mpi
        if mpi:
            self._comm = PETSc.COMM_WORLD
            self._rank = self._comm.Get_rank()
            self.tao_app_ctx = _TAOAppCtxMPI(objective, self._comm)
        else:
            self._comm = PETSc.COMM_SELF
            self.tao_app_ctx = _TAOAppCtx(objective)

        self.n_params = self.tao_app_ctx.nm
        self.n_points = self.tao_app_ctx.t.shape[0]

        # clear options
        OptDB = PETSc.Options()
        for option in OptDB.getAll().keys():
            OptDB.delValue(option)

    def solve(
        self, method: str = None, extra_options: Union[List[str], str] = None, verbose=0
    ) -> Model:
        if method is None:
            if (
                hasattr(self, "method") and self.method is not None
            ):  # method set by set_method()
                method = self.method
            else:  # default option
                method = "nm"

        if (
            not self._use_mpi
            and not (
                method in _methods_need_objective or method in _methods_need_residual
            )
        ) or (
            self._use_mpi
            and not (
                method in _methods_need_objective_mpi
                or method in _methods_need_residual_mpi
            )
        ):
            raise ValueError(
                f"method {method} is not a valid option or hasn't been supported yet"
            )

        if extra_options:
            self.set_options(extra_options)
        self.set_options(f"-tao_type {method}")

        self._pre_solve(method)
        user = self.tao_app_ctx

        if method in _methods_need_objective:
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
            try:
                tao.solve(self.x)
            except:
                if self._rank == 0:
                    logging.error(f"Something wrong in solving by method {method}")
                    logging.error(traceback.format_exc())

        elif method in _methods_need_residual:
            # create TAO solver
            tao = PETSc.TAO().create(self._comm)
            tao.setType(method)
            tao.setFromOptions()

            # solve the problem
            tao.setResidual(user.evaluateResidual, self.f)
            tao.setJacobianResidual(user.evaluateJacobian, self.J, self.Jp)

            tao.solve(self.x)

        self._comm.barrier()

        if verbose:
            if self._use_mpi:
                print(self._rank)
                if self._rank == 0:
                    print("------------------", method, "with MPI ------------------")
                    print("x hat", flush=True)
                self.x.view()
            else:
                print("------------------", method, "------------------")
                self.x.view()

        params = self.x.getArray()
        tao.destroy()

        model = Model(
            **dict(
                [("p" + str(index[0]), val) for (index, val) in np.ndenumerate(params)]
            )
        )
        return model

    def set_options(self, options: Union[List[str], str]):
        # access PETSc options database
        OptDB = PETSc.Options()
        if isinstance(options, list):
            for option in options:
                if not isinstance(option, str):
                    raise ValueError(
                        "options of TAOSolver needs to be of type `str` or `List[str]`"
                    )
                OptDB.insertString(option)
        elif isinstance(options, str):
            OptDB.insertString(options)
        else:
            raise ValueError(
                "options of TAOSolver needs to be of type `str` or `List[str]`"
            )

    def _pre_solve(self, method: str):
        if self._use_mpi:
            self._pre_solve_mpi(method)
        else:
            self._pre_solve_non_mpi(method)

    def _pre_solve_non_mpi(self, method: str):
        if method in _methods_need_objective:
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

        elif method in _methods_need_residual:
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
        if method in _methods_need_objective_mpi:
            # create solution model vector
            self.x = PETSc.Vec().create(self._comm)
            self.x.setSizes(self.n_params)
            self.x.setType("mpi")
            self.x.setFromOptions()
            self.x.setUp()
            n, m = self.x.getOwnershipRange()
            self.x.setValues(range(n, m), self.tao_app_ctx.x0[n:m])
            self.x.assemble()

            # create time, observation, prediction vectors and Hessian matrix as mpi types
            self.t = PETSc.Vec().create(self._comm)
            self.t.setSizes(self.n_points)
            self.t.setType("mpi")
            self.t.setFromOptions()
            n, m = self.t.getOwnershipRange()
            self.t.setValues(range(n, m), self.tao_app_ctx.t[n:m])
            self.t.assemble()

            self.y = PETSc.Vec().create(self._comm)
            self.y.setSizes(self.n_points)
            self.y.setType("mpi")
            self.y.setFromOptions()
            n, m = self.t.getOwnershipRange()
            self.y.setValues(range(n, m), self.tao_app_ctx.y[n:m])
            self.y.assemble()

            self.f = PETSc.Vec().create(self._comm)
            self.f.setSizes(self.n_points)
            self.f.setType("mpi")
            self.f.setFromOptions()

            self.H = PETSc.Mat().create(self._comm)
            self.H.setType("mpidense")
            self.H.setSizes([self.n_params, self.n_params])
            self.H.setFromOptions()
            self.H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self.H.setUp()

        elif method in _methods_need_residual_mpi:
            # create solution model vector
            self.x = PETSc.Vec().create(self._comm)
            self.x.setSizes(self.n_params)
            self.x.setType("mpi")
            self.x.setFromOptions()
            self.x.setUp()
            n, m = self.x.getOwnershipRange()
            self.x.setValues(range(n, m), self.tao_app_ctx.x0[n:m])
            self.x.assemble()

            # create time, observation, prediction vectors and Jacobian matrix as mpi types
            self.t = PETSc.Vec().create(self._comm)
            self.t.setSizes(self.n_points)
            self.t.setType("mpi")
            self.t.setFromOptions()
            n, m = self.t.getOwnershipRange()
            self.t.setValues(range(n, m), self.tao_app_ctx.t[n:m])
            self.t.assemble()

            self.y = PETSc.Vec().create(self._comm)
            self.y.setSizes(self.n_points)
            self.y.setType("mpi")
            self.y.setFromOptions()
            n, m = self.t.getOwnershipRange()
            self.y.setValues(range(n, m), self.tao_app_ctx.y[n:m])
            self.y.assemble()

            self.f = PETSc.Vec().create(self._comm)
            self.f.setSizes(self.n_points)
            self.f.setType("mpi")
            self.f.setFromOptions()

            self.J = PETSc.Mat().create(self._comm)
            self.J.setType("mpidense")
            self.J.setSizes([self.n_points, self.n_params])
            self.J.setFromOptions()
            self.J.setUp()

            self.Jp = PETSc.Mat().create(self._comm)
            self.Jp.setType("mpidense")
            self.Jp.setSizes([self.n_points, self.n_params])
            self.Jp.setFromOptions()
            self.Jp.setUp()

        self.tao_app_ctx.set_MPI_variables(t_mpitype=self.t, y_mpitype=self.y)


class _TAOAppCtx:
    """Application context struct in C, required in TAO.
    This class is initialised and converted from an objective.
    """

    def __init__(self, objective: BaseObjective):
        self._obj = objective
        if (
            objective.data_y is None
            or objective.data_x is None
            or objective.initial_model is None
        ):
            raise ValueError("Data x, y and initial model are required for TAO solver")

        self.y = objective.data_y()
        self.t = objective.data_x()
        self.x0 = objective.initial_model()
        self.nm = objective.params_size()

    def formObjective(self, tao, x):
        try:
            res = self._obj.misfit(x)
        except Exception as e:
            logging.error(traceback.format_exc())
            print("An error occurred while forming objective:", e)

        return res

    def formGradient(self, tao, x, G):
        if self._obj.gradient:
            try:
                grad = self._obj.gradient(x)
            except Exception as e:
                logging.error(traceback.format_exc())
                print("An error occurred while forming gradient:", e)

            G.setArray(grad)
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
            try:
                hess = self._obj.hessian(x)
            except Exception as e:
                logging.error(traceback.format_exc())
                print("An error occurred while forming Hessian:", e)

            H.setValues(range(0, self.nm), range(0, self.nm), hess)
            H.assemble()
        else:
            self.formHessian = None

    def evaluateJacobian(self, tao, x, J, Jp):
        if self._obj.jacobian:
            try:
                jac = self._obj.jacobian(x)
            except Exception as e:
                logging.error(traceback.format_exc())
                print("An error occurred while evaluating Jacobian:", e)

            J.setValues(range(0, self.t.shape[0]), range(0, self.nm), jac)
            J.assemble()
            Jp.setValues(range(0, self.t.shape[0]), range(0, self.nm), jac)
            Jp.assemble()
        else:
            self.evaludateJacobian = None

    def evaluateResidual(self, tao, x, f):
        if self._obj.residual:
            try:
                res = self._obj.residual(x)
            except Exception as e:
                logging.error(traceback.format_exc())
                print("An error occurred while evaluating residuals:", e)

            f.setArray(res)
            f.assemble()
        else:
            self.evaluateResidual = None


class _TAOAppCtxMPI(_TAOAppCtx):
    def __init__(self, objective: BaseObjective, comm):
        super().__init__(objective)
        self._comm = comm
        self._rank = comm.Get_rank()

    def set_MPI_variables(self, **kwargs):
        # self.t should be passed in here
        self.__dict__.update(kwargs)

    def formSequentialModelVector(self, x):
        self.xseq = PETSc.Vec().create(PETSc.COMM_SELF)
        self.xseq.setType("seq")
        try:
            scatter, self.xseq = PETSc.Scatter.toAll(x)
            scatter.begin(x, self.xseq)
            scatter.end(x, self.xseq)
        except:
            logging.error(traceback.format_exc())

    def formObjective(self, tao, x):
        # print("formObjective")
        if self._obj.misfit_mpi:
            try:
                self.formSequentialModelVector(x)
                n, m = self.t_mpitype.getOwnershipRange()
                xseq = self.xseq.getArray()
                res = self._obj.misfit_mpi(xseq, n, m)
                return res
            except Exception as e:
                logging.error(traceback.format_exc())
                print(
                    f"An error occurred while forming objective: {e} on process"
                    f" #{self._rank}"
                )
        else:
            self.formObjective = None

    def formGradient(self, tao, x, G):
        # print("formGradient")
        if self._obj.gradient_mpi:
            try:
                self.formSequentialModelVector(x)
                n, m = self.t_mpitype.getOwnershipRange()
                xseq = self.xseq.getArray()
                grad = self._obj.gradient_mpi(xseq, n, m)
                G.setValues(range(0, G.getSize()), grad)
                G.assemble()
            except Exception as e:
                logging.error(traceback.format_exc())
                print(
                    f"An error occurred while forming gradient: {e} on process"
                    f" #{self._rank}"
                )
        else:
            self.formGradient = None

    def formObjGrad(self, tao, x, G):
        # print("formObjGrad")
        try:
            return super().formObjGrad(tao, x, G)
        except:
            logging.error(traceback.format_exc())

    def formHessian(self, tao, x, H, HP):
        # print("formHessian")
        if self._obj.hessian_mpi:
            try:
                self.formSequentialModelVector(x)
                n, m = self.t_mpitype.getOwnershipRange()
                xseq = self.xseq.getArray()
                hess = self._obj.hessian_mpi(xseq, n, m)
                H.setValues(range(0, self.nm), range(0, self.nm), hess)
                H.assemble()
            except Exception as e:
                logging.error(traceback.format_exc())
                print(
                    f"An error occurred while forming Hessian: {e} on process"
                    f" #{self._rank}"
                )
        else:
            self.formHessian = None

    def evaluateJacobian(self, tao, x, J, Jp):
        if self._obj.jacobian_mpi:
            try:
                self.formSequentialModelVector(x)
                n, m = self.t_mpitype.getOwnershipRange()
                xseq = self.xseq.getArray()
                jac = self._obj.jacobian_mpi(xseq, n, m)
                J.setValues(range(n, m), range(0, self.nm), jac)
                J.assemble()
                Jp.setValues(range(n, m), range(0, self.nm), jac)
                Jp.assemble()
            except Exception as e:
                logging.error(traceback.format_exc())
                print(
                    f"An error occurred while evaluating Jacobian: {e} on process"
                    f" #{self._rank}"
                )
        else:
            self.evaluateJacobian = None

    def evaluateResidual(self, tao, x, f):
        if self._obj.residual_mpi:
            try:
                self.formSequentialModelVector(x)
                n, m = self.t_mpitype.getOwnershipRange()
                xseq = self.xseq.getArray()
                res = self._obj.residual_mpi(xseq, n, m)
                f.setValues(range(n, m), res)
                f.assemble()
            except Exception as e:
                logging.error(traceback.format_exc())
                print(
                    f"An error occurred while evaluating residuals: {e} on process"
                    f" #{self._rank}"
                )
        else:
            self.evaluateResidual = None
