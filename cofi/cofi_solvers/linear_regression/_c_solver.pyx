# distutils: sources = lib/_c_solver_lib.c
# distutils: include_dirs = lib/

cimport _c_solver
from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import LeastSquareObjective, Model

# import ctypes
from libc.stdlib cimport malloc, free
import numpy as np
# cimport numpy as np
from warnings import warn

cdef hello_wrapper():
    hello()

def c_solve(g, y):
    cdef int n = g.shape[0]
    cdef int m = g.shape[1]
    print(n, m)
    cdef double **g_pt = <double **>malloc(n * sizeof(double *))
    cdef double *y_pt = <double *>malloc(n * sizeof(double))
    cdef double *res_pt = <double *>malloc(m * sizeof(double))
    if not g_pt or not y_pt or not res_pt:
        raise MemoryError
    
    cdef int i
    cdef int j
    for i in range(n):
        g_pt[i] = <double *>malloc(m * sizeof(double))
        for j in range(m):
            g_pt[i][j] = g[i,j]
        y_pt[j] = y[j]

    solve(m, n, g_pt, y_pt, res_pt)
    print("finished solving")
    res = np.zeros(m)
    for i in range(m):
        res[i] = res_pt[i]
    return res


"""
cdef c_solve(np.ndarray[np.double_t,ndim=2,mode='c']g, np.ndarray[np.double_t,ndim=1,mode='c']y, np.ndarray[np.double_t,ndim=1,mode='c']res):
    print(0)
    cdef int n = g.shape[0]
    cdef int m = g.shape[1]
    print(1)
    cdef np.ndarray[double, ndim=2, mode="c"] G_tmp = np.ascontiguousarray(g, dtype=ctypes.c_double)
    cdef double **G_pt = <double **>malloc(n * sizeof(double*))
    cdef np.ndarray[double, ndim=1, mode="c"] Y_tmp = np.ascontiguousarray(y, dtype=ctypes.c_double)
    cdef double *Y_pt = <double *>malloc(n * sizeof(double))
    cdef np.ndarray[double, ndim=1, mode="c"] res_tmp = np.ascontiguousarray(res, dtype=ctypes.c_double)
    cdef double *res_pt = <double *>malloc(m * sizeof(double))

    if not G_pt or not Y_pt or not res_pt:
        raise MemoryError
    try:
        Y_pt = &Y_tmp[0]
        res_pt = &res_tmp[0]
        for i in range(n):
            G_pt[i] = &G_tmp[i,0]
        # solve(m, n, <double **> &G_pt[0], <double *> &Y_pt[0], <double *> &res_pt[0])
        solve(m, n, <double **> G_pt, <double *> Y_pt, <double *> res_pt);
        # return np.array(res)
    finally:
        free(G_pt)
        free(Y_pt)
        free(res_pt)
"""

"""
def assignValues2D(self, np.ndarray[np.double_t,ndim=2,mode='c']mat):
    row_size,column_size = np.shape(mat)
    cdef np.ndarray[double, ndim=2, mode="c"] temp_mat = np.ascontiguousarray(mat, dtype = ctypes.c_double)
    cdef double ** mat_pointer = <double **>malloc(column_size * sizeof(double*))
    if not mat_pointer:
        raise MemoryError
    try:
        cdef int i
        for i in range(row_size):
            mat_pointer[i] = &temp_mat[i, 0]

        assign_values2D(<double **> &mat_pointer[0], row_size, column_size)
        return np.array(mat)
    finally:
        free(mat_pointer)
"""


class LRNormalEquationC(BaseSolver):
    def __init__(self, objective: LeastSquareObjective):
        self.objective = objective

    def solve(self) -> Model:
        warn(
            "You are using linear regression formula solver, please note that this is"
            " only for small scale of data"
        )

        G = self.objective.design_matrix()
        Y = self.objective.data_y()

        res = c_solve(G, Y)
        # print(res)

        # res = np.linalg.inv(G.T @ G) @ (G.T @ Y)
        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model
