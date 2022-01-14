cdef extern from "_c_solver_lib.h":
    void hello()
    void solve(int m, int n, double **g, double *y, double *res)
