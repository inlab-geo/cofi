#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifndef _C_SOLVER_H_   /* Include guard */
#define _C_SOLVER_H_

EXPORT_SYMBOL void hello();

EXPORT_SYMBOL void solve(int m, int n, double **g, double *y, double *res);

#endif
