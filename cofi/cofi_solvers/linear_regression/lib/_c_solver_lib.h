#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifndef _C_SOLVER_H_   /* Include guard */
#define _C_SOLVER_H_

EXPORT_SYMBOL void hello();

void solve(int m, int n, double g[n][m], double y[n], double res[m]);

#endif
