#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifndef _C_SOLVER_H_   /* Include guard */
#define _C_SOLVER_H_

EXPORT_SYMBOL void hello();

// void solve(float g[][], float y[], float *res[]);

#endif // FOO_H_
