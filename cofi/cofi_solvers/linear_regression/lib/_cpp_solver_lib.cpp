#include <iostream>
#include <vector>
#include <math.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


// ----------------
// Regular C++ code
// ----------------

void hello() {
    std::cout << "Hello, world!" << std::endl;
}

static void display(int row, int col, std::vector< std::vector<double> > mat) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            std::cout << mat[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static std::vector< std::vector<double> > get_cofactor(int n, int p, int q, std::vector< std::vector<double> > mat) {
    std::vector< std::vector<double> > res(n-1, std::vector<double>(n-1, 0));
    int i = 0, j = 0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row != p && col != q) {
                res[i][j++] = mat[row][col];
                if (j == n-1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
    return res;
}

static double get_determinant(int n, std::vector< std::vector<double> > mat) {
    if (n == 1) return mat[0][0];
    
    double det = 0;
    std::vector< std::vector<double> > tmp(n, std::vector<double>(n, 0));    // tmp: nxn
    int sign = 1;
    for (int f = 0; f < n; f++) {
        tmp = get_cofactor(n, 0, f, mat);;
        det += sign * mat[0][f] * get_determinant(n-1, tmp);
        sign *= -1;
    }
    return det;
}

static std::vector< std::vector<double> > inverse(int n, std::vector< std::vector<double> > mat) {
    std::vector< std::vector<double> > mat_inv(n, std::vector<double>(n, 0));

    double det = get_determinant(n, mat);

    std::vector< std::vector<double> > tmp(n, std::vector<double>(n, 0)); // tmp: nxn
    std::vector< std::vector<double> > fac(n, std::vector<double>(n, 0)); // fac: nxn
    int p, q, a, b, i, j;
    for (q = 0; q < n; q++) {
        for (p = 0; p < n; p++) {
            a = 0;
            b = 0;
            tmp = get_cofactor(n, q, p, mat);
            fac[q][p] = pow(-1, q + p) * get_determinant(n-1, tmp);
        }
    }

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            mat_inv[i][j] = fac[j][i] / det;

    return mat_inv;
}

// a: mxn, b: mxk, res: nxk
std::vector< std::vector<double> > matrix_mult_transA(int m, int n, int k, std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) {
    std::vector< std::vector<double> > res(n, std::vector<double>(k, 0));
    int i, j, p;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            res[i][j] = 0;
            for (p = 0; p < m; p++)
                res[i][j] += a[p][i] * b[p][j];
        }
    }
    return res;
}

// a: mxn, b: kxn, res: mxk
std::vector< std::vector<double> > matrix_mult_transB(int m, int n, int k, std::vector< std::vector<double> > a, std::vector< std::vector<double> > b) {
    std::vector< std::vector<double> > res(m, std::vector<double>(k, 0));
    int i, j, p;
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            res[i][j] = 0;
            for (p = 0; p < n; p++)
                res[i][j] += a[i][p] * b[j][p];
        }
    }
    return res;
}

// a: mxn, b: n, res: m
std::vector<double> matrix_mult_vect(int m, int n, std::vector< std::vector<double> > a, std::vector<double> b) {
    std::vector<double> res(m, 0);
    int i, j, p;
    for (i = 0; i < m; i++) {
        res[i] = 0;
        for (j = 0; j < n; j++) {
            res[i] += a[i][j] * b[j];
        }
    }
    return res;
}

// g: nxm, y: n, res: m
std::vector<double> solve(int m, int n, std::vector< std::vector<double> > g, std::vector<double> y) {
    // gtg: mxm
    std::vector< std::vector<double> > gtg = matrix_mult_transA(n, m, m, g, g);

    // gtg_inv: mxm
    std::vector< std::vector<double> > gtg_inv = inverse(m, gtg);

    // gtg_inv_gt: mxn
    std::vector< std::vector<double> > gtg_inv_gt = matrix_mult_transB(m, m, n, gtg_inv, g);

    // double res[m];
    std::vector<double> res = matrix_mult_vect(m, n, gtg_inv_gt, y);

    return res;
}

int main() {
    int m = 2, n = 3;
    std::vector< std::vector<double> > g = {{0, 1}, {1, 0}, {1, 2}};
    std::vector<double> y = {6, 12, 6};
    std::vector<double> res = solve(m, n, g, y);
    printf(" %f\n %f\n", res[0], res[1]);
}


// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(_cpp_solver_lib, m) {
    m.doc() = "_cpp_solver_lib";
    m.def("hello", &hello, "Prints \"Hello, world!\"");
    m.def("solve", &solve, "Solves linear regression problem using normal equation");
}
