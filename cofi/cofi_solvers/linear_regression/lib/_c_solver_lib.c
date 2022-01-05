#include "_c_solver_lib.h"

#include<stdlib.h>
#include<stdio.h>
#include<math.h>


void hello() {
    printf("Hello, world!\n");
}

static void display(int row, int col, double mat[row][col])
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			printf(" %f", mat[i][j]);
		printf("\n");
	}
    printf("\n");
}

static void get_cofactor(int n, int p, int q, double mat[n][n], double res[n][n]) {
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
}

static double get_determinant(int n, double mat[n][n]) {
    if (n == 1) return mat[0][0];
    
    double det = 0;
    double tmp[n][n];
    int sign = 1;
    for (int f = 0; f < n; f++) {
        get_cofactor(n, 0, f, mat, tmp);
        det += sign * mat[0][f] * get_determinant(n-1, tmp);
        sign *= -1;
    }
    return det;
}

static void inverse(int n, double mat[n][n], double mat_inv[n][n]) {
    double det = get_determinant(n, mat);

    double tmp[n][n], fac[n][n];
    int p, q, a, b, i, j;
    for (q = 0; q < n; q++) {
        for (p = 0; p < n; p++) {
            a = 0;
            b = 0;
            get_cofactor(n, q, p, mat, tmp);
            fac[q][p] = pow(-1, q + p) * get_determinant(n-1, tmp);
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mat_inv[i][j] = fac[j][i] / det;
        }
    }
}

// void matrix_mult(int m, int n, int k, double a[m][n], double b[n][k], double res[m][k]) {
//     int i, j, p;
//     for (i = 0; i < m; i++) {
//         for (j = 0; j < k; j++) {
//             res[i][j] = 0;
//             for (p = 0; p < n; p++)
//                 res[i][j] += a[i][p] * b[p][j];
//         }
//     }
// }

void matrix_mult_transA(int m, int n, int k, double a[m][n], double b[m][k], double res[n][k]) {
    int i, j, p;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            res[i][j] = 0;
            for (p = 0; p < m; p++)
                res[i][j] += a[p][i] * b[p][j];
        }
    }
}

void matrix_mult_transB(int m, int n, int k, double a[m][n], double b[k][n], double res[m][k]) {
    int i, j, p;
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            res[i][j] = 0;
            for (p = 0; p < n; p++)
                res[i][j] += a[i][p] * b[j][p];
        }
    }
}

void matrix_mult_vect(int m, int n, double a[m][n], double b[n], double res[m]) {
    int i, j, p;
    for (i = 0; i < m; i++) {
        res[i] = 0;
        for (j = 0; j < n; j++) {
            res[i] += a[i][j] * b[j];
        }
    }
}

// void solve(int m, int n, double g[n][m], double y[n], double res[m]) {
void solve(int m, int n, double **g_pt, double *y_pt, double *res_pt) {
    double g[n][m];
    double y[n];
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            g[i][j] = **(g_pt + i*m + j);
        }
        y[i] = *(y_pt + i);
    }

    double gtg[m][m];
    matrix_mult_transA(n, m, m, g, g, gtg);

    double gtg_inv[m][m];
    inverse(n, gtg, gtg_inv);

    // TODO: gtg_inv gt y
    // TODO: write a function to do matrix multipliccation!!!!
    double gtg_inv_gt[m][n];
    matrix_mult_transB(m, m, n, gtg_inv, g, gtg_inv_gt);

    double res[m];
    matrix_mult_vect(m, n, gtg_inv_gt, y, res);
    
    for (i = 0; i < m; i++)
        *(res_pt + i) = res[i];
}

int main() {
    printf("Hello, world!\n");

    // -> TEST INVERSE
    int n = 2;
    double a[2][2] = {{1.0,2.0}, {3.0,4.0}};
    double a_inv[n][n];
    inverse(n, a, a_inv);
    display(n,n,a_inv);
    
    // -> TEST MATRIX MULT TRANSPOSE A
    double g[2][3] = {{1.0,2.0,1.0}, {3.0,4.0,5.0}};
    double gtg[3][3];
    matrix_mult_transA(2,3,3,g,g,gtg);
    display(3,3,gtg);

    // -> TEST MATRIX MULT TRANSPOSE B
    double gt[3][2] = {{1.0,3.0}, {2.0,4.0}, {1.0,5.0}};
    double gtg2[3][3];
    matrix_mult_transB(3,2,3,gt,gt,gtg2);
    display(3,3,gtg2);

    // -> TEST MATRIX MULT VECTOR
    double vec[3] = {1.0, 2.0, 3.0};
    double matvec_res[2];
    matrix_mult_vect(2,3,g,vec,matvec_res);
    printf(" %f\n %f\n", matvec_res[0], matvec_res[1]);
}
