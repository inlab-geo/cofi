#include "_c_solver_lib.h"

#include<stdlib.h>
#include<stdio.h>
#include<math.h>


void hello() {
    printf("Hello, world!\n");
}

static double **malloc_mat(int rows, int cols) {
    double **mat = malloc(rows * sizeof(*mat));
    for (int i = 0; i < rows; i++) 
        mat[i] = malloc(cols * sizeof(*(mat[i])));
    return mat;
}

static void free_mat(int rows, double **mat) {
    for (int i = 0; i < rows; i++)
        free(mat[i]);
    free(mat);
}

static void display(int row, int col, double **mat)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			printf(" %f", mat[i][j]);
		printf("\n");
	}
    printf("\n");
}

static void get_cofactor(int n, int p, int q, double **mat, double **res) {
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

static double get_determinant(int n, double **mat) {
    if (n == 1) return mat[0][0];
    
    double det = 0;
    double **tmp = malloc_mat(n, n);   // tmp: nxn
    int sign = 1;
    for (int f = 0; f < n; f++) {
        get_cofactor(n, 0, f, mat, tmp);
        det += sign * mat[0][f] * get_determinant(n-1, tmp);
        sign *= -1;
    }
    free_mat(n, tmp);
    return det;
}

static void inverse(int n, double **mat, double **mat_inv) {
    double det = get_determinant(n, mat);

    double **tmp = malloc_mat(n, n);    // tmp: nxn
    double **fac = malloc_mat(n, n);    // fac: nxn
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

    free_mat(n, tmp);
    free_mat(n, fac);
}

// a: mxn, b: mxk, res: nxk
void matrix_mult_transA(int m, int n, int k, double **a, double **b, double **res) {
    int i, j, p;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            res[i][j] = 0;
            for (p = 0; p < m; p++)
                res[i][j] += a[p][i] * b[p][j];
        }
    }
}

// a: mxn, b: kxn, res: mxk
void matrix_mult_transB(int m, int n, int k, double **a, double **b, double **res) {
    int i, j, p;
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            res[i][j] = 0;
            for (p = 0; p < n; p++)
                res[i][j] += a[i][p] * b[j][p];
        }
    }
}

// a: mxn, b: n, res: m
void matrix_mult_vect(int m, int n, double **a, double *b, double *res) {
    int i, j, p;
    for (i = 0; i < m; i++) {
        res[i] = 0;
        for (j = 0; j < n; j++) {
            res[i] += a[i][j] * b[j];
        }
    }
}

// g: nxm, y: n, res: m
void solve(int m, int n, double **g, double *y, double *res) {
    // gtg: mxm
    double **gtg = malloc_mat(m, m);
    matrix_mult_transA(n, m, m, g, g, gtg);

    // gtg_inv: mxm
    double **gtg_inv = malloc_mat(m, m);
    inverse(m, gtg, gtg_inv);

    // gtg_inv_gt: mxn
    double **gtg_inv_gt = malloc_mat(m, n);
    matrix_mult_transB(m, m, n, gtg_inv, g, gtg_inv_gt);

    // double res[m];
    matrix_mult_vect(m, n, gtg_inv_gt, y, res);

    free_mat(m, gtg);
    free_mat(m, gtg_inv);
    free_mat(m, gtg_inv_gt);
}


int main() {
    // printf("Hello, world!\n"); fflush(stdout);

    // // -> TEST INVERSE
    // int n = 2;
    // double a[2][2] = {{1.0,2.0}, {3.0,4.0}};
    // double a_inv[n][n];
    // inverse(n, a, a_inv);
    // display(n,n,a_inv);
    
    // // -> TEST MATRIX MULT TRANSPOSE A
    // double g[2][3] = {{1.0,2.0,1.0}, {3.0,4.0,5.0}};
    // double gtg[3][3];
    // matrix_mult_transA(2,3,3,g,g,gtg);
    // display(3,3,gtg);

    // // -> TEST MATRIX MULT TRANSPOSE B
    // double gt[3][2] = {{1.0,3.0}, {2.0,4.0}, {1.0,5.0}};
    // double gtg2[3][3];
    // matrix_mult_transB(3,2,3,gt,gt,gtg2);
    // display(3,3,gtg2);

    // // -> TEST MATRIX MULT VECTOR
    // double vec[3] = {1.0, 2.0, 3.0};
    // double matvec_res[2];
    // matrix_mult_vect(2,3,g,vec,matvec_res);
    // printf(" %f\n %f\n", matvec_res[0], matvec_res[1]);

    // -> TEST SOLVE
    int i, j;
    int m = 2, n = 3;
    double **g_pt = (double **) malloc(n * sizeof(double *));
    g_pt[0] = (double *) malloc(m * sizeof(double));
    g_pt[1] = (double *) malloc(m * sizeof(double));
    g_pt[2] = (double *) malloc(m * sizeof(double));
    g_pt[0][0] = 0;
    g_pt[0][1] = 1;
    g_pt[1][0] = 1;
    g_pt[1][1] = 0;
    g_pt[2][0] = 1;
    g_pt[2][1] = 2;
    double *y_pt = (double *) malloc(n * sizeof(double));
    y_pt[0] = 6;
    y_pt[1] = 12;
    y_pt[2] = 6;
    double *res_pt = (double *) malloc(m * sizeof(double));
    solve(m, n, g_pt, y_pt, res_pt);
    printf(" %f\n %f\n", res_pt[0], res_pt[1]);
}
