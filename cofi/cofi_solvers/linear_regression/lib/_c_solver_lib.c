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
            printf("fac[q][p] <- sign * determinant=%f\n", get_determinant(n-1, tmp));
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mat_inv[i][j] = fac[j][i] / det;
        }
    }
}

void solve(int m, int n, double g[n][m], double y[n], double res[m]) {
    double gtg[m][m];
    int i, j, k;
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
            for (k = 0; k < n; k++)
                gtg[i][j] += g[i][k] * g[j][k];

    double gtg_inv[m][m];
    inverse(n, gtg, gtg_inv);

    // TODO: gtg_inv gt y
    // TODO: write a function to do matrix multipliccation!!!!
    double gtg_inv_gt[m][n];
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < m; k++) {
                gtg_inv_gt[i][j] += gtg_inv[i][k] * g[j][k];
            }
        }
    }
}

int main() {
    printf("Hello, world!\n");

    int n = 2;
    double a[2][2] = {{1.0,2.0}, {3.0,4.0}};
    double a_inv[n][n];
    inverse(n, a, a_inv);
}

