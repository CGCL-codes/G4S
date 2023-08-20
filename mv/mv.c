#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl_cblas.h>
#include <mkl.h>
void matrix_multiply_dsymv(double* A, double* B, double* C, int dim)
{
    double alpha = 1.0;
    cblas_dsymv(CblasColMajor, CblasUpper, dim, alpha, A, dim, B, 1, 0.0, C, 1);
}

void matrix_multiply_dtrmv(double* A, double* B, double* C, int dim)
{
    cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, dim, A, dim, B, 1);
}

void matrix_multiply_sspmv(double* A, double* B, double* C, int dim)
{
    double alpha = 1.0;
    cblas_dspmv(CblasColMajor, CblasUpper, dim, alpha, A, B, 1, 0.0, C, 1);
}

void matrix_multiply_dgemv(double* A, double* B, double* C, int dim)
{
    double alpha = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, dim, dim, alpha, A, dim, B, 1, 0.0, C, 1);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("请输入正确的文件名\n");
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("无法打开文件\n");
        return 1;
    }
    printf("从文件 %s 读取矩阵A:\n", argv[1]);

    int dim, nnz;
    // 跳过文件头部信息
    char line[256];
    while (fgets(line, sizeof(line), file) != NULL) {
        if (line[0] == '%') {
            continue;
        } else {
            break;
        }
    }
    fscanf(file, "%d %*d %d\n", &dim, &nnz);

    double *A = (double*)malloc(dim * dim * sizeof(double));
    double *B = (double*)malloc(dim * sizeof(double));
    double *C = (double*)malloc(dim * sizeof(double));

    for (int i = 0; i < nnz; i++) {
        int row, col;
        fscanf(file, "%d %d\n", &row, &col);
        A[(row - 1) * dim + col - 1] = rand();
    }

    for (int i = 0; i < dim; i++) {
        B[i] = 1.0;
    }

    clock_t start_time, end_time;
    double cpu_time_used;

    start_time = clock();
    matrix_multiply_dsymv(A, B, C, dim);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("matrix_multiply_dsymv 运行时间：%f毫秒\n", cpu_time_used * 1000);

    start_time = clock();
    matrix_multiply_dtrmv(A, B, C, dim);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("matrix_multiply_dtrmv 运行时间：%f毫秒\n", cpu_time_used * 1000);

    start_time = clock();
    matrix_multiply_sspmv(A, B, C, dim);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("matrix_multiply_sspmv 运行时间：%f毫秒\n", cpu_time_used * 1000);

    start_time = clock();
    matrix_multiply_dgemv(A, B, C, dim);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("matrix_multiply_dgemv 运行时间：%f毫秒\n", cpu_time_used * 1000);

    return 0;
}
