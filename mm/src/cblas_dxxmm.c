#include <mkl.h>
#include <stdio.h>
#include <time.h>
int num_entries;

void generate_sparse_matrix_indices(double* val, MKL_INT n, MKL_INT nz, MKL_INT* rowind, MKL_INT* colind) {
    // 初始化每一列的非零元素计数器
    MKL_INT col_counts[n];
    for (MKL_INT i = 0; i < n; i++) {
        col_counts[i] = 0;
    }

    // 遍历非零元素，填充行索引和列索引数组
    MKL_INT nnz = 0;  // 当前已经处理的非零元素数量
    for (MKL_INT i = 0; i < n; i++) {
        for (MKL_INT j = 0; j < n; j++) {
            if (val[i * n + j] != 0.0) {
                rowind[nnz] = i;
                colind[nnz] = j;
                col_counts[j]++;
                nnz++;
            }
        }
    }

    // 根据每一列的非零元素数量，对列索引进行偏移
    MKL_INT offset = 0;
    for (MKL_INT i = 0; i < n; i++) {
        MKL_INT count = col_counts[i];
        col_counts[i] = offset;
        offset += count;
    }

    // 对行索引和列索引进行排序，以确保按照列的顺序存储非零元素
    for (MKL_INT i = 0; i < nnz; i++) {
        MKL_INT col = colind[i];
        MKL_INT count = col_counts[col];
        while (count < i) {
            MKL_INT temp_row = rowind[count];
            rowind[count] = rowind[i];
            rowind[i] = temp_row;

            MKL_INT temp_col = colind[count];
            colind[count] = colind[i];
            colind[i] = temp_col;

            count = col_counts[col];
        }
        col_counts[col]++;
    }
}

clock_t start_time, end_time;
double cpu_time_used;

struct timeval t1, t2;
void matrix_multiply_dsymm(double* A, double* B, double* C, int dim)
{
    double alpha = 1.0;
    double beta = 0.0;
    char side = 'L';
    char uplo = 'U';
    int num = 10;

    int times = 10;
    int num_ = times;
    double res__=0.0;
    while(times--){
        gettimeofday(&t1, NULL);
        cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, dim, dim, alpha, A, dim, B, dim, beta, C, dim);
        gettimeofday(&t2, NULL);
        res__ += ((double)t2.tv_sec)*1000000+t2.tv_usec - ((double)t1.tv_sec)*1000000-t1.tv_usec;
    }
    printf("cblas_dsymm 运行时间：%f 毫秒\n", res__/1000/num_);
    
}

void matrix_multiply_dtrmm(double* A, double* B, double* C, int dim)
{
    double alpha = 1.0;
    char side = 'L';
    char uplo = 'U';
    char transa = 'N';
    char diag = 'N';
    int times = 10;
    int num_ = times;
    double res__=0.0;
    while(times--){
        gettimeofday(&t1, NULL);
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, dim, dim, alpha, A, dim, B, dim);
        gettimeofday(&t2, NULL);
        res__ += ((double)t2.tv_sec)*1000000+t2.tv_usec - ((double)t1.tv_sec)*1000000-t1.tv_usec;
    }
    printf("cblas_dtrmm 运行时间：%f 毫秒\n", res__/1000/num_);
    
}
void matrix_multiply_dgemm(double* A, double* B, double* C, int dim)
{
    double alpha = 1.0;
    double beta = 0.0;
    int times = 10;
    int num_ = times;
    double res__=0.0;
    while(times--){
        gettimeofday(&t1, NULL);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, A, dim, B, dim, beta, C, dim);
        gettimeofday(&t2, NULL);
        res__ += ((double)t2.tv_sec)*1000000+t2.tv_usec - ((double)t1.tv_sec)*1000000-t1.tv_usec;
    }
    printf("cblas_dgemm 运行时间：%f 毫秒\n", res__/1000/num_);
}
void matrix_multiply_spmm(double* valA, double* valB, double* valC, int dim)
{
    char transa = 'N';
    // 计算需要的数组大小并分配内存
    MKL_INT rowindA[num_entries];
    MKL_INT colindA[num_entries];
    // 生成稀疏矩阵的行索引和列索引数组
    generate_sparse_matrix_indices(valA, dim, num_entries, rowindA, colindA);

    MKL_INT rowindB[num_entries];
    MKL_INT colindB[num_entries];
    // 生成稀疏矩阵的行索引和列索引数组
    generate_sparse_matrix_indices(valB, dim, num_entries, rowindB, colindB);
    // 创建稀疏矩阵A的描述符
    sparse_matrix_t A, B, C;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, dim, dim, rowindA, rowindA + 1, colindA, valA);
    mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, dim, dim, rowindB, rowindB + 1, colindB, valB);
    // 设置输出矩阵 C 的存储格式为 CSR
    struct matrix_descr C_desc;
    C_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    // 计算矩阵乘法 C = A * B
    start_time = clock();
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, &C);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("mkl_sparse_spmm 运行时间：%f 毫秒\n", cpu_time_used* 1000);
}
int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("请在运行指令中指定文件名作为参数。\n");
        return -1;
    }
    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("无法打开文件 %s。\n", argv[1]);
        return -1;
    }
    printf("从文件 %s 读取矩阵 A:\n", argv[1]);
    // 跳过文件头部信息
    char line[256];
    while (fgets(line, sizeof(line), file) != NULL) {
        if (line[0] == '%') {
            continue;
        } else {
            break;
        }
    }
    
    int dim;
    if (sscanf(line, "%d%d%d", &dim, &dim, &num_entries) != 3) {
        printf("dim dim num_entries 读取文件失败。\n");
        fclose(file);
        return -1;
    }
    double* A = (double*)malloc(dim * dim * sizeof(double));
    double* B = (double*)malloc(dim * dim * sizeof(double));
    
    // 读取文件数据到矩阵 A 和 B
    for (int i = 0; i < num_entries; i++) {
        if (fgets(line, sizeof(line), file) == NULL) {
            printf("读取文件失败。\n");
            fclose(file);
            free(A);
            free(B);
            return -1;
        }
        double value;
        int row, col;
        sscanf(line, "%d%d%lf", &row, &col, &value);
        A[(row-1) * dim + (col-1)] = value;  // A 的值写入对应位置
    }
    for( int i=1;i<=dim;i++ ){
        for( int j=1;j<=dim;j++){
            B[(i-1) * dim + (j-1)] = rand();
        }
    }
    printf("B complete\n");
    fclose(file);
    
    double C[dim * dim];
    double C_dtrmm[dim * dim];
    
    // 调用矩阵乘法函数
    matrix_multiply_dsymm(A, B, C, dim);


    matrix_multiply_dtrmm(A, B, C_dtrmm, dim);

    matrix_multiply_dgemm(A, B, C_dtrmm, dim);
    
    return 0;
}
