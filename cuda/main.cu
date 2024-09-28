#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define data_type float

typedef struct Matrix {
    data_type *data;
    size_t rows;
    size_t cols;
    size_t len;
    bool row_major;
} Matrix;

// Perform matrix multiplication using cuBLAS
extern "C" void dot(Matrix res, Matrix left, Matrix right) {
    // Check for dimension mismatch
    if (left.cols != right.rows) {
        fprintf(stderr, "Matrix dimension mismatch: cannot multiply\n");
        exit(1);
    }

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set parameters for cuBLAS
    const data_type alpha = 1.0f;
    const data_type beta = 0.0f;

    // cuBLAS expects column-major matrices by default, so if row_major is true, we need to transpose the matrices.
    cublasOperation_t left_op = left.row_major ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t right_op = right.row_major ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(
        handle,
        left_op,          // Transpose left if necessary
        right_op,         // Transpose right if necessary
        (int)left.rows,   // Number of rows in matrix A and C
        (int)right.cols,  // Number of columns in matrix B and C
        (int)left.cols,   // Number of columns in matrix A and rows in matrix B
        &alpha,           // Scaling factor for the product
        left.data,        // Matrix A
        (int)left.rows,   // Leading dimension of A
        right.data,       // Matrix B
        (int)right.rows,  // Leading dimension of B
        &beta,            // Scaling factor for C
        res.data,         // Result matrix C
        (int)res.rows     // Leading dimension of C
    );

    // Clean up cuBLAS handle
    cublasDestroy(handle);
}