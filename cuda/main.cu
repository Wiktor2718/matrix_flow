#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>

// config
#define SIZE 32
//#define NDEBUG
#define data_type float

typedef struct Matrix {
    data_type *data;
    size_t rows;
    size_t cols;
    bool row_major;
} Matrix;

#define element_wise_kernel_flat(name, operator)                                                        \
    extern "C" __global__ void name(data_type *res, data_type *a, data_type *b, size_t len) {           \
        int index = blockIdx.x * blockDim.x + threadIdx.x;                                              \
        if (index < len) res[index] = a[index] operator b[index];                                       \
    }

#define element_wise_kernel_t_right(name, operator)                                                     \
    extern "C" __global__ void name(data_type *res, data_type *a, data_type *b,                         \
    size_t rows, size_t cols, size_t len) {                                                             \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                \
        if (idx < len) {                                                                                \
            size_t index = (idx * cols) % (rows * cols) + idx / rows;                                   \
            assert(index < len);                                                                        \
            res[idx] = a[idx] operator b[index];                                                        \
        }                                                                                               \
    }

#define element_wise_kernel_t_left(name, operator)                                                     \
    extern "C" __global__ void name(data_type *res, data_type *a, data_type *b,                         \
    size_t rows, size_t cols, size_t len) {                                                             \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                \
        if (idx < len) {                                                                                \
            size_t index = (idx * cols) % (rows * cols) + idx / rows;                                   \
            assert(index < len);                                                                        \
            res[idx] = a[index] operator b[idx];                                                        \
        }                                                                                               \
    }

#define element_wise_operation(name, flat, transpose)                                                   \
    extern "C" void name(Matrix res, Matrix a, Matrix b, size_t len) {                                  \
        int block_size = SIZE*SIZE;                                                                     \
        int grid_size = (len + block_size - 1) / block_size;                                            \
        if (a.row_major == b.row_major) {                                                               \
            flat<<<grid_size, block_size>>>(res.data, a.data, b.data, len);                             \
        } else {                                                                                        \
            transpose<<<grid_size, block_size>>>(res.data, a.data, b.data, b.rows, b.cols, len);        \
        }                                                                                               \
    }

element_wise_kernel_flat(_matrix_add_flat, +)
element_wise_kernel_flat(_matrix_sub_flat, -)
element_wise_kernel_flat(_matrix_mul_flat, *)
element_wise_kernel_flat(_matrix_div_flat, /)

element_wise_kernel_t_right(_matrix_add_t_right, +)
element_wise_kernel_t_right(_matrix_sub_t_right, -)
element_wise_kernel_t_right(_matrix_mul_t_right, *)
element_wise_kernel_t_right(_matrix_div_t_right, /)

element_wise_kernel_t_left(_matrix_sub_t_left, -)
element_wise_kernel_t_left(_matrix_div_t_left, /)

element_wise_operation(matrix_add_t_right, _matrix_add_flat, _matrix_add_t_right)
element_wise_operation(matrix_sub_t_right, _matrix_sub_flat, _matrix_sub_t_right)
element_wise_operation(matrix_mul_t_right, _matrix_mul_flat, _matrix_mul_t_right)
element_wise_operation(matrix_div_t_right, _matrix_div_flat, _matrix_div_t_right)

element_wise_operation(matrix_sub_t_left, _matrix_sub_flat, _matrix_sub_t_left)
element_wise_operation(matrix_div_t_left, _matrix_div_flat, _matrix_div_t_left)

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