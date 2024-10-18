#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>
#include <curand_kernel.h>

// config
#define SIZE 8
#define BLOCK_SIZE 32
#define NDEBUG
#define value_type float

typedef struct Matrix {
    value_type *data;
    size_t rows;
    size_t cols;
    bool row_major;
} Matrix;

#define element_wise_kernel_flat(name, operator)                                                        \
    extern "C" __global__ void name(value_type *res, value_type *a, value_type *b, size_t len) {        \
        int index = blockIdx.x * blockDim.x + threadIdx.x;                                              \
        if (index < len) res[index] = a[index] operator b[index];                                       \
    }

#define element_wise_kernel_t_right(name, operator)                                                     \
    extern "C" __global__ void name(value_type *res, value_type *a, value_type *b,                      \
    size_t rows, size_t cols, size_t len) {                                                             \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                \
        if (idx < len) {                                                                                \
            size_t index = (idx * cols) % (rows * cols) + idx / rows;                                   \
            assert(index < len);                                                                        \
            res[idx] = a[idx] operator b[index];                                                        \
        }                                                                                               \
    }

#define element_wise_kernel_t_left(name, operator)                                                      \
    extern "C" __global__ void name(value_type *res, value_type *a, value_type *b,                      \
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

// matrix scalar operations
#define with_scalar_kernel(name, operator)                                                              \
    __global__ void name(value_type *res, value_type *a, value_type b, size_t len) {                    \
            int index = blockIdx.x * blockDim.x + threadIdx.x;                                          \
            if (index < len) res[index] = a[index] operator b;                                          \
    }

#define with_scalar_operation(name, kernel)                                                             \
    extern "C" void name(value_type *res, value_type *a, value_type b, size_t len) {                    \
        int block_size = SIZE*SIZE;                                                                     \
        int grid_size = (len + block_size - 1) / block_size;                                            \
        kernel<<<grid_size, block_size>>>(res, a, b, len);                                              \
    }

with_scalar_kernel(_matrix_scalar_add, +)
with_scalar_kernel(_matrix_scalar_sub, -)
with_scalar_kernel(_matrix_scalar_mul, *)
with_scalar_kernel(_matrix_scalar_div, /)

with_scalar_operation(matrix_scalar_add, _matrix_scalar_add)
with_scalar_operation(matrix_scalar_sub, _matrix_scalar_sub)
with_scalar_operation(matrix_scalar_mul, _matrix_scalar_mul)
with_scalar_operation(matrix_scalar_div, _matrix_scalar_div)

// scalar matrix operations
#define with_matrix_kernel(name, operator)                                                              \
    __global__ void name(value_type *res, value_type a, value_type *b, size_t len) {                    \
            int index = blockIdx.x * blockDim.x + threadIdx.x;                                          \
            if (index < len) res[index] = a operator b[index];                                          \
    }

#define with_matrix_operation(name, kernel)                                                             \
    extern "C" void name(value_type *res, value_type a, value_type *b, size_t len)  {                   \
        int block_size = SIZE*SIZE;                                                                     \
        int grid_size = (len + block_size - 1) / block_size;                                            \
        kernel<<<grid_size, block_size>>>(res, a, b, len);                                              \
    }

with_matrix_kernel(_scalar_matrix_sub, -)
with_matrix_kernel(_scalar_matrix_div, /)

with_matrix_operation(scalar_matrix_sub, _scalar_matrix_sub);
with_matrix_operation(scalar_matrix_div, _scalar_matrix_div);

// matrix operations
#define apply_kernel(name, transformation)                                                              \
    __global__ void name(value_type *res, value_type *matrix, size_t len) {                            \
        int index = blockIdx.x * blockDim.x + threadIdx.x;                                              \
        if (index < len) res[index] = transformation(matrix[index]);                                   \
    }

#define apply(name, kernel)                                                                             \
    extern "C" void name(value_type *res, value_type *matrix, size_t len) {                            \
        int block_size = SIZE*SIZE;                                                                     \
        int grid_size = (len + block_size - 1) / block_size;                                           \
        kernel<<<grid_size, block_size>>>(res, matrix, len);                                           \
    }

#define Sqware(x) ((x) * (x))
apply_kernel(_sqware, Sqware)
apply(sqware, _sqware)

#define Neg(x) -(x)
apply_kernel(_matrix_neg, Neg)
apply(matrix_neg, _matrix_neg)

// activations
#define LreLu(x) ( ((x) > 0) ? (x) : ((x) * 0.01) )
#define LreLuP(x) ( ((x) > 0) ? 1. : .01 )
#define Tanh(x) ( tanhf((x)) )
#define Sigmoid(x) ( 1.0f / (1.0f + expf(-(x))) )

apply_kernel(_leaky_relu, LreLu)
apply(leaky_relu, _leaky_relu)
apply_kernel(_leaky_relu_prime, LreLuP)
apply(leaky_relu_prime, _leaky_relu_prime)
apply_kernel(_matrix_tanh, Tanh)
apply(matrix_tanh, _matrix_tanh)
apply_kernel(_sigmoid, Sigmoid)
apply(sigmoid, _sigmoid)

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
    const value_type alpha = 1.0f;
    const value_type beta = 0.0f;

    // cuBLAS expects column-major matrices by default, so if row_major is true, we need to transpose the matrices.
    cublasOperation_t left_op = left.row_major ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t right_op = right.row_major ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(
        handle,
        left_op,                 // Transpose left if necessary
        right_op,                // Transpose right if necessary
        (int)left.rows,          // Number of rows in matrix A and C
        (int)right.cols,         // Number of columns in matrix B and C
        (int)left.cols,          // Number of columns in matrix A and rows in matrix B
        &alpha,                  // Scaling factor for the product
        left.data,               // Matrix A
        left_op == CUBLAS_OP_T ? (int)left.cols : (int)left.rows,  // Leading dimension of A
        right.data,              // Matrix B
        right_op == CUBLAS_OP_T ? (int)right.cols : (int)right.rows, // Leading dimension of B
        &beta,                   // Scaling factor for C
        res.data,                // Result matrix C
        (int)res.rows            // Leading dimension of C
    ); 

    // Clean up cuBLAS handle
    cublasDestroy(handle);
}

// random
__global__ void _init_curand_states(curandState *state, unsigned long seed, size_t len) { int idx = threadIdx.x + blockIdx.x * blockDim.x; if (idx < len) curand_init(seed, idx, 0, &state[idx]); }

__global__ void _normal(value_type *res, curandState *state, size_t len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) res[idx] = curand_normal(&state[idx]);
}

__global__ void _uniform(value_type *res, curandState *state, size_t len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) res[idx] = curand_uniform(&state[idx]);
}

extern "C" void normal(value_type *res, size_t len) {
    int block_size = SIZE*SIZE;
    int grid_size = (len + block_size - 1) / block_size;

    curandState *d_states;
    cudaMalloc(&d_states, len * sizeof(curandState));

    _init_curand_states<<<grid_size, block_size>>>(d_states, time(NULL), len);
    cudaDeviceSynchronize();

    _normal<<<grid_size, block_size>>>(res, d_states, len);
    cudaDeviceSynchronize();

    cudaFree(&d_states);
}

extern "C" void uniform(value_type *res, size_t len) {
    int block_size = SIZE*SIZE;
    int grid_size = (len + block_size - 1) / block_size;

    curandState *d_states;
    cudaMalloc(&d_states, len * sizeof(curandState));

    _init_curand_states<<<grid_size, block_size>>>(d_states, time(NULL), len);
    cudaDeviceSynchronize();

    _uniform<<<grid_size, block_size>>>(res, d_states, len);
    cudaDeviceSynchronize();

    cudaFree(&d_states);
}

// MLP
typedef struct {
    value_type *ptr;
    size_t rows;
    size_t cols;
} RM_Handle;

__global__ void _forward_mlp(float* X, float* W, float* B, float* Y, size_t N, size_t D, size_t M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float value = 0.0f;
        for (int k = 0; k < D; ++k) {
            value += X[row * D + k] * W[k * M + col];  // XW multiplication
        }
        value += B[col];  // Add bias
        Y[row * M + col] = value;
    }
}

extern "C" void forward_mlp(RM_Handle X, RM_Handle W, RM_Handle B, RM_Handle Y) {
    size_t N = X.rows;
    size_t D = W.rows;
    size_t M = W.cols;

    dim3 blockDim(SIZE, SIZE);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    _forward_mlp<<<gridDim, blockDim>>>(X.ptr, W.ptr, B.ptr, Y.ptr, N, D, M);
}

__global__ void _backward_mlp(float* X, float* W, float* dY, float* dW, float* dB, float* dX, size_t N, size_t D, size_t M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Gradient with respect to bias (dB)
    if (row == 0 && col < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += dY[i * M + col];
        }
        dB[col] = sum;
    }

    // Gradient with respect to weights (dW) = X^T * dY
    if (row < D && col < M) {
        float dw_val = 0.0f;
        for (int i = 0; i < N; ++i) {
            dw_val += X[i * D + row] * dY[i * M + col];
        }
        dW[row * M + col] = dw_val;
    }

    // Gradient with respect to input (dX) = dY * W^T
    if (row < N && col < D) {
        float dx_val = 0.0f;
        for (int k = 0; k < M; ++k) {
            dx_val += dY[row * M + k] * W[col * M + k];
        }
        dX[row * D + col] = dx_val;
    }
}

extern "C" void backward_mlp(RM_Handle X, RM_Handle W, RM_Handle dY, RM_Handle dW, RM_Handle dB, RM_Handle dX) {
    size_t N = X.rows;
    size_t D = W.rows;
    size_t M = W.cols;

    dim3 blockDim(SIZE, SIZE);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (D + blockDim.y - 1) / blockDim.y);
    _backward_mlp<<<gridDim, blockDim>>>(X.ptr, W.ptr, dY.ptr, dW.ptr, dB.ptr, dX.ptr, N, D, M);
}

__global__ void _update_params(float* W, float* dW, float* B, float* dB, float learning_rate, size_t D, size_t M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Update weights
    if (row < D && col < M) {
        W[row * M + col] -= learning_rate * dW[row * M + col];
    }

    // Update biases
    if (row == 0 && col < M) {
        B[col] -= learning_rate * dB[col];
    }
}

extern "C" void update_params(RM_Handle W, RM_Handle dW, RM_Handle B, RM_Handle dB, value_type learning_rate) {
    size_t D = W.rows;
    size_t M = W.cols;

    dim3 blockDim(SIZE, SIZE);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (D + blockDim.y - 1) / blockDim.y);
    _update_params<<<gridDim, blockDim>>>(W.ptr, dW.ptr, B.ptr, dB.ptr, learning_rate, D, M);
}
#define row_major_apply(name, kernel) \
    extern "C" void name(RM_Handle Y, RM_Handle X) { \
        int len = X.rows * X.cols; \
        int block_size = BLOCK_SIZE; \
        int grid_size = (len + block_size - 1) / block_size; \
        kernel<<<grid_size, block_size>>>(Y.ptr, X.ptr, len); \
    }

row_major_apply(forward_leaky_relu, _leaky_relu)
row_major_apply(forward_tanh, _matrix_tanh)
row_major_apply(forward_sigmoid, _sigmoid)

__global__ void _backward_leaky_relu(float* dY, float* X, float* dX, size_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) dX[i] = dY[i] * LreLuP(X[i]);
}

__global__ void _backward_tanh(float* dY, float* X, float* dX, size_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float tanh_val = Tanh(X[i]);
        dX[i] = dY[i] * (1.0f - tanh_val * tanh_val);
    }
}

__global__ void _backward_sigmoid(float* dY, float* X, float* dX, size_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float sigmoid_val = Sigmoid(X[i]);
        dX[i] = dY[i] * (sigmoid_val * (1.0f - sigmoid_val));
    }
}

#define activation_backward(name, kernel) \
    extern "C" void name(RM_Handle dY, RM_Handle X, RM_Handle dX) { \
        int len = X.rows * X.cols; \
        int block_size = BLOCK_SIZE; \
        int grid_size = (len + block_size - 1) / block_size; \
        kernel<<<grid_size, block_size>>>(dY.ptr, X.ptr, dX.ptr, len); \
    }

activation_backward(backward_leaky_relu, _backward_leaky_relu)
activation_backward(backward_tanh, _backward_tanh)
activation_backward(backward_sigmoid, _backward_sigmoid)