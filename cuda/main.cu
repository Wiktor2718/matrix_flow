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

// PartailEq
#define cmp_kernel_flat(name, type) \
    __global__ void name(type *a, type *b, int *res, size_t len) { \
        int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        if (idx < len && a[idx] != b[idx]) atomicMin(res, 0); \
    }

#define cmp_kernel_t(name, type) \
    __global__ void name(type *a, type *b, int *res, size_t rows, size_t cols, size_t len) { \
        int idx = blockIdx.x * blockDim.x + threadIdx.x; \
        size_t index = (idx * cols) % (rows * cols) + idx / rows; \
        if (idx < len && a[idx] != b[index]) atomicMin(res, 0); \
    }

#define comparison_flat(name, kernel, type) \
    extern "C" int name(type a, type b, size_t len) { \
        int *res_device; \
        cudaMalloc((void**)&res_device, sizeof(int)); \
        int res_host = 1; \
        cudaMemcpy(res_device, &res_host, sizeof(int), cudaMemcpyHostToDevice); \
        \
        int threadsPerBlock = 256; \
        int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock; \
        kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, res_device, len); \
        cudaDeviceSynchronize(); \
        \
        cudaMemcpy(&res_host, res_device, sizeof(int), cudaMemcpyDeviceToHost); \
        cudaFree(res_device); \
        \
        return res_host; \
    }

cmp_kernel_flat(_matrix_value_type_cmp_flat, value_type)
cmp_kernel_t(_matrix_value_type_cmp_t, value_type)

extern "C" int matrix_value_type_cmp(Matrix a, Matrix b, size_t len) {
    int *res_device;
    cudaMalloc((void**)&res_device, sizeof(int));
    int res_host = 1;
    cudaMemcpy(res_device, &res_host, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    if (a.row_major == b.row_major) {
        _matrix_value_type_cmp_flat<<<blocksPerGrid, threadsPerBlock>>>(a.data, b.data, res_device, len);
    } else {
        _matrix_value_type_cmp_t<<<blocksPerGrid, threadsPerBlock>>>(a.data, b.data, res_device, a.rows, a.cols, len);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(&res_host, res_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(res_device);

    return res_host;
}

// vectors
cmp_kernel_flat(_vec_usize_cmp, size_t)
cmp_kernel_flat(_vec_i8_cmp,    int8_t)
cmp_kernel_flat(_vec_i16_cmp,   int16_t)
cmp_kernel_flat(_vec_i32_cmp,   int32_t)
cmp_kernel_flat(_vec_i64_cmp,   int64_t)

cmp_kernel_flat(_vec_u8_cmp,    uint8_t)
cmp_kernel_flat(_vec_u16_cmp,   uint16_t)
cmp_kernel_flat(_vec_u32_cmp,   uint32_t)
cmp_kernel_flat(_vec_u64_cmp,   uint64_t)

cmp_kernel_flat(_vec_f32_cmp,   float)
cmp_kernel_flat(_vec_f64_cmp,   double)

comparison_flat(vec_usize_cmp, _vec_usize_cmp, size_t*)
comparison_flat(vec_i8_cmp,    _vec_i8_cmp,    int8_t*)
comparison_flat(vec_i16_cmp,   _vec_i16_cmp,   int16_t*)
comparison_flat(vec_i32_cmp,   _vec_i32_cmp,   int32_t*)
comparison_flat(vec_i64_cmp,   _vec_i64_cmp,   int64_t*)
comparison_flat(vec_u8_cmp,    _vec_u8_cmp,    uint8_t*)
comparison_flat(vec_u16_cmp,   _vec_u16_cmp,   uint16_t*)
comparison_flat(vec_u32_cmp,   _vec_u32_cmp,   uint32_t*)
comparison_flat(vec_u64_cmp,   _vec_u64_cmp,   uint64_t*)
comparison_flat(vec_f32_cmp,   _vec_f32_cmp,   float*)
comparison_flat(vec_f64_cmp,   _vec_f64_cmp,   double*)

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

// optimizers
__global__ void _adam_update_params(
    float* W, float* dW, float* B, float* dB, 
    float* mW, float* vW, float* mB, float* vB,
    float beta1, float beta2, float beta1_t, float beta2_t, 
    float learning_rate, float epsilon, 
    size_t D, size_t M) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Update weights
    if (row < D && col < M) {
        int index = row * M + col;

        // Adam update for weights
        mW[index] = beta1 * mW[index] + (1.0f - beta1) * dW[index];
        vW[index] = beta2 * vW[index] + (1.0f - beta2) * (dW[index] * dW[index]);

        // Compute bias-corrected moments
        float mW_hat = mW[index] / (1.0f - beta1_t);
        float vW_hat = vW[index] / (1.0f - beta2_t);

        // Update weights
        W[index] -= learning_rate * mW_hat / (sqrtf(vW_hat) + epsilon);
    }

    // Update biases
    if (row == 0 && col < M) {
        // Adam update for biases
        mB[col] = beta1 * mB[col] + (1.0f - beta1) * dB[col];
        vB[col] = beta2 * vB[col] + (1.0f - beta2) * (dB[col] * dB[col]);

        // Compute bias-corrected moments
        float mB_hat = mB[col] / (1.0f - beta1_t);
        float vB_hat = vB[col] / (1.0f - beta2_t);

        // Update biases
        B[col] -= learning_rate * mB_hat / (sqrtf(vB_hat) + epsilon);
    }
}

extern "C" void adam_update_params(
    RM_Handle W, RM_Handle dW, RM_Handle B, RM_Handle dB, 
    RM_Handle mW, RM_Handle vW, RM_Handle mB, RM_Handle vB, 
    value_type beta1, value_type beta2, 
    value_type beta1_t, value_type beta2_t, 
    value_type learning_rate, value_type epsilon) 
{
    size_t D = W.rows;
    size_t M = W.cols;

    dim3 blockDim(SIZE, SIZE);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (D + blockDim.y - 1) / blockDim.y);
    
    _adam_update_params<<<gridDim, blockDim>>>(
        W.ptr, dW.ptr, B.ptr, dB.ptr, 
        mW.ptr, vW.ptr, mB.ptr, vB.ptr, 
        beta1, beta2, beta1_t, beta2_t, 
        learning_rate, epsilon, D, M);
}

// Losses
__global__ void _mse_prime(value_type *grad, value_type *y_true, value_type *y_pred, value_type scale, size_t len) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        value_type t = y_pred[i] - y_true[i];
        grad[i] = t * scale;
    }
}

extern "C" void mse_prime_launcher(RM_Handle grad, RM_Handle y_true, RM_Handle y_pred) {
    size_t len = grad.rows * grad.cols;
    int block_size = BLOCK_SIZE;
    int grid_size = (len + block_size - 1) / block_size;

    value_type scale = 2.0f / len;

    _mse_prime<<<grid_size, block_size>>>(grad.ptr, y_true.ptr, y_pred.ptr, scale, len);
}

#include <stdio.h>
#include <cuda.h>

// CUDA kernel to compute Mean Squared Error
__global__ void mse_kernel(value_type *mse, const value_type *y_true, const value_type *y_pred, size_t len) {
    extern __shared__ float shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes its squared error
    float error = 0.0f;
    if (idx < len) {
        float diff = y_pred[idx] - y_true[idx];
        error = diff * diff;
    }

    // Store the error in shared memory
    shared_data[tid] = error;
    __syncthreads();

    // Reduce the error within each block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Accumulate the block results in the first thread of each block
    if (tid == 0) {
        atomicAdd(mse, shared_data[0]);
    }
}

extern "C" value_type compute_mse(RM_Handle y_true, RM_Handle y_pred) {
    size_t len = y_pred.rows * y_pred.cols;

    value_type *d_mse;
    value_type mse = 0.0f;

    cudaMalloc((void**)&d_mse, sizeof(value_type));
    cudaMemcpy(d_mse, &mse, sizeof(value_type), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (len + blockSize - 1) / blockSize;
    mse_kernel<<<gridSize, blockSize, blockSize * sizeof(value_type)>>>(d_mse, y_true.ptr, y_pred.ptr, len);

    cudaMemcpy(&mse, d_mse, sizeof(value_type), cudaMemcpyDeviceToHost);

    mse /= len;

    cudaFree(d_mse);

    return mse;
}
