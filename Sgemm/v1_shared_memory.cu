#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define IDX2C(i, j, n) ((i) * (n) + (j))

void matrix_init(int m, int n, float* A) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[IDX2C(i, j, n)] = 1.0 * (float)drand48() - 1.0;
        }
    }
}

void cpu_sgemm(int M, int N, int K, float* A, float* B, float* C) {
    // for (int m = 0; m < M; ++m) {
    //     for (int n = 0; n < N; ++n) {
    //         float sum = 0.0f;
    //         for (int k = 0; k < K; ++k) { 
    //             sum += A[IDX2C(m, k, K)] * B[IDX2C(k, n, N)];
    //         }
    //         C[IDX2C(m, n, N)] = sum;
    //     }
    // }
    // Above is the naive way, extremely slow due to cache misses along the K dimension in B(column access, big stride)
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                C[IDX2C(m, n, N)] += A[IDX2C(m, k, K)] * B[IDX2C(k, n, N)];
            }
        }
    }
}

void check_results(float* C, float* C_ref, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabsf(C[IDX2C(i, j, n)] - C_ref[IDX2C(i, j, n)]) > 1e-3f) {
                printf("Mismatch at (%d, %d): %f vs %f\n", i, j, C[IDX2C(i, j, n)], C_ref[IDX2C(i, j, n)]);
                return;
            }
        }
    }
    printf("Results match!\n");
}

__global__ void cuda_sgemm(int M, int N, int K, float* A, float* B, float* C) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) {
        return;
    }
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        sum += A[IDX2C(m, k, K)] * B[IDX2C(k, n, N)];
    }
    C[IDX2C(m, n, N)] = sum;
}

int main() {
    int m = 4096;
    int n = 4096;
    int k = 512; // More like Q and K in attention

    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    float* A = (float*)calloc(size_A, 1);
    float* B = (float*)calloc(size_B, 1);

    float* C_cpu = (float*)malloc(size_C);
    float* C_gpu = (float*)malloc(size_C);

    matrix_init(m, k, A);
    matrix_init(k, n, B);

    float*d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    constexpr int BLOCK_X_SIZE = 32;
    constexpr int BLOCK_Y_SIZE = 8;
    dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    dim3 grid((n + BLOCK_X_SIZE - 1) / BLOCK_X_SIZE, (m + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE);
    
    cuda_sgemm<<<grid, block>>>(m, n, k, d_A, d_B, d_C);
    cpu_sgemm(m, n, k, A, B, C_cpu);

    cudaMemcpy(C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    check_results(C_gpu, C_cpu, m, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
}