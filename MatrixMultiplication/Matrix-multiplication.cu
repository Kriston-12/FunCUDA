#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256 // Size of a thread block

// This will be a most fundamental matrix multiplication in cuda
// C = A @ B. This is a very raw/slow implementation that still has iterative things
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // This is the row index, Imagine it as the the y axis. -- Along y axis is along rows
    int col = blockIdx.x * blockDim.x + threadIdx.x; // This is the column 

    if (row < M && col < N) {
        float value = 0;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

// C = A @ B. @ means matrix multiplicaiton. 
void matrixMul(float* A, float* B, float* C, int M, int K, int N) {
    size_t sizeA = M * K * sizeof(float);  // A.shape = M * K
    size_t sizeB = K * N * sizeof(float);  // B.shape = K * N
    size_t sizeC = M * N * sizeof(float);  // C.shape = M * N

    // Allocate gpu memory
    float *deviceA, *deviceB, *deviceC;
    //cudaError_t cudaMalloc(void **devPtr, size_t size)
    cudaMalloc((void**)&deviceA, sizeA);
    cudaMalloc((void**)&deviceB, sizeB);
    cudaMalloc((void**)&deviceC, sizeC);
    

    // Copy memory from CPU to GPU. cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    cudaMemcpy(deviceA, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, sizeB, cudaMemcpyHostToDevice);

    // Define threads/block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Define blocks/grid. Make sure block.shape * grid.shape is exactly M * N -- One thread account for one pixel/element
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixMulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, M, K, N);

    // Copy data from GPU-deviceC back to CPU-C
    cudaMemcpy(C, deviceC, sizeC, cudaMemcpyDeviceToHost);

    // Deallocation 
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

}

// Generate a random matrix
void generateRandomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f; 
    }
}

int main() {

    const int M = 1024, K = 1024, N = 1024;
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    generateRandomMatrix(A, M, K);
    generateRandomMatrix(B, K, N);
    

    matrixMul(A, B, C, M, K, N);

    // This is just for testing, do not use this in practice
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    free(A);
    free(B);
    free(C);

    return 0;
}