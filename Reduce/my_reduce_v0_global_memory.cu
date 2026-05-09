#include <cstdio>
#include <cuda_runtime.h>
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 256

bool check(float* out, float* res, int n) {
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (abs(out[i] - res[i]) > 1e-5) {
            printf("Error at %d: %f != %f\n", i, out[i], res[i]);
            flag = false;
        }
    }
    return flag;
}

void initialize_cpu_input(float* input, int n) {
    for (int i = 0; i < n; i++) {
        input[i] = rand() / (float)RAND_MAX;
    }
}

void cpu_reduce(float* input, float* output, int blockNum) {
    for (int i = 0; i < blockNum; i++) {
        for (int j = 0; j < THREADS_PER_BLOCK; j++) {
            output[i] += input[i * THREADS_PER_BLOCK + j];
        }
    }
}

__global__ void reduce0(float* input, float* output) {
    // int thread_idx = threadIdx.x;
    // int global_idx = blockIdx.x * blockDim.x + thread_idx.x;

}

int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    int block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // reduce along all threads in each block, so we only need one output per block
    float *gpu_output = (float*)malloc(block_num * sizeof(float));
    float *cpu_output = (float*)malloc(block_num * sizeof(float));


    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    initialize_cpu_input(input, N);
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREADS_PER_BLOCK, 1);

    cpu_reduce(input, cpu_output, block_num);
    reduce0<<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(gpu_output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    check(cpu_output, gpu_output, block_num);

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(gpu_output);
    free(cpu_output);
}