#include <cstdio>
#include <cuda_runtime.h>
#include <stdlib.h> 
#include <cuda.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;

bool check(float* out, float* res, int n) {
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(out[i] - res[i]) > 1) {
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

void cpu_reduce(float* input, float* output, int blockNum, int threadsPerBlock) {
    for (int i = 0; i < blockNum; i++) {
        for (int j = 0; j < threadsPerBlock; j++) {
            output[i] += input[i * threadsPerBlock + j];
        }
    }
}

__device__ void warp_reduce(volatile float* sdata, int tid) {
    sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + 32];
    sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + 16];
    sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + 8];
    sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + 4];
    sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + 2];
    sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + 1];
}

template<unsigned int NUM_ELEMENTS_PER_BLOCK, unsigned int shared_mem_size>
__global__ void reduce_shared_mem(float* input, float* output) {
    
    float *blockstart = input + blockIdx.x * NUM_ELEMENTS_PER_BLOCK;
    float sum = 0.f;

    for (int i = 0; i < NUM_ELEMENTS_PER_BLOCK / THREADS_PER_BLOCK; i++) {
        sum += blockstart[threadIdx.x + i * THREADS_PER_BLOCK];
    }
    __syncthreads();

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    int laneId = threadIdx.x % THREADS_PER_WARP;
    int warpId = threadIdx.x / THREADS_PER_WARP;

    __shared__ float warp_level_sums[shared_mem_size]; // shared_mem_size should be block_size / warp_size

    if (laneId == 0) {
        warp_level_sums[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (laneId < (blockDim.x / THREADS_PER_WARP)) ? warp_level_sums[laneId] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);    
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

int main() {
    constexpr int N = 32 * 1024 * 1024;
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    constexpr int block_num = 1024;
    constexpr int num_elements_per_block = N / block_num;

    // reduce along all threads in each block, so we only need one output per block
    float *gpu_output = (float*)malloc(block_num * sizeof(float));
    // float *cpu_output = (float*)malloc(block_num * sizeof(float));
    float *cpu_output = (float*)calloc(block_num, sizeof(float)); // initialize to zero

    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    initialize_cpu_input(input, N);
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREADS_PER_BLOCK, 1);

    cpu_reduce(input, cpu_output, block_num, num_elements_per_block);
    reduce_shared_mem<num_elements_per_block, THREADS_PER_BLOCK / THREADS_PER_WARP><<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(gpu_output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    check(cpu_output, gpu_output, block_num);

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(gpu_output);
    free(cpu_output);
}