#include <cstdio>
#include <cuda_runtime.h>
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 256

bool check(float* out, float* res, int n) {
    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(out[i] - res[i]) > 1e-2) {
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
        for (int j = 0; j < THREADS_PER_BLOCK * 2; j++) {
            output[i] += input[i * THREADS_PER_BLOCK * 2 + j];
        }
    }
}

__global__ void reduce_shared_mem(float* input, float* output) {
    __shared__ float sdata[THREADS_PER_BLOCK];
    // kernel中是关于一个block的计算
    // 原本一个kernel中的blockDim.x = 256个thread，也就是handle 256个elements
    // 所以 blockstart = input + blockIdx.x * blockDim.x. 也就是找到全局input中此block的起始位置
    // 每两个block的举例为 blockDim.x
    // 但是现在block_num减小，每个block处理512个元素，介于blockstart表示每个block起始位置
    // 也代表两个响铃block的间隔空间，所以blockstart = input + blockIdx.x * blockDim.x * 2
    float *blockstart = input + blockIdx.x * blockDim.x * 2;
    sdata[threadIdx.x] = blockstart[threadIdx.x] + blockstart[threadIdx.x + blockDim.x]; // load two elements and add them during load
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    int block_num = (N + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK * 2); // each block reduces 2*THREADS_PER_BLOCK elements

    // reduce along all threads in each block, so we only need one output per block
    float *gpu_output = (float*)malloc(block_num * sizeof(float));
    // float *cpu_output = (float*)malloc(block_num * sizeof(float));
    float *cpu_output = (float*)calloc(block_num, sizeof(float)); // initialize to zero

    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    initialize_cpu_input(input, N);
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 注意相比前面的实现，block_num减半了，但是每个block中的thread数量不变
    // 之前一个thread对应input中的一个元素，现在一个thread对应input中的两个元素
    // 在进入kernel的时候我们会将原来两个thread对应的元素在加载到shared memory的时候就进行一次加法。
    // 这样即使处理的thread减半，但是每个thread现在含有的是两个input元素的和，所以最终的结果是一样的。
    // 这里我们只使用了一个维度--x轴，所以无需担心计算中2d的block和grid的索引问题。
    dim3 Grid(block_num, 1);
    dim3 Block(THREADS_PER_BLOCK, 1);

    cpu_reduce(input, cpu_output, block_num);
    reduce_shared_mem<<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(gpu_output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    check(cpu_output, gpu_output, block_num);

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(gpu_output);
    free(cpu_output);
}