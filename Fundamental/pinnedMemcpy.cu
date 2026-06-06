#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void sumArraysGPU(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int nElem = 1 << 20;
    size_t nBytes = nElem * sizeof(float);

    float *a_h, *b_h, *c_h;
    cudaMallocHost((void **)&a_h, nBytes);
    cudaMallocHost((void **)&b_h, nBytes);
    cudaMallocHost((void **)&c_h, nBytes);

    float *a_d, *b_d, *c_d;
    cudaMalloc((void **)&a_d, nBytes);
    cudaMalloc((void **)&b_d, nBytes);
    cudaMalloc((void **)&c_d, nBytes);

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    cudaEvent_t start, stop;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host to Device
    cudaEventRecord(start, 0);
    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, nBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Pinned HtoD memcpy time: %.3f ms\n", ms);

    // Kernel
    dim3 block(1024);
    dim3 grid(nElem / block.x);
    cudaEventRecord(start, 0);
    sumArraysGPU<<<grid, block>>>(a_d, b_d, c_d);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %.3f ms\n", ms);

    // Device to Host
    cudaEventRecord(start, 0);
    cudaMemcpy(c_h, c_d, nBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Pinned DtoH memcpy time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFreeHost(a_h);
    cudaFreeHost(b_h);
    cudaFreeHost(c_h);

    return 0;
}
