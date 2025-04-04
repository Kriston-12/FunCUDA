#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 24)  // 16M elements
#define BLOCK_SIZE 256

// Array of Structs
struct ParticleAoS {
    float x, y, z;
};

// Kernel for reading AoS
__global__ void readAoS(ParticleAoS* particles, float* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        ParticleAoS p = particles[idx];
        out[idx] = p.x + p.y + p.z;
    }
}

// struct of Arrays 
struct ParticleSoA {
    float* x;
    float* y;
    float* z;
};

// Kernel for reading SoA
__global__ void readSoA(ParticleSoA p, float* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        out[idx] = p.x[idx] + p.y[idx] + p.z[idx];
    }
}

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    
    ParticleAoS* d_particles_aos;
    float* d_out_aos;
    checkCuda(cudaMalloc(&d_particles_aos, sizeof(ParticleAoS) * N), "malloc d_particles_aos");
    checkCuda(cudaMalloc(&d_out_aos, sizeof(float) * N), "malloc d_out_aos");

    // This is a coarse way to allocate struct of arrays 
    ParticleSoA p_soa;
    float *d_out_soa;
    checkCuda(cudaMalloc(&p_soa.x, sizeof(float) * N), "malloc x");
    checkCuda(cudaMalloc(&p_soa.y, sizeof(float) * N), "malloc y");
    checkCuda(cudaMalloc(&p_soa.z, sizeof(float) * N), "malloc z");
    checkCuda(cudaMalloc(&d_out_soa, sizeof(float) * N), "malloc d_out_soa");

    // Timers
    cudaEvent_t start, stop;
    float time_aos, time_soa;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x);

    // Array of Struct timing--expected to be slower 
    cudaEventRecord(start);
    readAoS<<<grid, block>>>(d_particles_aos, d_out_aos);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_aos, start, stop);

    // Struct of array timing 
    cudaEventRecord(start);
    readSoA<<<grid, block>>>(p_soa, d_out_soa);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_soa, start, stop);

    // 
    printf("Read AoS Time: %.3f ms\n", time_aos);
    printf("Read SoA Time: %.3f ms\n", time_soa);

    // Cleanup
    cudaFree(d_particles_aos);
    cudaFree(d_out_aos);
    cudaFree(p_soa.x);
    cudaFree(p_soa.y);
    cudaFree(p_soa.z);
    cudaFree(d_out_soa);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;

    // Output is here
    // Read AoS Time: 66.859 ms
    // Read SoA Time: 0.879 ms
}
