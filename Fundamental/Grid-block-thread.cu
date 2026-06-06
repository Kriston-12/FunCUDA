#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex() {
    printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
        gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
        gridDim.x,gridDim.y,gridDim.z);
}

// int main(int argc, char **argv) {
//     int nElem = 6;
//     dim3 block(3);
//     dim3 grid((nElem + block.x - 1) / block.x);
//     printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
//     printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
//     checkIndex<<<grid,block>>>();
//     cudaDeviceReset();
//     return 0;
// }

int main(int argc, char **argv) {
    int N = 12;

    // This initialization is the same as  dim3 blockName(4, 1, 1).
    // dim3 blockName(x, y, z). parameter: x--Dimension length along x coordinate. y--Dim length along y, same for z
    dim3 block(4); 

    // This is the same as dim3 grid((N + block.x - 1) / block.x, 1, 1);
    // The grammer semantics is the same as dim3 blockName(x, y, z) explaned above
    dim3 grid((N + block.x - 1) / block.x);

    printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
    
    checkIndex<<<grid,block>>>();
    cudaDeviceReset();
    return 0;

/*Output msg is as follows
grid.x 3 grid.y 1 grid.z 1
block.x 4 block.y 1 block.z 1
threadIdx:(0,0,0) blockIdx:(2,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(1,0,0) blockIdx:(2,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(2,0,0) blockIdx:(2,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(3,0,0) blockIdx:(2,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(0,0,0) blockIdx:(1,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(1,0,0) blockIdx:(1,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(2,0,0) blockIdx:(1,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(3,0,0) blockIdx:(1,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(0,0,0) blockIdx:(0,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(1,0,0) blockIdx:(0,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(2,0,0) blockIdx:(0,0,0) blockDim:(4,1,1)        gridDim(3,1,1)
threadIdx:(3,0,0) blockIdx:(0,0,0) blockDim:(4,1,1)        gridDim(3,1,1)*/    
}


