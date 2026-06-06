#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}


void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}


__global__ void printThreadIndex(float *A,const int nx,const int ny)
{
  int ix=threadIdx.x+blockIdx.x*blockDim.x;
  int iy=threadIdx.y+blockIdx.y*blockDim.y;
  unsigned int idx=iy*nx+ix;
  printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
          "global index %2d ival %2d\n",threadIdx.x,threadIdx.y,
          blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
}
int main(int argc,char** argv)
{
  initDevice(0);
  int nx=8,ny=6;
  int nxy=nx*ny;
  int nBytes=nxy*sizeof(float);

  // Allocate host memory
  float* A_host=(float*)malloc(nBytes);

  // Allocate device memory
  float *A_dev=NULL;
  CHECK(cudaMalloc((void**)&A_dev,nBytes));

  // Copy data from host memory to device memory 
  cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice);

  dim3 block(4,2); // Here we need to imagine it as a 2 x 4 matrix instead of a 4 x 2 matrix in our grid. Visual Explanantion see Readme.md
  printf("block.x is %d\nblock.y is %d\n", block.x, block.y); 

  dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1); // The grid will have a dimension of (8/4, 6/2) = (2, 3), namely 6 blocks 

  // We could imagine we have a 6x8 matrix, and this matrix is split into 6 smaller matrix 
  printThreadIndex<<<grid,block>>>(A_dev,nx,ny);

  CHECK(cudaDeviceSynchronize());
  cudaFree(A_dev);
  free(A_host);

  cudaDeviceReset();
  return 0;
}