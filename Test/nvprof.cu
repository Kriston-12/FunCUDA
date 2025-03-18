#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

#include <time.h>
#ifdef _WIN32
#   include <windows.h>
#else
#   include <sys/time.h>
#endif
#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year   = wtm.wYear - 1900;
  tm.tm_mon   = wtm.wMonth - 1;
  tm.tm_mday   = wtm.wDay;
  tm.tm_hour   = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm. tm_isdst  = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return (0);
}
#endif

#define CHECK(call) {\
    const cudaError_t error = call;\
    if(error!=cudaSuccess) {\
        printf("Error: %s:%d,", __FILE__, __LINE__);\
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

__global__ void sumMatrix(float* A, float* B, float* C, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * ix;
    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

int main(int argc,char** argv)
{
    //printf("strating...\n");
    //initDevice(0);
    int nx=1<<13;
    int ny=1<<13;
    int nxy=nx*ny;
    int nBytes=nxy*sizeof(float);

    //Malloc
    float* A_host=(float*)malloc(nBytes);
    float* B_host=(float*)malloc(nBytes);
    float* C_host=(float*)malloc(nBytes);
    float* C_from_gpu=(float*)malloc(nBytes);

    //cudaMalloc
    float *A_dev=NULL;
    float *B_dev=NULL;
    float *C_dev=NULL;
    CHECK(cudaMalloc((void**)&A_dev,nBytes));
    CHECK(cudaMalloc((void**)&B_dev,nBytes));
    CHECK(cudaMalloc((void**)&C_dev,nBytes));


    CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev,B_host,nBytes,cudaMemcpyHostToDevice));

    int dimx=argc>2?atoi(argv[1]):32;
    int dimy=argc>2?atoi(argv[2]):32;

    double iStart,iElaps;

    // 2d block and 2d grid
    dim3 block(dimx,dimy);
    dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);
    iStart=cpuSecond();
    sumMatrix<<<grid,block>>>(A_dev,B_dev,C_dev,nx,ny);
    CHECK(cudaDeviceSynchronize());
    iElaps=cpuSecond()-iStart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)|%f sec\n",
            grid.x,grid.y,block.x,block.y,iElaps);
    CHECK(cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost));

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
    cudaDeviceReset();
    return 0;
}