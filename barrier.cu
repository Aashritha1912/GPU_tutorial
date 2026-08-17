#include <stdio.h>

__device__ int count = 0;

__global__ void barrierKernel()
{
    int threadid = threadIdx.x;

    printf("Thread %d reached barrier\n", threadid);

    atomicAdd(&count, 1);

    while(count < blockDim.x);

    __syncthreads();

    printf("Thread %d passed barrier\n", threadid);
}

int main()
{
    barrierKernel<<<1,10>>>();

    cudaDeviceSynchronize();

    return 0;
}