#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define N 64
#define TILE 16

__global__ void K1(half *A, half *B, float *C)
{
    // One warp calculates one 16x16 tile
    int warp = threadIdx.x / 32;

    int row = warp / 4;
    int col = warp % 4;

    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fill_fragment(c, 0.0f);

    // 4 tiles in the K direction
    for(int k = 0; k < 4; k++)
    {
        wmma::fragment<
            wmma::matrix_a,16,16,16,half,wmma::row_major
        > a;

        wmma::fragment<
            wmma::matrix_b,16,16,16,half,wmma::row_major
        > b;

        wmma::load_matrix_sync(
            a,
            A + row*16*N + k*16,
            N
        );

        wmma::load_matrix_sync(
            b,
            B + k*16*N + col*16,
            N
        );

        wmma::mma_sync(c,a,b,c);
    }

    wmma::store_matrix_sync(
        C + row*16*N + col*16,
        c,
        N,
        wmma::mem_row_major
    );
}


int main()
{
    half *A, *B;
    float *C;

    half *d_A, *d_B;
    float *d_C;

    // Host memory
    A = (half*)malloc(N*N*sizeof(half));
    B = (half*)malloc(N*N*sizeof(half));
    C = (float*)malloc(N*N*sizeof(float));

    // Random values
    for(int i=0;i<N*N;i++)
    {
        A[i] = __float2half((rand()%10)+1);
        B[i] = __float2half((rand()%10)+1);
    }

    // GPU memory
    cudaMalloc(&d_A,N*N*sizeof(half));
    cudaMalloc(&d_B,N*N*sizeof(half));
    cudaMalloc(&d_C,N*N*sizeof(float));

    cudaMemcpy(d_A,A,N*N*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,N*N*sizeof(half),cudaMemcpyHostToDevice);

    // 16 warps = 16 tiles
    K1<<<1,512>>>(d_A,d_B,d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(C,d_C,N*N*sizeof(float),cudaMemcpyDeviceToHost);

    printf("First 4x4 elements of C:\n");

    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            printf("%8.2f ",C[i*N+j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
