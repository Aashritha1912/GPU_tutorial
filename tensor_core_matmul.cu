#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define N 64
#define TILE 16

__global__ void K1(half *A, half *B, float *C)
{
    // One warp computes one 16x16 tile
    int warp = threadIdx.x / 32;

    int row = warp / 4;
    int col = warp % 4;

    wmma::fragment<
        wmma::accumulator,
        16, 16, 16,
        float
    > c;

    wmma::fill_fragment(c, 0.0f);

    // 4 tiles in K dimension
    for (int k = 0; k < 4; k++)
    {
        wmma::fragment<
            wmma::matrix_a,
            16, 16, 16,
            half,
            wmma::row_major
        > a;

        wmma::fragment<
            wmma::matrix_b,
            16, 16, 16,
            half,
            wmma::row_major
        > b;

        wmma::load_matrix_sync(
            a,
            A + row * 16 * N + k * 16,
            N
        );

        wmma::load_matrix_sync(
            b,
            B + k * 16 * N + col * 16,
            N
        );

        // Tensor Core multiplication
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(
        C + row * 16 * N + col * 16,
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

    int size = N * N;

    A = (half *)malloc(size * sizeof(half));
    B = (half *)malloc(size * sizeof(half));
    C = (float *)malloc(size * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < size; i++)
    {
        A[i] = __float2half((rand() % 10) + 1);
        B[i] = __float2half((rand() % 10) + 1);
    }

    cudaMalloc(&d_A, size * sizeof(half));
    cudaMalloc(&d_B, size * sizeof(half));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(
        d_A, A,
        size * sizeof(half),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B, B,
        size * sizeof(half),
        cudaMemcpyHostToDevice
    );

    // 16 warps = 16 output tiles
    K1<<<1, 512>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(
        C, d_C,
        size * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    float *C_cpu;

    C_cpu = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;

            for (int k = 0; k < N; k++)
            {
                sum += __half2float(A[i * N + k])
                     * __half2float(B[k * N + j]);
            }

            C_cpu[i * N + j] = sum;
        }
    }

    float maxError = 0.0f;

    for (int i = 0; i < size; i++)
    {
        float error = fabs(C[i] - C_cpu[i]);

        if (error > maxError)
            maxError = error;
    }

    printf("\n");
printf("Tensor Core Matrix Multiplication\n\n");

printf("A : 64 x 64\n");
printf("B : 64 x 64\n");
printf("C : 64 x 64\n\n");

printf("Tile size : 16 x 16\n");
printf("Number of tiles : 16\n\n");

printf("Maximum error : %f\n", maxError);

if(maxError < 0.01)
    printf("Result : PASS\n");
else
    printf("Result : FAIL\n");

printf("\nFirst 4 x 4 elements of C:\n\n");

for(int i = 0; i < 4; i++)
{
    for(int j = 0; j < 4; j++)
        printf("%8.2f ", C[i*N+j]);

    printf("\n");
}
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);
    free(C_cpu);

    return 0;
}
