#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define N 64
#define TILE 16

// Tensor Core matrix multiplication kernel
// One warp (32 threads) computes one 16x16 tile of C.
__global__ void matrixMulTensorCore(const half *A, const half *B, float *C)
{
    // Each block corresponds to one 16x16 output tile
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    // Matrix A fragment: 16x16
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;

    // Matrix B fragment: 16x16
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag;

    // Accumulator fragment: 16x16
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;

                  
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // A 64x64 matrix has 4 tiles in the K dimension.
    // Therefore each output tile requires 4 MMA operations.
    for (int k = 0; k < N; k += TILE)
    {
        // Starting position of the 16x16 tile of A
        const half *A_tile = A + tileRow * TILE * N + k;

        // Starting position of the 16x16 tile of B
        const half *B_tile = B + k * N + tileCol * TILE;

        // Load 16x16 tile of A
        wmma::load_matrix_sync( a_frag,A_tile,N);

        // Load 16x16 tile of B
        wmma::load_matrix_sync(b_frag,B_tile,N);

        // Tensor Core matrix multiply-accumulate
        c_frag = wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
    }

    // Starting position of the output 16x16 tile
    float *C_tile =C + tileRow * TILE * N + tileCol * TILE;

    // Store the result
    wmma::store_matrix_sync(C_tile,c_frag,N,wmma::mem_row_major);
}


int main()
{
    half *h_A;
    half *h_B;
    float *h_C;

    half *d_A;
    half *d_B;
    float *d_C;

    size_t sizeA = N * N * sizeof(half);
    size_t sizeB = N * N * sizeof(half);
    size_t sizeC = N * N * sizeof(float);

    // Allocate host memory
    h_A = (half *)malloc(sizeA);
    h_B = (half *)malloc(sizeB);
    h_C = (float *)malloc(sizeC);

    // Initialize A and B
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(1.0f);
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    /*
       64x64 matrix / 16x16 tile = 4x4 tiles

             4 tiles
       ┌────┬────┬────┬────┐
       │    │    │    │    │
       ├────┼────┼────┼────┤
       │    │    │    │    │
       ├────┼────┼────┼────┤
       │    │    │    │    │
       ├────┼────┼────┼────┤
       │    │    │    │    │
       └────┴────┴────┴────┘
          4 tiles

       Total = 4 x 4 = 16 output tiles
    */

    // One block computes one 16x16 tile.
    // One warp = 32 threads.
    dim3 block(32);

    // 64/16 = 4 tiles in each direction.
    dim3 grid(N / TILE, N / TILE);

    // Launch kernel
    matrixMulTensorCore<<<grid, block>>>(d_A, d_B, d_C);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print selected results
    printf("C[0][0]   = %f\n", h_C[0]);
    printf("C[1][1]   = %f\n", h_C[N + 1]);
    printf("C[16][16] = %f\n", h_C[16 * N + 16]);
    printf("C[32][32] = %f\n", h_C[32 * N + 32]);
    printf("C[63][63] = %f\n", h_C[63 * N + 63]);

    /*
       Since every element of A and B is 1:

       C[i][j] = 1*1 + 1*1 + ... + 1*1
               = 64

       Therefore every element of C should be 64.
    */

    // Verify result
    bool correct = true;

    for (int i = 0; i < N * N; i++)
    {
        if (fabs(h_C[i] - 64.0f) > 0.001f)
        {
            correct = false;
            break;
        }
    }

    if (correct)
        printf("Matrix multiplication is CORRECT.\n");
    else
        printf("Matrix multiplication is INCORRECT.\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
