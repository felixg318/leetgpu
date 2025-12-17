#include <cuda_runtime.h>

#define TILE 16
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    __shared__ float Ads[TILE][TILE];
    __shared__ float Bds[TILE][TILE];

    int bx = blockIdx.x; int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = bx * TILE + tx;
    int row = by * TILE + ty;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1)/TILE; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        Ads[ty][tx] = (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bds[ty][tx] = (bRow < N && col < K) ? B[bRow * K + col] : 0.0f;

        __syncthreads();

        for(int i = 0; i < TILE; ++i)
            sum += Ads[ty][i] * Bds[i][tx];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
