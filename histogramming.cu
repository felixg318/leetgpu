#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* hist, int N, int bins) {

    extern __shared__ unsigned int hist_s[];
    for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x)
        hist_s[bin] = 0u;
    __syncthreads();
    
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    for(unsigned int i = idx; i < N; i += blockDim.x * gridDim.x)
        if(idx < N) 
            atomicAdd(&hist_s[input[idx]], 1);
    __syncthreads();
    
    for (unsigned int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
        unsigned int binVal = hist_s[bin];
        if (binVal > 0)
            atomicAdd(&hist[bin], binVal);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid( (N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    size_t shmem = (size_t)num_bins * sizeof(unsigned int);
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, shmem>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
