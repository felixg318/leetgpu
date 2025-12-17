#include <cuda_runtime.h>

__global__ void reduction_kernel(const float *input, float *output, int N){
    __shared__ float shmem[512];

    int tid = threadIdx.x; 
    int gid = tid + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = gid; i < N; i += stride) 
        sum += input[i];
    
    shmem[tid] = sum;
    __syncthreads(); 

    for(int s = blockDim.x/2; s > 0; s /= 2) {
        if(tid < s) 
            shmem[tid] += shmem[tid + s];
    
        __syncthreads();
    }

    if(tid == 0)
        atomicAdd(output, shmem[0]);
}

extern "C" void solve(const float* input, float* output, int N) { 
    int blockSize = 512;
    int gridSize = (N + blockSize - 1) / blockSize;
    reduction_kernel<<<gridSize, blockSize>>>(input, output, N);
    cudaDeviceSynchronize();
    float h_zero = 0.0f; cudaMemcpy(&output, &h_zero, sizeof(float), cudaMemcpyHostToDevice);
}
