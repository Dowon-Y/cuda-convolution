#include "vecaddKernel.h"

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int offset = GRID_SIZE * BLOCK_SIZE;
    int index;

    for (int i = 0; i < N; i++) {
        index =  offset * i + (blockIdx.x * blockDim.x + threadIdx.x);
        C[index] = A[index] + B[index];
    }
}
