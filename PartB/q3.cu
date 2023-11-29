#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	y[i] = x[i] + y[i];
}

int main(int argc, char** argv) {
	int K = 1;
	int N = 1<<20;
	int grid_size = 1;
	int block_size = 1;
	float *x, *y;

    if (argc == 4) {
        sscanf(argv[1], "%d", &K);
		sscanf(argv[2], "%d", &grid_size);
		sscanf(argv[3], "%d", &block_size);
    }
	std::cout << "-------------------------------------" << std::endl;
	std::cout << "K: " << K << ", grid size: " << grid_size << ", block size: " << block_size << std::endl;
	N = K * N;
	
	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));
	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	dim3 dimGrid(grid_size);
	dim3 dimBlock(block_size);

	// invoke kernel
	add<<<dimGrid, dimBlock>>>(N, x, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;
	// Free memory
	cudaFree(x);
	cudaFree(y);
	return 0;
}
