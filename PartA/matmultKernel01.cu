#include "matmultKernel.h"

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
	int offset_y = BLOCK_SIZE * B.width / FOOTPRINT_SIZE;
	int offset_x = BLOCK_SIZE * A.height / FOOTPRINT_SIZE;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// use coalesced memeroy access approach with offset
	float Cvalue_1 = 0, Cvalue_2 = 0, Cvalue_3 = 0, Cvalue_4 = 0;
	for (int e = 0; e < A.width; ++e) {
		Cvalue_1 += A.elements[row * A.width + e] * B.elements[e * B.width + col];
		Cvalue_2 += A.elements[(row + offset_y) * A.width + e] * B.elements[e * B.width + col];
		Cvalue_3 += A.elements[row * A.width + e] * B.elements[e * B.width + (col + offset_x)];
		Cvalue_4 += A.elements[(row + offset_y) * A.width + e] * B.elements[e * B.width + (col + offset_x)];
	}   
	C.elements[row * C.width + col] = Cvalue_1;
	C.elements[(row + offset_y) * C.width + col] = Cvalue_2;
	C.elements[row * C.width + (col + offset_x)] = Cvalue_3;
	C.elements[(row + offset_y) * C.width + (col + offset_x)] = Cvalue_4;
}
