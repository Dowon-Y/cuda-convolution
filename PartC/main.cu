#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1

#define BLOCK_SIZE 18
#define SHARED_I_SIZE C * BLOCK_SIZE * BLOCK_SIZE
#define SHARED_F_SIZE FW * FH * C

typedef struct {
  int width;
  int height;
  int channel;
  int stride;
  double* elements;
} Matrix;

typedef struct {
  int width;
  int height;
  int channel;
  int count;
  double* elements;
} Filter;

__global__ void SimpleConvolutionKernel(Matrix I, Filter F, Matrix O) {
    // one thread calculates one output of matrix O
    int outputChannelIndex = blockIdx.x;
    int col = threadIdx.x;
    int row = blockIdx.y;
    int inputGrid = I.width * I.height;
    int filterGrid = F.width * F.height;
    int filterOffset = outputChannelIndex * F.channel * filterGrid; // offset to move to the next count (K)
    int outputGrid = O.width * O.height;
    double sum = 0.0;
    for (int c = 0; c < F.channel; c++) {
        for (int j = 0; j < F.height; j++) {
            for (int i = 0; i < F.width; i++) {
                sum += I.elements[c * inputGrid + (row + j) * I.width + (col + i)] * 
                    F.elements[filterOffset + c * filterGrid + (F.height - j - 1) * F.width + (F.width - i - 1)];
            }
        }
    }
    O.elements[outputChannelIndex * outputGrid + row * O.width + col] = sum;
}

__global__ void TiledConvolutionKernel(Matrix I, Filter F, Matrix O) {
    int outputChannelIndex = blockIdx.z;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int inputGrid = I.width * I.height;
    int filterGrid = F.width * F.height;
    int filterSize = filterGrid * F.channel;
    int isubGrid = BLOCK_SIZE * BLOCK_SIZE;
    int outputGrid = O.width * O.height;

    // get the first elemenet (x = 0, y = 0) of block with c = 0
    double *Isub;
    int y_0_in_I = block_row * (BLOCK_SIZE - 2 * P);
    int x_0_in_I = block_col * (BLOCK_SIZE - 2 * P);
    Isub = &I.elements[y_0_in_I * I.width + x_0_in_I];
    
    // define and load I to shared memory (18 x 18 x (3 channels) in our example)
    __shared__ double shared_I[SHARED_I_SIZE];
    for (int c = 0; c < I.channel; c++) {
        shared_I[c * isubGrid + thread_row * BLOCK_SIZE + thread_col] = Isub[c * inputGrid + thread_row * I.width + thread_col];
    }

    // define and load F to shared memory
    __shared__ double shared_F[SHARED_F_SIZE];
    int thread_idx_in_block = thread_row * BLOCK_SIZE + thread_col;
    if (thread_idx_in_block < filterSize) {
        shared_F[thread_idx_in_block] = F.elements[outputChannelIndex * filterSize + thread_idx_in_block];
    }

    __syncthreads();
    
    // do convolution - 16 * 16 times
    if (thread_row < BLOCK_SIZE - 2 * P && thread_col < BLOCK_SIZE - 2 * P) {
        double sum = 0.0;
        for (int c = 0; c < F.channel; c++) {
            for (int j = 0; j < F.height; j++) {
                for (int i = 0; i < F.width; i++) {
                    sum += shared_I[c * isubGrid + (thread_row + j) * BLOCK_SIZE + (thread_col + i)] * 
                        shared_F[c * filterGrid + (F.height - j - 1) * F.width + (F.width - i - 1)];
                }
            }
        }
        int row = block_row * (BLOCK_SIZE - 2 * P) + thread_row;
        int col = block_col * (BLOCK_SIZE - 2 * P) + thread_col;
        O.elements[outputChannelIndex * outputGrid + row * O.width + col] = sum;
    }
}

// Create a matrix in host memory.
Matrix makeHostMatrix(int width, int height, int channel){
    Matrix newHostMatrix;
    newHostMatrix.width = width;
    newHostMatrix.height = height;
    newHostMatrix.channel = channel;
    size_t size = newHostMatrix.width * newHostMatrix.height * newHostMatrix.channel * sizeof(double);
    newHostMatrix.elements = (double*)malloc(size);
    return newHostMatrix;
}

// Initialize input matrix to I[c, x, y] = c · (x + y)
void initHostMatrix(Matrix M) {
    int gridSize = M.width * M.height;
    for (int c = 0; c < M.channel; c++) {
        for (int y = 0; y < M.height; y++) {
            for (int x = 0; x < M.width; x++) {
                M.elements[x + y * M.width + c * gridSize] = c * (x + y);
            }
        }
    }
}

Matrix paddHostInput(Matrix I, int padding) {
    Matrix I_0;
    I_0.width = I.width + 2 * padding;
    I_0.height = I.height + 2 * padding;
    int newGridSize = I_0.width * I_0.height; // to reduce calculation
    int oldGridSize = I.width * I.height; // to reduce calculation
    I_0.channel = I.channel;
    size_t size = I_0.width * I_0.height * I_0.channel * sizeof(double);
    I_0.elements = (double*)malloc(size);
    for (int c = 0; c < I_0.channel; c++) {
        for (int y = 0; y < I_0.height; y++) {
            for (int x = 0; x < I_0.width; x++) {
                if (y < padding || y >= I.height + padding || x < padding || x >= I.width + padding) {
                    I_0.elements[x + y * I_0.width + c * newGridSize] = 0;
                } else {
                    I_0.elements[x + y * I_0.width + c * newGridSize] = 
                        I.elements[(x - padding) + (y - padding) * I.width + c * oldGridSize];
                }
            }
        }
    }
    return I_0;
}

Matrix makeMyHostInput(int width, int height, int channel) {
    Matrix myInput = makeHostMatrix(width, height, channel);
    initHostMatrix(myInput);
    return myInput;
}

Filter makeHostFilter(int width, int height, int channel, int count) {
    Filter newHostFilter;
    newHostFilter.width = width;
    newHostFilter.height = height;
    newHostFilter.channel = channel;
    newHostFilter.count = count;
    size_t size = newHostFilter.width * newHostFilter.height * newHostFilter.channel *
        newHostFilter.count * sizeof(double);
    newHostFilter.elements = (double*)malloc(size);
    return newHostFilter;
}

// initialize filter to F[k, c, i, j] = (c + k) · (i + j)
void initHostFilter(Filter F) {
    int gridSize = F.width * F.height; // to reduce calculation
    for (int k = 0; k < F.count; k++) {
        for (int c = 0; c < F.channel; c++) {
            for (int j = 0; j < F.height; j++) {
                for (int i = 0; i < F.width; i++) {
                    F.elements[i + j * F.width + c * gridSize + k * gridSize * F.channel] =
                        (c + k) * (i + j);
                }
            }
        }
    }
}

Filter makeMyHostFilter(int width, int height, int channel, int count) {
    Filter myFilter = makeHostFilter(width, height, channel, count);
    initHostFilter(myFilter);
    return myFilter;
}

// Print a matrix stored in host memory.
void printMatrix(Matrix M, const char* name) {
    printf("\n%s \n",name);
    int gridSize = M.width * M.height;
    for (int c = 0; c < M.channel; c++) {
        for (int y = 0; y < M.height; y++) {
            for (int x = 0; x < M.width; x++) {
                printf("%f ", M.elements[x + y * M.width + c * gridSize]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// Create a new matrix in device memory.
Matrix makeDeviceMatrix(Matrix M, bool copy){
    Matrix newDeviceMatrix;
    newDeviceMatrix.width = M.width;
    newDeviceMatrix.stride = M.width;
    newDeviceMatrix.height = M.height;
    newDeviceMatrix.channel = M.channel;
    size_t size = M.width * M.height * M.channel * sizeof(double);
    cudaMalloc((void**) &newDeviceMatrix.elements, size);
    if (copy) {
        cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
    }
    return newDeviceMatrix;
}

// Create a new filter in device memory.
Filter makeDevicefilter(Filter F, bool copy){
    Filter newDeviceFilter;
    newDeviceFilter.width = F.width;
    newDeviceFilter.height = F.height;
    newDeviceFilter.channel = F.channel;
    newDeviceFilter.count = F.count;
    size_t size = F.width * F.height * F.channel * F.count * sizeof(double);
    cudaMalloc((void**) &newDeviceFilter.elements, size);
    if (copy) {
        cudaMemcpy(newDeviceFilter.elements, F.elements, size, cudaMemcpyHostToDevice);
    }
    return newDeviceFilter;
}


void convolution_c1(const Matrix input, const Filter filter, Matrix output, float& executionTime){
    Matrix device_input = makeDeviceMatrix(input, true);
    Filter device_filter = makeDevicefilter(filter, true);
    Matrix device_output = makeDeviceMatrix(output, false);

    // Define grid topology
    dim3 dimBlock(output.width); // 1024 in our example
    dim3 dimGrid(output.channel, output.height); // (64, 1024) in our example
    
    // Invoke kernel for warm up
    SimpleConvolutionKernel<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Synchronize to make sure everyone is done in the warmup.
    cudaThreadSynchronize();

    // Set up CUDA events
    cudaEvent_t start_G, stop_G;
    cudaEventCreate(&start_G);
    cudaEventCreate(&stop_G);

    // Record the start event
    cudaEventRecord(start_G);

    // Invoke kernel for real
    SimpleConvolutionKernel<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Synchronize to make sure everyone is done.
    cudaThreadSynchronize();

    // Record the stop event
    cudaEventRecord(stop_G);
    cudaEventSynchronize(stop_G);

    // Calculate the elapsed time
    cudaEventElapsedTime(&executionTime, start_G, stop_G);

    // Copy the result to the host memory from device memory
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);

    cudaEventDestroy(start_G);
    cudaEventDestroy(stop_G);
}

void convolution_c2(const Matrix input, const Filter filter, Matrix output, float& executionTime){
    Matrix device_input = makeDeviceMatrix(input, true);
    Filter device_filter = makeDevicefilter(filter, true);
    Matrix device_output = makeDeviceMatrix(output, false);

    // Define grid topology
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 18, 18 in this example
    int block_per_row = (input.width - 2*P) / (BLOCK_SIZE - 2*P);
    int block_per_col = (input.height - 2*P)/ (BLOCK_SIZE - 2*P);
    dim3 dimGrid(block_per_row, block_per_col, K); // 64, 64, 64 in this example
    
    // Invoke kernel for warm up
    TiledConvolutionKernel<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Synchronize to make sure everyone is done in the warmup.
    cudaThreadSynchronize();

    // Set up CUDA events
    cudaEvent_t start_G, stop_G;
    cudaEventCreate(&start_G);
    cudaEventCreate(&stop_G);

    // Record the start event
    cudaEventRecord(start_G);

    // Invoke kernel for real
    TiledConvolutionKernel<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);

    // Synchronize to make sure everyone is done.
    cudaThreadSynchronize();

    // Record the stop event
    cudaEventRecord(stop_G);
    cudaEventSynchronize(stop_G);

    // Calculate the elapsed time
    cudaEventElapsedTime(&executionTime, start_G, stop_G);

    // Copy the result to the host memory from device memory
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);

    cudaEventDestroy(start_G);
    cudaEventDestroy(stop_G);
}


void convolution_c3(const Matrix input, const Filter filter, Matrix output, float& executionTime){
    Matrix device_input = makeDeviceMatrix(input, true);
    Filter device_filter = makeDevicefilter(filter, true);
    Matrix device_output = makeDeviceMatrix(output, false);
    
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Create input and output tensor descriptor
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W);

    // Create output tensor descriptor
    cudnnTensorDescriptor_t outputDesc;
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W);

    // Create filter and output descriptors
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);

    // Create convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    // Declare and allocate memory for performance results
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    cudnnConvolutionFwdAlgo_t selectedAlgo;
    int returnedCount;
    cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc, 1, &returnedCount, &perfResults);
    selectedAlgo = perfResults.algo;

    // Get the workspace size required for the convolution operation
    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, selectedAlgo, &workspaceSize);

    // Allocate workspace
    void* workspace;
    cudaMalloc(&workspace, workspaceSize);

    // ------ perform convolution ------
    double alpha = 1.0, beta = 0.0;

    // Set up CUDA events
    cudaEvent_t start_G, stop_G;
    cudaEventCreate(&start_G);
    cudaEventCreate(&stop_G);

    // Record the start event
    cudaEventRecord(start_G);

    cudnnConvolutionForward(cudnn, &alpha, inputDesc, device_input.elements, filterDesc, device_filter.elements,
                            convDesc, selectedAlgo, workspace, workspaceSize, &beta,
                            outputDesc, device_output.elements);

    cudaThreadSynchronize();

    // Record the stop event
    cudaEventRecord(stop_G);
    cudaEventSynchronize(stop_G);

    // Calculate the elapsed time
    cudaEventElapsedTime(&executionTime, start_G, stop_G);

    // Copy the result to the host memory from device memory
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);

    cudaEventDestroy(start_G);
    cudaEventDestroy(stop_G);

    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

}

double getCheckSum(Matrix M) {
    double sum = 0.0;
    int grid = M.width * M.height;
    for (int c = 0; c < M.channel; c++) {
        for (int y = 0; y < M.height; y++) {
            for (int x = 0; x < M.width; x++) {
                sum += M.elements[x + y * M.width + c * grid];
            }
        }
    }
    return sum;
}

int main() {
    // Create matrices in host.
    Matrix I = makeMyHostInput(W, H, C);
    Matrix I_0 = paddHostInput(I, P);
    Filter F = makeMyHostFilter(FW, FH, C, K);
    Matrix O1 = makeHostMatrix(W, H, K);
    Matrix O2 = makeHostMatrix(W, H, K);
    Matrix O3 = makeHostMatrix(W, H, K);


    // C1
    float c1_time = 0.0;
    double c1_checksum = 0.0;
    convolution_c1(I_0, F, O1, c1_time);
    c1_checksum = getCheckSum(O1);

    // C2
    float c2_time = 0.0;
    double c2_checksum = 0.0;
    convolution_c2(I_0, F, O2, c2_time);
    c2_checksum = getCheckSum(O2);

    // C3
    float c3_time = 0.0;
    double c3_checksum = 0.0;
    convolution_c3(I, F, O3, c3_time);
    c3_checksum = getCheckSum(O3);

    // print output
    printf("%f, %.6f ms\n", c1_checksum, c1_time);
    printf("%f, %.6f ms\n", c2_checksum, c2_time);
    printf("%f, %.6f ms\n", c3_checksum, c3_time);

    free(I.elements);
    free(I_0.elements);
    free(F.elements);
    free(O1.elements);
    free(O2.elements);
    free(O3.elements);

    return 0;
}