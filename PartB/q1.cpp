#include <iostream>
#include <chrono>
#include <math.h>

void add(float *arr_x, float *arr_y, int N) {
    for (int i = 0; i < N; ++i) {
        arr_y[i] = arr_x[i] + arr_y[i];
    }
}

int main(int argc, char** argv) {
    int K = 1;
    int N = 1<<20;

    if (argc == 2) {
        sscanf(argv[1], "%d", &K);
    }

    N = K * N;
    float *arr_1 = (float*)malloc(N * sizeof(float));
    float *arr_2 = (float*)malloc(N * sizeof(float));

    // initialize array
    for (int i = 0; i < N; ++i) {
        arr_1[i] = 1.0f;
        arr_2[i] = 2.0f;
    }

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Number of elements: " << K << " millions" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    add(arr_1, arr_2, N);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << (float)duration.count() / 1000000 << " seconds" << std::endl;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(arr_2[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    // free memomry
    free(arr_1);
    free(arr_2);
    
    return 0;
}   


