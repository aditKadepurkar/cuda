/*
By Adit Kadepurkar
Date: 2024-12-13

This is the first CUDA kernel I've written. It's a simple vector addition kernel.
It includes a test to verify the result and times the kernel execution compared
to the CPU version of the same code.

The structure I have used here will be used in my future CUDA programs as well.
(ie: kernel, test, timing, etc)


Output:
CUDA Kernel Time: 0.02496 ms  Speedup(CPU): 96.1962
CUDA Copy + Kernel Time: 2.46704 ms  Speedup(CPU): 0.973254
CPU Time: 2.40106 ms  Speedup(CPU): 1

Analysis:
Definitely see a speedup in kernel execution time vs CPU version, but when we
include the overhead from the copy time between host and device and back, the
CPU performs better. This is expected as the kernel is very simple and the
overhead from copying data is significant.

These results are from an array of size 1000000. With larger arrays, the GPU
should perform better as the overhead from copying data will be less significant.
For smaller arrays, the CPU will perform better.
*/



#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // setting n to 10000
    const int n = 1000000;


    // these are the host and device pointers
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;


    // allocating memory on host
    h_a = (float *)malloc(n * sizeof(float));
    h_b = (float *)malloc(n * sizeof(float));
    h_c = (float *)malloc(n * sizeof(float));

    // give them some values just for the sake of it
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = (n - i) / 2;
    }

    // allocating memory on device
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));

    cudaEvent_t start_cpy, stop_cpy;

    // start the timer for full copy
    cudaEventCreate(&start_cpy);
    cudaEventCreate(&stop_cpy);
    cudaEventRecord(start_cpy, 0);

    // copy data over to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);


    // kernel parameters(if you are reading this later and have no idea what this is,
    // recall the grid of blocks and threads concept)
    int threads_per_block = 512;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // call the kernel
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);

    // stop the timer for kernel
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float cu_time;
    cudaEventElapsedTime(&cu_time, start, stop);


    // now we have to copy the result back to the host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // stop the timer for full copy
    cudaEventRecord(stop_cpy, 0);
    cudaEventSynchronize(stop_cpy);
    float cu_time_cpy;
    cudaEventElapsedTime(&cu_time_cpy, start_cpy, stop_cpy);

    // verify the result
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cout << "Error at index " << i << std::endl;
            break;
        }
    }

    std::cout << "Cuda Version Successful!" << std::endl;


    // CPU version of the same code
    float *cpu_c = (float *)malloc(n * sizeof(float));
    
    // start the timer for CPU version
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < n; i++) {
        cpu_c[i] = h_a[i] + h_b[i];
    }

    // stop the timer for CPU version
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);

    // verify the result
    for (int i = 0; i < n; i++) {
        if (cpu_c[i] != h_c[i]) {
            std::cout << "Error at index " << i << std::endl;
            break;
        }
    }

    std::cout << "CPU Version Successful!" << std::endl;

    // print the times
    std::cout << "CUDA Kernel Time: " << cu_time << " ms  " << "Speedup(CPU): " << cpu_time / cu_time << std::endl;
    std::cout << "CUDA Copy + Kernel Time: " << cu_time_cpy << " ms  " << "Speedup(CPU): " << cpu_time / cu_time_cpy << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms  " << "Speedup(CPU): " << cpu_time / cpu_time << std::endl;


    // free memory on host and device
    free(cpu_c);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);


    return 0;

}
