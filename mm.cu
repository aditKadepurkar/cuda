/*
 Matrix multiplication kernels
 naive matrix multiplication kernel
 and tensor core matrix multiplication kernel

 these kernels are written for matrices of dim 1024 x 1024 currently




 Compilation: nvcc -o mm mm.cu -arch=sm_86
  - This arch flag is for the RTX A5500

*/

#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda.h>


#define STATIC_SIZE 4096

using namespace nvcuda;

// C = A * B
__global__ void naive_mm_kernel_1024(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float total = 0.0f;

    for (int k = 0; k < STATIC_SIZE; k++) {
        total += A[j * STATIC_SIZE + k] * B[i + k * STATIC_SIZE];
    }

    C[j * STATIC_SIZE + i] = total;
}

// Tensor core matrix multiplication kernel
__global__ void tensor_mm_kernel_1024(half *A, half *B, float *C) {

    // int i = blockIdx.x;
    // int j = blockIdx.y;

    // 16 x 16 tile frags
    // for my reference, the api: wmma::fragment<usage, m, n, k, dtype, layout>
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);

}

int main() {
    // initializing the matrices
    float *A_h = new float[STATIC_SIZE * STATIC_SIZE];
    float *B_h = new float[STATIC_SIZE * STATIC_SIZE];
    float *C_h = new float[STATIC_SIZE * STATIC_SIZE];

    float *A_d, *B_d, *C_d;

    // init A and B randomly
    srand(time(NULL));
    for (int i = 0; i < STATIC_SIZE; i++) {
        for (int j = 0; j < STATIC_SIZE; j++) {
            A_h[i * STATIC_SIZE + j] = (float)rand() / RAND_MAX;
            B_h[i * STATIC_SIZE + j] = (float)rand() / RAND_MAX;
        }
    }

    cudaEvent_t start_naive_copy, stop_naive_copy;
    cudaEvent_t start_naive, stop_naive;
    float time_naive, time_naive_copy;

    cudaEventCreate(&start_naive_copy);
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive_copy);
    cudaEventCreate(&stop_naive);

    cudaEventRecord(start_naive_copy, 0);

    // malloc on device
    cudaMalloc((void **)&A_d, STATIC_SIZE * STATIC_SIZE * sizeof(float));
    cudaMalloc((void **)&B_d, STATIC_SIZE * STATIC_SIZE * sizeof(float));
    cudaMalloc((void **)&C_d, STATIC_SIZE * STATIC_SIZE * sizeof(float));

    cudaMemcpy(A_d, A_h, STATIC_SIZE * STATIC_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, STATIC_SIZE * STATIC_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(2, 2);
    dim3 gridDim(2, 2);


    cudaEventRecord(start_naive, 0);
    naive_mm_kernel_1024<<<gridDim, blockDim>>>(A_d, B_d, C_d);
    cudaEventRecord(stop_naive, 0);

    // copy back to the host
    cudaMemcpy(C_h, C_d, STATIC_SIZE * STATIC_SIZE * sizeof(half), cudaMemcpyDeviceToHost);


    cudaEventElapsedTime(&time_naive, start_naive, stop_naive);

    cudaEventRecord(stop_naive_copy, 0);

    cudaEventElapsedTime(&time_naive_copy, start_naive_copy, stop_naive_copy);


    // std::cout << "A: " << std::endl;

    // for (int i = 0; i < STATIC_SIZE; i++) {
    //     for (int j = 0; j < STATIC_SIZE; j++) {
    //         std::cout << __half2float(A_h[i * STATIC_SIZE + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "B: " << std::endl;

    // for (int i = 0; i < STATIC_SIZE; i++) {
    //     for (int j = 0; j < STATIC_SIZE; j++) {
    //         std::cout << __half2float(B_h[i * STATIC_SIZE + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "C: " << std::endl;

    // for (int i = 0; i < STATIC_SIZE; i++) {
    //     for (int j = 0; j < STATIC_SIZE; j++) {
    //         std::cout << __half2float(C_h[i * STATIC_SIZE + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);


    half *A_d_half, *B_d_half;

    float *C_d_2;

    half *A_h_half = new half[STATIC_SIZE * STATIC_SIZE];
    half *B_h_half = new half[STATIC_SIZE * STATIC_SIZE];
    float *C_h_2 = new float[STATIC_SIZE * STATIC_SIZE];

    for (int i = 0; i < STATIC_SIZE * STATIC_SIZE; i++) {
        A_h_half[i] = __float2half(A_h[i]);
        B_h_half[i] = __float2half(B_h[i]);
    }


    dim3 tensorblockDim(32, 1);
    dim3 tensorgridDim(STATIC_SIZE / 16, STATIC_SIZE / 16);

    // start timer
    cudaEvent_t start_tensor_copy, stop_tensor_copy;
    cudaEvent_t start_tensor, stop_tensor;
    float time_tensor_copy, time_tensor;

    cudaEventCreate(&start_tensor_copy);
    cudaEventCreate(&stop_tensor_copy);
    cudaEventCreate(&start_tensor);
    cudaEventCreate(&stop_tensor);

    cudaEventRecord(start_tensor_copy, 0);

    // malloc on device
    cudaMalloc((void **)&A_d_half, STATIC_SIZE * STATIC_SIZE * sizeof(half));
    cudaMalloc((void **)&B_d_half, STATIC_SIZE * STATIC_SIZE * sizeof(half));
    cudaMalloc((void **)&C_d_2, STATIC_SIZE * STATIC_SIZE * sizeof(float));

    cudaMemcpy(A_d_half, A_h_half, STATIC_SIZE * STATIC_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d_half, B_h_half, STATIC_SIZE * STATIC_SIZE * sizeof(half), cudaMemcpyHostToDevice);

    cudaEventRecord(start_tensor, 0);

    tensor_mm_kernel_1024<<<tensorgridDim, tensorblockDim>>>(A_d_half, B_d_half, C_d_2);

    cudaEventRecord(stop_tensor, 0);

    // copy back to the host
    cudaMemcpy(C_h_2, C_d_2, STATIC_SIZE * STATIC_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_tensor_copy, 0);
    cudaEventElapsedTime(&time_tensor, start_tensor, stop_tensor);
    cudaEventElapsedTime(&time_tensor_copy, start_tensor_copy, stop_tensor_copy);

    std::cout << "Naive Implementation Time Copy: " << time_naive_copy << std::endl;
    std::cout << "Tensor Implementation Time Copy: " << time_tensor_copy << std::endl;
    std::cout << "Speedup: " << time_naive_copy / time_tensor_copy << "x" << std::endl << std::endl;
    std::cout << "Naive Implementation Time: " << time_naive << std::endl;
    std::cout << "Tensor Implementation Time Copy: " << time_tensor << std::endl;
    std::cout << "Speedup: " << time_naive / time_tensor << "x" << std::endl;

    cudaFree(A_d_half);
    cudaFree(B_d_half);
    cudaFree(C_d_2);

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
    delete[] A_h_half;
    delete[] B_h_half;
    delete[] C_h_2;

    return 0;
}




