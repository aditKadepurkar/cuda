/*
This is my second CUDA program.
We will be implementing RRT(Rapidly-exploring Random Tree) in CUDA.
This is a path planning algorithm that is used in robotics. Read more
about it here: https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree

We will be implementing this algorithm to work in an N-dimensional(SPACE_DIM in this case) space.

*/

#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <device_launch_parameters.h>

// Define the dimension of the space.
#define SPACE_DIM 2
#define BLOCK_SIZE 256
#define MAX_TREE_SIZE 100000
#define MAX_THEADS_PER_BLOCK 1024

// Struct for representing points in the SPACE_DIM-dimensional space.
struct Point {
    float coords[SPACE_DIM];

    __host__ __device__ Point() {
        for (int i = 0; i < SPACE_DIM; ++i) coords[i] = 0.0f;
    }

    __host__ __device__ Point(const float* values) {
        for (int i = 0; i < SPACE_DIM; ++i) {
            coords[i] = values[i];
        }
    }

    __host__ __device__ Point(float x, float y) {
        coords[0] = x;
        coords[1] = y;
    }

    __host__ __device__ float distance(const Point& other) const {
        float sum = 0.0f;
        for (int i = 0; i < SPACE_DIM; ++i) {
            float diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return sqrtf(sum);
    }
};


__global__ void initRandomStates(curandState* states, unsigned long seed);
__global__ void samplePoints(Point* samples, curandState* states, int n_samples, Point lower_bound, Point upper_bound);
__global__ void findNearestNeighbors(const Point* tree, const Point* samples, int* nearest_indices, int tree_size, int n_samples);
__global__ void collisionCheck(const Point* samples, const Point* tree, const int* nearest_indices, int* collision_free, int n_samples);
__global__ void extendTree(Point* tree, const Point* samples, const int* nearest_indices, const int* collision_free, int* new_nodes, int tree_size, int n_samples);


__global__ void initRandomStates(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

int main() {
    Point lower_bound(0.0f, 0.0f);
    Point upper_bound(10.0f, 10.0f);

    // Initialize RRT tree (host-side)
    std::vector<Point> tree;
    tree.emplace_back(0.0f, 0.0f); // Start at the origin

    // various variables we will need on the device
    Point* d_tree;
    Point* d_samples;
    curandState* d_states;
    int* d_nearest_indices;
    int* d_collision_free;

    const int sample_count = 1024;

    // malloc GPU memory for each of these
    cudaMalloc(&d_tree, MAX_TREE_SIZE * sizeof(Point));
    cudaMalloc(&d_samples, sample_count * sizeof(Point));
    cudaMalloc(&d_states, sample_count * sizeof(curandState));
    cudaMalloc(&d_nearest_indices, sample_count * sizeof(int));
    cudaMalloc(&d_collision_free, sample_count * sizeof(int));

    
    // Initialize random states
    initRandomStates<<<(sample_count + BLOCK_SIZE - 1), sample_count>>>(d_states, time(0));

    while (tree.size() < MAX_TREE_SIZE) {
        // sample points
        samplePoints<<<(sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_samples,
            d_states, 
            sample_count, 
            lower_bound, 
            upper_bound
        );

        // move to the device
        cudaMemcpy(d_tree, tree.data(), tree.size() * sizeof(Point), cudaMemcpyHostToDevice);

        // find nearest neighbors
        findNearestNeighbors<<<(sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_tree,
            d_samples,
            d_nearest_indices,
            tree.size(),
            sample_count
        );

        // collision check
        collisionCheck<<<(sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_samples,
            d_tree,
            d_nearest_indices,
            d_collision_free,
            sample_count
        );

        // extend tree
        extendTree<<<(sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_tree,
            d_samples,
            d_nearest_indices,
            d_collision_free,
            d_tree,
            tree.size(),
            sample_count
        );

        // copy the tree back to the host
        cudaMemcpy(tree.data(), d_tree, tree.size() * sizeof(Point), cudaMemcpyDeviceToHost);



    }


    return 0;
}


