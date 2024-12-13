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

// Define the dimension of the space.
#define SPACE_DIM 2

// Struct for representing points in the SPACE_DIM-dimensional space.
struct Point {
    float coords[SPACE_DIM];

    __host__ __device__ Point() {
        for (int i = 0; i < SPACE_DIM; ++i) coords[i] = 0.0f;
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


int main() {
    





    return 0;
}


