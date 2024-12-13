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
#define FLT_MAX 3.402823466e+10F

// Struct for representing points in the SPACE_DIM-dimensional space.
struct Point {
    float coords[SPACE_DIM];
    int parent_idx;  // Index of the parent node in the tree

    __host__ __device__ Point() {
        for (int i = 0; i < SPACE_DIM; ++i) coords[i] = 0.0f;
        parent_idx = -1;  // -1 indicates no parent (root node)
    }

    __host__ __device__ Point(const float* values) {
        for (int i = 0; i < SPACE_DIM; ++i) {
            coords[i] = values[i];
        }
        parent_idx = -1;
    }

    __host__ __device__ Point(float x, float y) {
        coords[0] = x;
        coords[1] = y;
        parent_idx = -1;
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

__global__ void samplePoints(Point* samples, curandState* states, int n_samples, Point lower_bound, Point upper_bound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        for (int i = 0; i < SPACE_DIM; ++i) {
            float rand_num = curand_uniform(&states[idx]);
            samples[idx].coords[i] = lower_bound.coords[i] + rand_num * (upper_bound.coords[i] - lower_bound.coords[i]);
        }
    }
}

__global__ void findNearestNeighbors(const Point* tree, const Point* samples, int* nearest_indices, int tree_size, int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        float min_dist = FLT_MAX;
        int nearest_idx = -1;
        for (int i = 0; i < tree_size; ++i) {
            float dist = samples[idx].distance(tree[i]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = i;
            }
        }
        nearest_indices[idx] = nearest_idx;
    }
}

__global__ void collisionCheck(const Point* samples, const Point* tree, const int* nearest_indices, int* collision_free, int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        collision_free[idx] = 1;
        for (int i = 0; i < SPACE_DIM; ++i) {
            if (samples[idx].coords[i] < 0.0f || samples[idx].coords[i] > 10.0f) {
                collision_free[idx] = 0;
                break;
            }
        }
    }
}

__global__ void extendTree(Point* tree, const Point* samples, const int* nearest_indices, const int* collision_free, int* new_nodes, int tree_size, int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        if (collision_free[idx]) {
            const float step_size = 0.5f;  // Adjust this value as needed
            Point& nearest = tree[nearest_indices[idx]];
            const Point& sample = samples[idx];
            float dist = nearest.distance(sample);
            
            if (dist > 0.0f) {
                for (int i = 0; i < SPACE_DIM; ++i) {
                    tree[tree_size + idx].coords[i] = nearest.coords[i] + 
                        (sample.coords[i] - nearest.coords[i]) * (step_size / dist);
                }
            } else {
                tree[tree_size + idx] = nearest;
            }
            new_nodes[idx] = 1;
        } else {
            new_nodes[idx] = 0;
        }
    }
}

void printPath(const std::vector<Point>& tree, int goal_idx) {
    std::vector<int> path;
    int current_idx = goal_idx;
    
    while (current_idx != -1) {
        path.push_back(current_idx);
        current_idx = tree[current_idx].parent_idx;
    }

    std::cout << "Path from start to goal:" << std::endl;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        std::cout << "(" << tree[*it].coords[0] << ", " 
                 << tree[*it].coords[1] << ")" << std::endl;
    }
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

    const int sample_count = 128;

    // malloc GPU memory for each of these
    cudaMalloc(&d_tree, MAX_TREE_SIZE * sizeof(Point));
    cudaMalloc(&d_samples, sample_count * sizeof(Point));
    cudaMalloc(&d_states, sample_count * sizeof(curandState));
    cudaMalloc(&d_nearest_indices, sample_count * sizeof(int));
    cudaMalloc(&d_collision_free, sample_count * sizeof(int));

    
    // Initialize random states
    initRandomStates<<<(sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, time(0));

    // counter to keep track of the number of iterations
    int iterations = 0;

    // Main loop
    while (tree.size() < MAX_TREE_SIZE) {
        iterations++;

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

        collisionCheck<<<(sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_samples,
            d_tree,
            d_nearest_indices,
            d_collision_free,
            sample_count
        );


        // print all the collision free values

        std::vector<int> collision_free(sample_count);
        cudaMemcpy(collision_free.data(), d_collision_free, sample_count * sizeof(int), cudaMemcpyDeviceToHost);

        // for (int i = 0; i < sample_count; ++i) {
        //     std::cout << "Collision free: " << collision_free[i] << std::endl;
        // }

        // Allocate memory for new nodes indicator
        int* new_nodes = new int[sample_count];

        // Extend the tree on the host
        std::vector<Point> samples(sample_count);
        cudaMemcpy(samples.data(), d_samples, sample_count * sizeof(Point), cudaMemcpyDeviceToHost);

        std::vector<int> nearest_indices(sample_count);
        cudaMemcpy(nearest_indices.data(), d_nearest_indices, sample_count * sizeof(int), cudaMemcpyDeviceToHost);


        const float step_size = 0.5f;
        for (int i = 0; i < sample_count; ++i) {
            if (collision_free[i]) {
            Point& nearest = tree[nearest_indices[i]];
            const Point& sample = samples[i];
            float dist = nearest.distance(sample);
            
            if (dist > 0.0f) {
                Point new_point;
                for (int j = 0; j < SPACE_DIM; ++j) {
                new_point.coords[j] = nearest.coords[j] + 
                    (sample.coords[j] - nearest.coords[j]) * (step_size / dist);
                }
                new_point.parent_idx = nearest_indices[i];  // Set parent index
                tree.push_back(new_point);
                new_nodes[i] = 1;
            } else {
                Point new_point = nearest;
                new_point.parent_idx = nearest_indices[i];  // Set parent index
                tree.push_back(new_point);
                new_nodes[i] = 1;
            }
            } else {
            new_nodes[i] = 0;
            }
        }

        delete[] new_nodes;

        // Free the memory for new nodes indicator
        cudaFree(new_nodes);

        // Check if we have reached the goal
        // Find the nearest node to the goal among all nodes
        float min_dist = FLT_MAX;
        int idx = -1;
        for (int i = 0; i < tree.size(); i++) {
            float dist = tree[i].distance(Point(7.2f, 4.1f));
            if (dist < min_dist) {
            min_dist = dist;
            idx = i;
            }
        }

        // std::cout << "Iteration: " << iterations << "    Min Distance: " << min_dist << std::endl;

        if (min_dist < 0.1f) {
            std::cout << "Goal reached!" << std::endl;
            // print the path
            printPath(tree, idx);

            break;
        }
    }

    std::cout << "Tree size: " << tree.size() << std::endl;
    if (tree.size() < MAX_TREE_SIZE) {
        cudaFree(d_tree);
        // cudaFree(d_new_nodes);
    } else {
        std::cout << "Goal not reached!" << std::endl;

        // find the nearest node to the goal
        float min_dist = FLT_MAX;
        int nearest_idx = -1;
        for (int i = 0; i < tree.size(); ++i) {
            float dist = tree[i].distance(Point(7.2f, 4.1f));
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        // print the path
        std::cout << "Nearest node: " << tree[nearest_idx].coords[0] << ", " << tree[nearest_idx].coords[0] << "    Distance: " << min_dist << std::endl;

    }

    // free the memory
    cudaFree(d_tree);
    cudaFree(d_samples);
    cudaFree(d_states);
    cudaFree(d_nearest_indices);
    cudaFree(d_collision_free);


    return 0;
}

