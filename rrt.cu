/*
This is my second CUDA program.
We will be implementing RRT(Rapidly-exploring Random Tree) in CUDA.
This is a path planning algorithm that is used in robotics. Read more
about it here: https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree

We will be implementing this algorithm to work in an N-dimensional(SPACE_DIM in this case) space.

See an example visualization of CPU(red) vs CUDA(blue) paths here(local dir): rrt_paths.svg
- the cuda path is pretty much perfect because it can generate so many samples and there is always
  one that is basically perfectly in the direction of the goal. This will change with more obstacles.

Still WIP

*/

#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
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

std::vector<Point> printPath(const std::vector<Point>& tree, int goal_idx) {
    std::vector<int> path;
    std::vector<Point> path_points;
    std::vector<Point> path_points_reversed;
    int current_idx = goal_idx;
    
    while (current_idx != -1) {
        path.push_back(current_idx);
        path_points.push_back(tree[current_idx]);
        current_idx = tree[current_idx].parent_idx;
    }

    std::cout << "Path from start to goal:" << std::endl;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        path_points_reversed.push_back(tree[*it]);
        std::cout << "(" << tree[*it].coords[0] << ", " 
                 << tree[*it].coords[1] << ")" << std::endl;
    }

    return path_points_reversed;
}

void generateSVG(const std::vector<Point>& cuda_path, const std::vector<Point>& cpu_path) {
    std::ofstream svg("rrt_paths.svg");
    svg << "<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='1000'>" << std::endl;
    
    // background
    svg << "<rect width='1000' height='1000' fill='white'/>" << std::endl;
    
    // black to represent borders and in the future obstacles(TODO)
    svg << "<rect width='1000' height='1000' fill='none' stroke='black' stroke-width='2'/>" << std::endl;
    
    // CUDA path (blue)
    for (size_t i = 1; i < cuda_path.size(); ++i) {
        int x1 = cuda_path[i-1].coords[0] * 100;
        int y1 = 1000 - (cuda_path[i-1].coords[1] * 100);
        int x2 = cuda_path[i].coords[0] * 100;
        int y2 = 1000 - (cuda_path[i].coords[1] * 100);
        
        svg << "<line x1='" << x1 << "' y1='" << y1 
            << "' x2='" << x2 << "' y2='" << y2 
            << "' stroke='blue' stroke-width='2'/>" << std::endl;
        
        svg << "<circle cx='" << x1 << "' cy='" << y1 
            << "' r='5' fill='blue'/>" << std::endl;
    }
    
    // CPU path (red)
    for (size_t i = 1; i < cpu_path.size(); ++i) {
        int x1 = cpu_path[i-1].coords[0] * 100;
        int y1 = 1000 - (cpu_path[i-1].coords[1] * 100);
        int x2 = cpu_path[i].coords[0] * 100;
        int y2 = 1000 - (cpu_path[i].coords[1] * 100);
        
        svg << "<line x1='" << x1 << "' y1='" << y1 
            << "' x2='" << x2 << "' y2='" << y2 
            << "' stroke='red' stroke-width='2'/>" << std::endl;
        
        svg << "<circle cx='" << x1 << "' cy='" << y1 
            << "' r='5' fill='red'/>" << std::endl;
    }
    
    // start and end points (green)
    svg << "<circle cx='0' cy='1000' r='10' fill='green'/>" << std::endl;
    svg << "<circle cx='720' cy='600' r='10' fill='green'/>" << std::endl;
    
    svg << "</svg>" << std::endl;
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

    const float step_size = 0.1f;

    std::vector<Point> cuda_path;

    const int sample_count = 512;

    // malloc GPU memory for each of these
    cudaMalloc(&d_tree, MAX_TREE_SIZE * sizeof(Point));
    cudaMalloc(&d_samples, sample_count * sizeof(Point));
    cudaMalloc(&d_states, sample_count * sizeof(curandState));
    cudaMalloc(&d_nearest_indices, sample_count * sizeof(int));
    cudaMalloc(&d_collision_free, sample_count * sizeof(int));

    
    // start the cuda timer
    cudaEvent_t start_cuda;
    cudaEvent_t stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda, 0);


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
            cuda_path = printPath(tree, idx);

            break;
        }
    }

    // stop the timer
    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start_cuda, stop_cuda);


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

    // start the timer for CPU version
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // cpu version of the same code
    // RRT cpu version:
    std::vector<Point> cpu_tree;
    cpu_tree.emplace_back(0.0f, 0.0f); // Start at the origin

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);

    int cpu_iterations = 0;

    std::vector<Point> cpu_path;


    while (cpu_tree.size() < MAX_TREE_SIZE) {
        cpu_iterations++;
        
        // Sample random point
        Point random_point;
        for (int i = 0; i < SPACE_DIM; ++i) {
            random_point.coords[i] = dis(gen);
        }
        
        // Find nearest neighbor
        float min_dist = FLT_MAX;
        int nearest_idx = -1;
        for (int i = 0; i < cpu_tree.size(); ++i) {
            float dist = random_point.distance(cpu_tree[i]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = i;
            }
        }
        
        // Create new node
        Point new_point;
        if (min_dist > 0.0f) {
            for (int i = 0; i < SPACE_DIM; ++i) {
                new_point.coords[i] = cpu_tree[nearest_idx].coords[i] + 
                    (random_point.coords[i] - cpu_tree[nearest_idx].coords[i]) * (step_size / min_dist);
            }
            new_point.parent_idx = nearest_idx;
            
            // Simple collision check
            bool collision_free = true;
            for (int i = 0; i < SPACE_DIM; ++i) {
                if (new_point.coords[i] < 0.0f || new_point.coords[i] > 10.0f) {
                    collision_free = false;
                    break;
                }
            }
            
            if (collision_free) {
                cpu_tree.push_back(new_point);
                
                // Check if goal reached
                float dist_to_goal = new_point.distance(Point(7.2f, 4.1f));
                if (dist_to_goal < 0.1f) {
                    std::cout << "CPU: Goal reached in " << cpu_iterations << " iterations!" << std::endl;
                    cpu_path = printPath(cpu_tree, cpu_tree.size() - 1);
                    break;
                }
            }
        }
    }

    // stop the timer for CPU version
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);

    std::cout << "Tree size: " << cpu_tree.size() << std::endl;
    if (!(cpu_tree.size() < MAX_TREE_SIZE)) {
        std::cout << "Goal not reached!" << std::endl;

        // find the nearest node to the goal
        float min_dist = FLT_MAX;
        int nearest_idx = -1;
        for (int i = 0; i < cpu_tree.size(); ++i) {
            float dist = cpu_tree[i].distance(Point(7.2f, 4.1f));
            if (dist < min_dist) {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        // print the path
        std::cout << "Nearest node: " << cpu_tree[nearest_idx].coords[0] << ", " << cpu_tree[nearest_idx].coords[0] << "    Distance: " << min_dist << std::endl;

    }


    // print the times:
    std::cout << "CUDA Time: " << iterations << " iterations    " << cuda_time << "ms"  << "    Speedup: " << cpu_time / cuda_time << std::endl;
    std::cout << "CPU Time: " << cpu_iterations << " iterations    " << cpu_time << "ms"  << "    Speedup: " << cpu_time / cpu_time << std::endl;


    // visualize the paths
    generateSVG(cuda_path, cpu_path);

    // I tried getting opencv to work for a while but it would fail to compile
    // and so I just wrote a method to generate an svg file instead

    return 0;
}

