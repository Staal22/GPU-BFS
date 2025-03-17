#include <fast_matrix_market/fast_matrix_market.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
        << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct triplet_matrix
{
    int64_t nrows = 0, ncols = 0;
    std::vector<int64_t> rows, cols;
    std::vector<double> vals;
};

struct csr_matrix
{
    int64_t nrows = 0, ncols = 0;
    std::vector<int64_t> row_ptr;
    std::vector<int64_t> cols;
    std::vector<double> vals;
};

struct device_csr_matrix
{
    int64_t nrows, ncols;
    int64_t* row_ptr;
    int64_t* cols;
    double* vals;
};

__global__ void bfs_kernel(
    const int64_t* row_ptr,
    const int64_t* cols,
    const double* vals,
    int level,
    int* current_frontier,
    int* next_frontier,
    int* visited,
    int* levels,
    int frontier_size,
    int* next_frontier_size
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < frontier_size)
    {
        int node = current_frontier[tid];

        // Process all neighbors of this node
        int start = row_ptr[node];
        int end = row_ptr[node + 1];

        for (int edge = start; edge < end; edge++)
        {
            int neighbor = cols[edge];

            // If the neighbor hasn't been visited yet
            if (atomicCAS(&visited[neighbor], 0, 1) == 0)
            {
                // Mark the level of this neighbor
                levels[neighbor] = level;

                // Add to the next frontier
                int idx = atomicAdd(next_frontier_size, 1);
                next_frontier[idx] = neighbor;
            }
        }
    }
}

csr_matrix csr_from_coo(const triplet_matrix& coo)
{
    csr_matrix csr;
    csr.nrows = coo.nrows;
    csr.ncols = coo.ncols;
    int64_t nnz = coo.vals.size();

    // Initialize row_ptr with zeros
    csr.row_ptr.assign(csr.nrows + 1, 0);

    // Count the number of entries in each row
    for (int64_t i = 0; i < nnz; ++i)
    {
        if (coo.rows[i] < 0 || coo.rows[i] >= csr.nrows)
        {
            throw std::out_of_range("Row index out of bounds during conversion.");
        }
        csr.row_ptr[coo.rows[i] + 1]++;
    }

    // Cumulative sum to get row_ptr
    for (int64_t i = 0; i < csr.nrows; ++i)
    {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    // Allocate space for cols and vals
    csr.cols.resize(nnz);
    csr.vals.resize(nnz);

    // Temporary copy of row_ptr to keep track of the current position in each row
    std::vector<int64_t> current_pos = csr.row_ptr;

    // Populate cols and vals
    for (int64_t i = 0; i < nnz; ++i)
    {
        int64_t row = coo.rows[i];
        int64_t dest = current_pos[row]++;
        csr.cols[dest] = coo.cols[i];
        csr.vals[dest] = coo.vals[i];
    }

    // Optionally, sort the column indices within each row
    for (int64_t row = 0; row < csr.nrows; ++row)
    {
        int64_t start = csr.row_ptr[row];
        int64_t end = csr.row_ptr[row + 1];
        std::vector<std::pair<int64_t, double>> col_val_pairs;
        for (int64_t i = start; i < end; ++i)
        {
            col_val_pairs.emplace_back(csr.cols[i], csr.vals[i]);
        }
        std::sort(col_val_pairs.begin(), col_val_pairs.end(),
                  [](const std::pair<int64_t, double>& a, const std::pair<int64_t, double>& b)
                  {
                      return a.first < b.first;
                  });
        for (int64_t i = start, j = 0; i < end; ++i, ++j)
        {
            csr.cols[i] = col_val_pairs[j].first;
            csr.vals[i] = col_val_pairs[j].second;
        }
    }

    return csr;
}

device_csr_matrix copy_csr_to_device(const csr_matrix& csr)
{
    device_csr_matrix d_csr;
    d_csr.nrows = csr.nrows;
    d_csr.ncols = csr.ncols;

    // Allocate and copy row_ptr
    CUDA_CHECK(cudaMalloc(&d_csr.row_ptr, (csr.nrows + 1) * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_csr.row_ptr, csr.row_ptr.data(), (csr.nrows + 1) * sizeof(int64_t), cudaMemcpyHostToDevice))
    ;

    // Allocate and copy cols
    CUDA_CHECK(cudaMalloc(&d_csr.cols, csr.cols.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_csr.cols, csr.cols.data(), csr.cols.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Allocate and copy vals (if needed)
    CUDA_CHECK(cudaMalloc(&d_csr.vals, csr.vals.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_csr.vals, csr.vals.data(), csr.vals.size() * sizeof(double), cudaMemcpyHostToDevice));

    return d_csr;
}

void free_device_csr(device_csr_matrix& d_csr)
{
    CUDA_CHECK(cudaFree(d_csr.row_ptr));
    CUDA_CHECK(cudaFree(d_csr.cols));
    CUDA_CHECK(cudaFree(d_csr.vals));
}

std::vector<int> bfs_csr_cuda(const csr_matrix& csr, int start_node)
{
    // Validate start node
    if (start_node < 0 || start_node >= csr.nrows)
    {
        throw std::out_of_range("Start node is out of bounds");
    }

    // Copy CSR matrix to device
    device_csr_matrix d_csr = copy_csr_to_device(csr);

    // Allocate arrays for BFS
    int* d_current_frontier;
    int* d_next_frontier;
    int* d_visited;
    int* d_levels;
    int* d_frontier_size;
    int* d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_current_frontier, csr.nrows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, csr.nrows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, csr.nrows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_levels, csr.nrows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

    // Initialize visited and levels arrays
    CUDA_CHECK(cudaMemset(d_visited, 0, csr.nrows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_levels, -1, csr.nrows * sizeof(int)));

    // Initialize the first frontier with the start node
    int initial_frontier[1] = {start_node};
    int initial_frontier_size = 1;
    int zero = 0;

    CUDA_CHECK(cudaMemcpy(d_current_frontier, initial_frontier, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &initial_frontier_size, sizeof(int), cudaMemcpyHostToDevice));

    // Mark the start node as visited and set its level to 0
    int visited_init[1] = {1};
    int level_init[1] = {0};
    CUDA_CHECK(cudaMemcpy(&d_visited[start_node], visited_init, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&d_levels[start_node], level_init, sizeof(int), cudaMemcpyHostToDevice));

    // BFS traversal
    int level = 1;
    int h_frontier_size = initial_frontier_size;

    while (h_frontier_size > 0)
    {
        // Reset next frontier size
        CUDA_CHECK(cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice));

        // Set kernel dimensions
        int block_size = 256;
        int grid_size = (h_frontier_size + block_size - 1) / block_size;

        // Launch kernel
        bfs_kernel<<<grid_size, block_size>>>(
            d_csr.row_ptr,
            d_csr.cols,
            d_csr.vals,
            level,
            d_current_frontier,
            d_next_frontier,
            d_visited,
            d_levels,
            h_frontier_size,
            d_next_frontier_size
        );

        // Check for kernel errors
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Get the size of the next frontier
        CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));

        // Swap current and next frontier
        std::swap(d_current_frontier, d_next_frontier);

        level++;
    }

    // Copy results back to host
    std::vector<int> levels(csr.nrows);
    CUDA_CHECK(cudaMemcpy(levels.data(), d_levels, csr.nrows * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_current_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_levels));
    CUDA_CHECK(cudaFree(d_frontier_size));
    CUDA_CHECK(cudaFree(d_next_frontier_size));
    free_device_csr(d_csr);

    return levels;
}

std::vector<int> find_connected_components(const csr_matrix& csr)
{
    std::vector<int> components(csr.nrows, -1);
    int component_count = 0;

    for (int i = 0; i < csr.nrows; i++)
    {
        if (components[i] == -1)
        {
            // This node hasn't been visited yet
            std::vector<int> bfs_levels = bfs_csr_cuda(csr, i);

            // Mark all reachable nodes as part of this component
            for (int j = 0; j < csr.nrows; j++)
            {
                if (bfs_levels[j] != -1)
                {
                    components[j] = component_count;
                }
            }

            component_count++;
        }
    }

    return components;
}

void bfs_battery(const csr_matrix& csr, std::vector<int> sample_nodes = {0})
{
    try
    {
        std::cout << "\n==========================================\n";
        std::cout << "DETAILED BFS CONNECTIVITY ANALYSIS\n";
        std::cout << "==========================================\n\n";

        // For each sample node, run BFS and analyze results
        for (int start_node: sample_nodes)
        {
            std::cout << "\n---------------------------------------------\n";
            std::cout << "BFS Analysis from Node " << start_node + 1 << "\n";
            std::cout << "---------------------------------------------\n";

            std::vector<int> levels = bfs_csr_cuda(csr, start_node);

            // Find maximum level
            int max_level = *std::max_element(levels.begin(), levels.end());

            // Count nodes at each level
            std::vector level_counts(max_level + 1, 0);
            for (int level: levels)
            {
                if (level != -1)
                {
                    level_counts[level]++;
                }
            }

            // Print level distribution
            std::cout << "Level distribution:\n";
            for (int i = 0; i <= max_level; i++)
            {
                std::cout << "  Level " << i << ": " << level_counts[i] << " nodes";

                // Print percentage
                double percentage = 100.0 * level_counts[i] / csr.nrows;
                std::cout << " (" << std::fixed << std::setprecision(2) << percentage << "% of total)\n";
            }

            // Count unreachable nodes
            int unreachable_count = std::count(levels.begin(), levels.end(), -1);
            double unreachable_percentage = 100.0 * unreachable_count / csr.nrows;
            std::cout << "  Unreachable: " << unreachable_count << " nodes";
            std::cout << " (" << std::fixed << std::setprecision(2) << unreachable_percentage << "% of total)\n";

            // Sample some nodes from each level for detailed inspection
            std::cout << "\nSample nodes at each level:\n";

            const int SAMPLES_PER_LEVEL = 3; // Number of sample nodes to show per level

            for (int i = 0; i <= max_level; i++)
            {
                std::cout << "  Level " << i << ": ";

                // Collect nodes at this level
                std::vector<int> nodes_at_level;
                for (int j = 0; j < csr.nrows; j++)
                {
                    if (levels[j] == i)
                    {
                        nodes_at_level.push_back(j);
                    }
                }

                // Sample and print nodes
                int samples = std::min(SAMPLES_PER_LEVEL, (int) nodes_at_level.size());
                for (int j = 0; j < samples; j++)
                {
                    int node_idx = j * nodes_at_level.size() / samples;
                    int node = nodes_at_level[node_idx];

                    std::cout << node + 1;

                    // Count outgoing and incoming edges for this node
                    int outgoing = csr.row_ptr[node + 1] - csr.row_ptr[node];

                    // Incoming requires a scan through the matrix
                    int incoming = 0;
                    for (int row = 0; row < csr.nrows; row++)
                    {
                        for (int64_t j = csr.row_ptr[row]; j < csr.row_ptr[row + 1]; j++)
                        {
                            if (csr.cols[j] == node)
                            {
                                incoming++;
                                break;
                            }
                        }
                    }

                    std::cout << " (out: " << outgoing << ", in: " << incoming << ")";

                    if (j < samples - 1)
                    {
                        std::cout << ", ";
                    }
                }
                std::cout << "\n";
            }

            // Show some unreachable nodes if any
            if (unreachable_count > 0)
            {
                std::cout << "  Unreachable: ";
                int shown = 0;
                for (int j = 0; j < csr.nrows && shown < SAMPLES_PER_LEVEL; j++)
                {
                    if (levels[j] == -1)
                    {
                        if (shown > 0) std::cout << ", ";
                        std::cout << j;
                        shown++;
                    }
                }
                if (unreachable_count > SAMPLES_PER_LEVEL)
                {
                    std::cout << ", ... (" << (unreachable_count - SAMPLES_PER_LEVEL) << " more)";
                }
                std::cout << "\n";
            }

            // For the first sampled node, show paths to a few interesting nodes
            if (start_node == sample_nodes[0])
            {
                std::cout << "\nSample paths from node " << start_node + 1 << ":\n";

                // Get some sample destination nodes at different levels
                std::vector<int> sample_destinations;
                for (int level = 1; level <= max_level; level++)
                {
                    for (int j = 0; j < csr.nrows; j++)
                    {
                        if (levels[j] == level)
                        {
                            sample_destinations.push_back(j);
                            break;
                        }
                    }
                }

                // For each sample destination, reconstruct and print the path
                for (int dest: sample_destinations)
                {
                    std::cout << "  Path to node " << dest + 1 << " (level " << levels[dest] << "): ";

                    // Reconstruct path (approximate, since we don't store actual paths in BFS)
                    std::vector<int> path;
                    int current = dest;
                    path.push_back(current);

                    while (current != start_node && levels[current] > 0)
                    {
                        // Find a node at the previous level that connects to current
                        for (int j = 0; j < csr.nrows; j++)
                        {
                            if (levels[j] == levels[current] - 1)
                            {
                                // Check if j connects to current
                                for (int64_t edge = csr.row_ptr[j]; edge < csr.row_ptr[j + 1]; edge++)
                                {
                                    if (csr.cols[edge] == current)
                                    {
                                        current = j;
                                        path.push_back(current);
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // Print the path in reverse order (from source to destination)
                    for (int i = path.size() - 1; i >= 0; i--)
                    {
                        std::cout << path[i] + 1;
                        if (i > 0) std::cout << " -> ";
                    }
                    std::cout << "\n";
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in detailed BFS analysis: " << e.what() << std::endl;
    }
}


int main()
{
    triplet_matrix mat;
    try
    {
        // Open the file
        std::ifstream file("../Hamrle1.mtx");
        if (!file)
        {
            std::cerr << "Failed to open file\n";
            return 1;
        }

        // Read the matrix market file
        // For coordinate (COO) format
        fast_matrix_market::read_matrix_market_triplet(
            file,
            mat.nrows, mat.ncols,
            mat.rows, mat.cols, mat.vals
        );
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    csr_matrix csr = csr_from_coo(mat);

    // Print matrix info
    std::cout << "COO Matrix: " << mat.nrows << "x" << mat.ncols
            << " with " << mat.vals.size() << " non-zeros\n";

    std::cout << "CSR Matrix: " << csr.nrows << "x" << csr.ncols
            << " with " << csr.vals.size() << " non-zeros\n";

    // run_bfs_example(csr);
    bfs_battery(csr);

    return 0;
}
