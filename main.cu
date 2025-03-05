#include <iostream>
// #include <nvshmem.h>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <vector>
#include <fstream>

#ifdef __JETBRAINS_IDE__
// #include <host/nvshmem_api.h>
#endif

struct triplet_matrix
{
    int64_t nrows = 0, ncols = 0;
    std::vector<int64_t> rows, cols;
    std::vector<double> vals;
};

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

    // Print matrix info
    std::cout << "Matrix: " << mat.nrows << "x" << mat.ncols
            << " with " << mat.vals.size() << " non-zeros\n";

    return 0;
}
