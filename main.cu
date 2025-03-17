#include <fast_matrix_market/fast_matrix_market.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

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


    return 0;
}
