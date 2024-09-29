#include "gemm.hpp"

#include <iostream>

namespace algorithms {

#define BLOCK_SIZE (16ul)

namespace {
void assert_matrix_dimensions(const utils::Matrix<fp>& a,
                              const utils::Matrix<fp>& b,
                              const utils::Matrix<fp>& c) {
  assert(a.get_col_size() == b.get_row_size() &&
         "Dimensions of input matrices do not match");
  assert(a.get_row_size() == c.get_row_size() &&
         b.get_col_size() == c.get_col_size() &&
         "Output matrix dimensions do not match input matrices");
}

void print_matrix(const fp mat[BLOCK_SIZE][BLOCK_SIZE]) {
  for (size_t i = 0; i < BLOCK_SIZE; ++i) {
    for (size_t j = 0; j < BLOCK_SIZE; ++j) {
      std::cout << mat[i][j] << ",";
    }
    std::cout << std::endl;
  }
}

}  // namespace

void gemm_naive(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c) {
  assert_matrix_dimensions(a, b, c);

  c.zero_init();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < a.get_row_size(); ++i) {
    for (size_t j = 0; j < b.get_col_size(); ++j) {
      c.at(i, j) = static_cast<algorithms::fp>(0);

      for (size_t k = 0; k < a.get_col_size(); ++k) {
        c.at(i, j) += a.at(i, k) * b.at(k, j);
      }
    }
  }
}

void gemm_transpose(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                    utils::Matrix<fp>& c) {
  assert_matrix_dimensions(a, b, c);

  c.zero_init();
  static utils::Matrix<fp> b_copy(b.get_col_size(), b.get_row_size());
  b.transpose_to(b_copy);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < a.get_row_size(); ++i) {
    for (size_t j = 0; j < b.get_col_size(); ++j) {
      c.at(i, j) = static_cast<algorithms::fp>(0);

      for (size_t k = 0; k < a.get_col_size(); ++k) {
        c.at(i, j) += a.at(i, k) * b_copy.at(j, k);
      }
    }
  }
}

void gemm_block(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c, const size_t block_size) {
  assert_matrix_dimensions(a, b, c);

  c.zero_init();
  static utils::Matrix<fp> b_copy(b.get_col_size(), b.get_row_size());
  b.transpose_to(b_copy);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < a.get_row_size(); i += block_size) {
    for (size_t j = 0; j < b.get_col_size(); j += block_size) {
      for (size_t k = 0; k < a.get_col_size(); k += block_size) {
        size_t block_a_rows_size = std::min(block_size, a.get_row_size() - i);
        size_t block_a_cols_size = std::min(block_size, a.get_col_size() - k);
        size_t block_b_cols_size = std::min(block_size, b.get_col_size() - j);

        for (size_t ii = 0; ii < block_a_rows_size; ++ii) {
          for (size_t jj = 0; jj < block_b_cols_size; ++jj) {
            fp partial_sum = static_cast<fp>(0);

            for (size_t kk = 0; kk < block_a_cols_size; ++kk) {
              partial_sum += a.at(i + ii, k + kk) * b_copy.at(j + jj, k + kk);
            }

            c.at(i + ii, j + jj) += partial_sum;
          }
        }
      }
    }
  }
}

alignas(64) fp local_a[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) fp local_b[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) fp local_c[BLOCK_SIZE][BLOCK_SIZE];

#ifdef _OPENMP
#pragma omp threadprivate(local_a, local_b, local_c)
#endif

void gemm_super(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c) {
  assert_matrix_dimensions(a, b, c);

  c.zero_init();
  static utils::Matrix<fp> b_copy(b.get_col_size(), b.get_row_size());
  b.transpose_to(b_copy);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < a.get_row_size(); i += BLOCK_SIZE) {
    for (size_t j = 0; j < b.get_col_size(); j += BLOCK_SIZE) {
      for (size_t ii = 0; ii < BLOCK_SIZE; ++ii) {
        for (size_t jj = 0; jj < BLOCK_SIZE; ++jj) {
          local_c[ii][jj] = static_cast<fp>(0);
        }
      }

      for (size_t k = 0; k < a.get_col_size(); k += BLOCK_SIZE) {
        size_t block_a_rows_size = std::min(BLOCK_SIZE, a.get_row_size() - i);
        size_t block_a_cols_size = std::min(BLOCK_SIZE, a.get_col_size() - k);
        size_t block_b_cols_size = std::min(BLOCK_SIZE, b.get_col_size() - j);

        for (size_t ii = 0; ii < block_a_rows_size; ++ii) {
          for (size_t jj = 0; jj < block_a_cols_size; ++jj) {
            local_a[ii][jj] = a.at(i + ii, j + jj);
          }
        }

        for (size_t ii = 0; ii < block_b_cols_size; ++ii) {
          for (size_t jj = 0; jj < block_a_cols_size; ++jj) {
            local_b[ii][jj] = b_copy.at(i + ii, j + jj);
          }
        }

        for (size_t ii = 0; ii < block_a_rows_size; ++ii) {
          for (size_t jj = 0; jj < block_b_cols_size; ++jj) {
            for (size_t kk = 0; kk < block_a_cols_size; ++kk) {
              local_c[ii][jj] += local_a[ii][kk] * local_b[jj][kk];
            }

            c.at(i + ii, j + jj) += local_c[ii][jj];
          }
        }
      }
    }
  }
}

}  // namespace algorithms
