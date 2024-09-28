#include "gemm.hpp"

#include <iostream>

namespace algorithms {

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
}  // namespace

void gemm_naive(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c) {
  assert_matrix_dimensions(a, b, c);

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

  static utils::Matrix<fp> b_copy(b.get_col_size(), b.get_row_size());
  b.transpose_to(b_copy);

  for (size_t i = 0; i < a.get_row_size(); ++i) {
    for (size_t j = 0; j < b.get_col_size(); ++j) {
      c.at(i, j) = static_cast<algorithms::fp>(0);

      for (size_t k = 0; k < a.get_col_size(); ++k) {
        c.at(i, j) += a.at(i, k) * b_copy.at(j, k);
      }
    }
  }
}

}  // namespace algorithms
