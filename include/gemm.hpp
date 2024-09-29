#ifndef GEMM_HPP
#define GEMM_HPP

#include "matrix.hpp"

namespace algorithms {

using fp = double;

void gemm_naive(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c);

void gemm_transpose(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                    utils::Matrix<fp>& c);

void gemm_block(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c, const size_t block_size);

void gemm_super(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c);

}  // namespace algorithms

#endif  // GEMM_HPP