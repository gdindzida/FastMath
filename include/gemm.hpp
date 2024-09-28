#ifndef GEMM_HPP
#define GEMM_HPP

#include "matrix.hpp"

namespace algorithms {

using fp = float;

void gemm_naive(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                utils::Matrix<fp>& c);

void gemm_transpose(const utils::Matrix<fp>& a, const utils::Matrix<fp>& b,
                    utils::Matrix<fp>& c);

}  // namespace algorithms

#endif  // GEMM_HPP