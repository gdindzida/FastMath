#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gemm.hpp"
#include "matrix.hpp"

using Matrix = utils::Matrix<algorithms::fp>;

TEST(GemmTest, TestMatMul) {
  // Arrange
  size_t n1 = 3ul;
  size_t n2 = 2ul;
  size_t n3 = 4ul;

  Matrix a(n1, n2);
  a.at(0, 0) = 1;
  a.at(0, 1) = 0;
  a.at(1, 0) = 0;
  a.at(1, 1) = 3;
  a.at(2, 0) = 4;
  a.at(2, 1) = 5;

  Matrix b(n2, n3);
  b.at(0, 0) = 2;
  b.at(0, 1) = 7;
  b.at(0, 2) = 1;
  b.at(0, 3) = 4;
  b.at(1, 0) = 3;
  b.at(1, 1) = 0;
  b.at(1, 2) = 1;
  b.at(1, 3) = 3;

  Matrix c(n1, n3);

  Matrix expected_c(n1, n3);
  expected_c.at(0, 0) = 2;
  expected_c.at(0, 1) = 7;
  expected_c.at(0, 2) = 1;
  expected_c.at(0, 3) = 4;
  expected_c.at(1, 0) = 9;
  expected_c.at(1, 1) = 0;
  expected_c.at(1, 2) = 3;
  expected_c.at(1, 3) = 9;
  expected_c.at(2, 0) = 23;
  expected_c.at(2, 1) = 28;
  expected_c.at(2, 2) = 9;
  expected_c.at(2, 3) = 31;

  // Act
  algorithms::gemm_naive(a, b, c);

  // Assert
  EXPECT_EQ(expected_c.at(0, 0), c.at(0, 0));
  EXPECT_EQ(expected_c.at(0, 1), c.at(0, 1));
  EXPECT_EQ(expected_c.at(0, 2), c.at(0, 2));
  EXPECT_EQ(expected_c.at(0, 3), c.at(0, 3));
  EXPECT_EQ(expected_c.at(1, 0), c.at(1, 0));
  EXPECT_EQ(expected_c.at(1, 1), c.at(1, 1));
  EXPECT_EQ(expected_c.at(1, 2), c.at(1, 2));
  EXPECT_EQ(expected_c.at(1, 3), c.at(1, 3));
  EXPECT_EQ(expected_c.at(2, 0), c.at(2, 0));
  EXPECT_EQ(expected_c.at(2, 1), c.at(2, 1));
  EXPECT_EQ(expected_c.at(2, 2), c.at(2, 2));
  EXPECT_EQ(expected_c.at(2, 3), c.at(2, 3));
}