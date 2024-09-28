#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "matrix.hpp"

template <typename T>
class MatrixTest : public testing::Test {};

using TestTypes = ::testing::Types<double, float, int16_t, uint16_t, int32_t,
                                   uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(MatrixTest, TestTypes);

TYPED_TEST(MatrixTest, TestCreateCopyMoveToString) {
  // Arrange
  const auto matrix_row_size = 3u;
  const auto matrix_col_size = 2u;
  const uint32_t test_number1 = 5u;
  const uint32_t test_number2 = 6u;
  const auto matrix_row = 2u;
  const auto matrix_col = 1u;

  // Act
  utils::Matrix<TypeParam> m1(matrix_row_size, matrix_col_size);
  m1.at(matrix_row, matrix_col) = static_cast<TypeParam>(test_number1);

  auto m2 = m1;
  m2.at(matrix_row, matrix_col) = static_cast<TypeParam>(test_number2);

  auto m3 = std::move(m1);

  // Assert
  const auto zero_str = std::to_string(static_cast<TypeParam>(0u));
  const auto num1_str = std::to_string(static_cast<TypeParam>(test_number1));
  const auto num2_str = std::to_string(static_cast<TypeParam>(test_number2));

  EXPECT_EQ("", m1.as_string());
  EXPECT_EQ(zero_str + "," + zero_str + "," + zero_str + "," + zero_str + "," +
                zero_str + "," + num2_str + ",",
            m2.as_string());
  EXPECT_EQ(zero_str + "," + zero_str + "," + zero_str + "," + zero_str + "," +
                zero_str + "," + num1_str + ",",
            m3.as_string());
}

TYPED_TEST(MatrixTest, TestTransposeIsCorrect) {
  // Arrange
  size_t n1 = 3ul;
  size_t n2 = 2ul;

  utils::Matrix<TypeParam> a(n1, n2);
  a.at(0, 0) = 1;
  a.at(0, 1) = 0;
  a.at(1, 0) = 0;
  a.at(1, 1) = 3;
  a.at(2, 0) = 4;
  a.at(2, 1) = 5;

  utils::Matrix<TypeParam> expected_aT(n2, n1);
  expected_aT.at(0, 0) = 1;
  expected_aT.at(0, 1) = 0;
  expected_aT.at(0, 2) = 4;
  expected_aT.at(1, 0) = 0;
  expected_aT.at(1, 1) = 3;
  expected_aT.at(1, 2) = 5;

  // Act
  a.transpose();

  // Assert
  EXPECT_EQ(expected_aT.get_row_size(), a.get_row_size());
  EXPECT_EQ(expected_aT.get_col_size(), a.get_col_size());
  EXPECT_EQ(expected_aT.at(0, 0), a.at(0, 0));
  EXPECT_EQ(expected_aT.at(0, 1), a.at(0, 1));
  EXPECT_EQ(expected_aT.at(0, 2), a.at(0, 2));
  EXPECT_EQ(expected_aT.at(1, 0), a.at(1, 0));
  EXPECT_EQ(expected_aT.at(1, 1), a.at(1, 1));
  EXPECT_EQ(expected_aT.at(1, 2), a.at(1, 2));
}