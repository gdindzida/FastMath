#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tensor.hpp"

template <typename T>
class TensorTest : public testing::Test {};

using TestTypes = ::testing::Types<double, float, int16_t, uint16_t, int32_t,
                                   uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(TensorTest, TestTypes);

TYPED_TEST(TensorTest, TestCreateCopyMoveToString) {
  // Arrange
  const auto tensor_dimensions = {5u};
  const uint32_t test_number1 = 5u;
  const uint32_t test_number2 = 6u;
  const auto tensor_position = {2u};

  // Act
  utils::Tensor<TypeParam> t1(tensor_dimensions);
  t1.at(tensor_position) = static_cast<TypeParam>(test_number1);

  auto t2 = t1;
  t2.at(tensor_position) = static_cast<TypeParam>(test_number2);

  auto t3 = std::move(t1);

  // Assert
  const auto zero_str = std::to_string(static_cast<TypeParam>(0u));
  const auto num1_str = std::to_string(static_cast<TypeParam>(test_number1));
  const auto num2_str = std::to_string(static_cast<TypeParam>(test_number2));

  EXPECT_EQ("", t1.as_string());
  EXPECT_EQ(zero_str + "," + zero_str + "," + num2_str + "," + zero_str + "," +
                zero_str + ",",
            t2.as_string());
  EXPECT_EQ(zero_str + "," + zero_str + "," + num1_str + "," + zero_str + "," +
                zero_str + ",",
            t3.as_string());
}

TYPED_TEST(TensorTest, TestAssertDimensionsAreEqualCorrect) {
  // Arrange
  const auto tensor_dimensions = {5u, 2u};
  const uint32_t test_number = 5u;
  const auto tensor_position = {2u, 1u};

  // Act
  utils::Tensor<TypeParam> t1(tensor_dimensions);
  t1.at(tensor_position) = static_cast<TypeParam>(test_number);

  // Assert
  EXPECT_EQ(static_cast<TypeParam>(test_number), t1.at(tensor_position));
}

TYPED_TEST(TensorTest, TestAssertDimensionsAreEqualIncorrect) {
  // Arrange
  const auto tensor_dimensions = {5u, 2u};
  const uint32_t test_number = 5u;
  const auto tensor_position = {2u, 1u, 1u};

  // Act & Assert
  utils::Tensor<TypeParam> t1(tensor_dimensions);

  EXPECT_DEATH(t1.at(tensor_position) = static_cast<TypeParam>(test_number),
               ".*Tensor dimensions and coordinates size do not match.*");
}

TYPED_TEST(TensorTest, TestTensorDimensionsAreCorrect) {
  // Arrange
  const auto tensor_dimensions = {5u, 2u, 7u};

  // Act
  utils::Tensor<TypeParam> t1(tensor_dimensions);

  // Assert
  const auto got_tensor_dimensions = t1.get_dims();
  const std::vector<uint32_t> expected_tensor_dimensions = tensor_dimensions;
  EXPECT_EQ(got_tensor_dimensions.size(), tensor_dimensions.size());
  EXPECT_EQ(got_tensor_dimensions[0], expected_tensor_dimensions[0]);
  EXPECT_EQ(got_tensor_dimensions[1], expected_tensor_dimensions[1]);
  EXPECT_EQ(got_tensor_dimensions[2], expected_tensor_dimensions[2]);
}