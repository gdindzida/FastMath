#include <iostream>
#include <vector>

#include "gemm.hpp"
#include "tensor.hpp"
#include "utils.hpp"

using Tensor = utils::Tensor<float>;

int main() {
  std::vector<int> nums{1, 2, 3, 4, 5};

  // Range-based for loop (C++20 feature)
  for (auto num : nums) {
    std::cout << num << ' ';
  }
  std::cout << std::endl;

  utils::print_lib_name();
  algorithms::print_lib_name();

  std::cout << "Tensor testing..." << std::endl;

  Tensor t({10});

  t.at({5}) = 5;

  t.print();

  Tensor t2(t);

  t2.at({5}) = 6;

  t.print();
  t2.print();

  Tensor t3(std::move(t));

  t.print();
  t2.print();
  t3.print();

  t3.at({1, 2});

  return 0;
}