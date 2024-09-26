#include <iostream>
#include <vector>

#include "utils.hpp"

int main() {
  std::vector<int> nums{1, 2, 3, 4, 5};

  // Range-based for loop (C++20 feature)
  for (auto num : nums) {
    std::cout << num << ' ';
  }
  std::cout << std::endl;

  print_lib_name();

  return 0;
}