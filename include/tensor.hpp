#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace utils {

template <typename T>
class Tensor {
 public:
  Tensor(const std::initializer_list<uint32_t>& dim_sizes) {
    initialize(dim_sizes);
  }

  Tensor(const Tensor& other) {
    initialize(other._dim_sizes);
    std::copy(other._data.get(), other._data.get() + other._mem_size,
              _data.get());
  }

  Tensor(Tensor&& other) noexcept {
    _mem_size = other._mem_size;
    _dim_sizes = std::move(other._dim_sizes);
    _data = std::move(other._data);

    other._mem_size = 0;
  }

  ~Tensor() = default;

  T& at(const std::vector<uint32_t> coordinates) {
    assert(coordinates.size() == _dim_sizes.size());

    uint32_t index = 0u;
    uint32_t prod = _mem_size;
    for (uint32_t dim_index = 0u; dim_index < _dim_sizes.size(); ++dim_index) {
      auto dim_size = _dim_sizes[dim_index];
      prod = _mem_size / dim_size;

      index += coordinates[dim_index] * prod;
    }

    return _data.get()[index];
  }

  void print() {
    for (uint32_t index = 0u; index < _mem_size; ++index)
      std::cout << _data.get()[index] << ",";
    std::cout << std::endl;
  }

 private:
  uint32_t _mem_size;
  std::vector<uint32_t> _dim_sizes;
  std::unique_ptr<T[]> _data;

  void initialize(const std::vector<uint32_t>& dim_sizes) {
    _dim_sizes = dim_sizes;
    _mem_size = 1u;
    for (auto& dim_size : dim_sizes) _mem_size *= dim_size;
    _data = std::make_unique<T[]>(_mem_size);
  }
};

}  // namespace utils

#endif  // TENSOR_HPP