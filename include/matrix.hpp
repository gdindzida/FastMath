#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace utils {

template <typename T>
class Matrix {
 public:
  Matrix(const size_t row_size, const size_t col_size) {
    initialize(row_size, col_size);
  }

  Matrix(const Matrix& other) {
    initialize(other._row_size, other._col_size);
    std::copy(other._data.get(), other._data.get() + other._mem_size,
              _data.get());
  }

  Matrix(Matrix&& other) noexcept {
    _mem_size = other._mem_size;
    _row_size = other._row_size;
    _col_size = other._col_size;
    _data = std::move(other._data);

    other._mem_size = 0;
    other._row_size = 0;
    other._col_size = 0;
  }

  ~Matrix() = default;

  T& at(const size_t& row, const size_t& col) const {
    assert(row < _row_size && "Row is out of bounds");
    assert(col < _col_size && "Column is out of bounds");
    size_t index = row * _col_size + col;

    return _data.get()[index];
  }

  std::string as_string() const {
    std::string output = "";
    for (size_t index = 0u; index < _mem_size; ++index)
      output += std::to_string(_data.get()[index]) + ",";
    return output;
  }

  const size_t& get_row_size() const { return _row_size; }
  const size_t& get_col_size() const { return _col_size; }

  void transpose() {
    Matrix temp = *this;
    size_t row_size_temp = _row_size;
    size_t col_size_temp = _col_size;
    std::swap(_row_size, _col_size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < col_size_temp; ++j) {
      for (size_t i = 0; i < row_size_temp; ++i) {
        this->at(j, i) = temp.at(i, j);
      }
    }
  }

  void transpose_to(Matrix& dest) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < _col_size; ++j) {
      for (size_t i = 0; i < _row_size; ++i) {
        dest.at(j, i) = this->at(i, j);
      }
    }
  }

  void random_init(const int32_t min, const int32_t max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);

    for (size_t i = 0; i < _mem_size; ++i) {
      int32_t random_number = dis(gen);
      _data.get()[i] = static_cast<T>(random_number);
    }
  }

 private:
  size_t _mem_size;
  size_t _row_size;
  size_t _col_size;
  std::unique_ptr<T[]> _data;

  void initialize(const size_t row_size, const size_t col_size) {
    _row_size = row_size;
    _col_size = col_size;
    _mem_size = _row_size * _col_size;
    _data = std::make_unique<T[]>(_mem_size);
  }
};

}  // namespace utils

#endif  // MATRIX_HPP