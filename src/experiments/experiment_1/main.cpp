#include <benchmark/benchmark.h>

#include <iostream>
#include <vector>

#include "gemm.hpp"
#include "matrix.hpp"

using Matrix = utils::Matrix<algorithms::fp>;

constexpr size_t n1 = 50ul;
constexpr size_t n2 = 60ul;
constexpr size_t n3 = 90ul;

struct BM_Data {
  BM_Data() : a(n1, n2), b(n2, n3), c(n1, n3) {
    a.random_init(0, 5);
    b.random_init(0, 5);
  }

  Matrix a;
  Matrix b;
  Matrix c;
};

static void BM_Naive(benchmark::State& state) {
  // Setup
  BM_Data data{};

  // Benchmark
  for (auto _ : state) {
    algorithms::gemm_naive(data.a, data.b, data.c);
  }
}

static void BM_Transpose(benchmark::State& state) {
  // Setup
  BM_Data data{};

  // Benchmark
  for (auto _ : state) {
    algorithms::gemm_transpose(data.a, data.b, data.c);
  }
}

BENCHMARK(BM_Naive);

BENCHMARK(BM_Transpose);

BENCHMARK_MAIN();
