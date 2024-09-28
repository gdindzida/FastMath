#include <benchmark/benchmark.h>

#include <iostream>
#include <vector>

#include "gemm.hpp"
#include "matrix.hpp"

using Matrix = utils::Matrix<algorithms::fp>;

constexpr size_t n1 = 1000ul;
constexpr size_t n2 = 1000ul;
constexpr size_t n3 = 1000ul;

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

constexpr int32_t number_of_iterations = 3;

BENCHMARK(BM_Naive)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(number_of_iterations);

BENCHMARK(BM_Transpose)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(number_of_iterations);

BENCHMARK_MAIN();
