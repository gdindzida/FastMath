#include <benchmark/benchmark.h>
#include <cblas.h>

#include <iostream>
#include <vector>

#include "gemm.hpp"
#include "matrix.hpp"

using Matrix = utils::Matrix<algorithms::fp>;

constexpr size_t n1 = 1024ul;
constexpr size_t n2 = 1024ul;
constexpr size_t n3 = 1024ul;

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

static void BM_Block(benchmark::State& state) {
  // Setup
  BM_Data data{};
  const size_t block_size = 32;

  // Benchmark
  for (auto _ : state) {
    algorithms::gemm_block(data.a, data.b, data.c, block_size);
  }
}

static void BM_Super(benchmark::State& state) {
  // Setup
  BM_Data data{};

  // Benchmark
  for (auto _ : state) {
    algorithms::gemm_super(data.a, data.b, data.c);
  }
}

static void BM_Blas(benchmark::State& state) {
  // Setup
  BM_Data data{};

  // Benchmark
  for (auto _ : state) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n3, n2, 1.0,
                &(data.a.at(0, 0)), n2, &(data.b.at(0, 0)), n3, 0.0,
                &(data.c.at(0, 0)), n3);
  }
}

constexpr int32_t number_of_iterations = 30;

BENCHMARK(BM_Naive)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(number_of_iterations);

BENCHMARK(BM_Transpose)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(number_of_iterations);

BENCHMARK(BM_Block)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(number_of_iterations);

BENCHMARK(BM_Super)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(number_of_iterations);

BENCHMARK(BM_Blas)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(number_of_iterations);

BENCHMARK_MAIN();
