#include <benchmark/benchmark.h>

#include <iostream>
#include <vector>

#include "gemm.hpp"
#include "tensor.hpp"
#include "utils.hpp"

using Tensor = utils::Tensor<float>;

void my_experiment() {}

static void BM_Function(benchmark::State& state) {
  for (auto _ : state) {
    my_experiment();
  }
}

BENCHMARK(BM_Function);

BENCHMARK_MAIN();
