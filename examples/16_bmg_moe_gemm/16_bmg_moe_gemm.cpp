// Copyright (C) 2026 Intel Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SYCL_INTEL_TARGET
#undef SYCL_INTEL_TARGET
#endif
#define SYCL_INTEL_TARGET 20

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include "gpt_oss_120b_workloads.hpp"
#include "kernel/moe/xe20/w4a16/gemm_xe2_policy.hpp"
#include "kernel/moe/xe20/w4a16/grouped_gemm_xe2.hpp"

namespace {

using bf16_t = cutlass::bfloat16_t;
using namespace cute;

// A direct, single-launch W4A16 baseline. The kernel sources under w4a16 are
// copied unchanged from sgl-kernel-xpu. No local bucket dispatch, heuristic,
// tile override, fused activation, diagnostic path, or compiler tuning remains.
template <typename, typename, typename, bool>
class W4A16Kernel;

template <class Policy, typename ElementS, typename ElementA, bool HasZero>
sycl::event launch_w4a16(
    sycl::queue& queue,
    const ElementA* activations,
    const uint8_t* packed_weights,
    const ElementS* scales,
    const ElementS* zeros,
    ElementA* output,
    int n, int k, const int32_t* rows, int experts, int group_size, int32_t* counter) {
  using op_t = XE_DPAS_TT<8, float, ElementA>;
  using wg_tile = typename Policy::WGTile;
  using sg_layout = typename Policy::SGLayout;
  using mma_t = typename TiledMMAHelper<MMA_Atom<op_t>, Layout<wg_tile>, sg_layout>::TiledMMA;
  const auto mma = mma_t{};
  const int threads = size(mma);
  constexpr int kThreadsPerSm = 512;
  if (kThreadsPerSm % threads) throw std::runtime_error("invalid W4A16 workgroup size");
  const int sms = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  const sycl::range<3> local(1, 1, threads);
  const sycl::range<3> global(1, sms * kThreadsPerSm / threads, 1);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<W4A16Kernel<Policy, ElementS, ElementA, HasZero>>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto) {
          moe_w4a16::MoEGEMM<void, void, void, 'R', 'C', 'R', HasZero>(
              activations, packed_weights, scales, zeros, static_cast<const float*>(nullptr),
              output, mma, rows, experts, group_size, n, k, counter, local_mem);
        });
  });
}

struct Problem {
  int experts = 8;
  int rows_per_expert = 8;
  int n = 256;
  int k = 256;
  int group_size = 32;
};

int signed_nibble(uint8_t byte, bool high) {
  const int value = high ? byte >> 4 : byte & 0x0f;
  return value < 8 ? value : value - 16;
}

bool run_accuracy(sycl::queue& queue, const Problem& p) {
  const int total_m = p.experts * p.rows_per_expert;
  const size_t a_count = size_t(total_m) * p.k;
  const size_t w_count = size_t(p.experts) * p.n * p.k / 2;
  const size_t s_count = size_t(p.experts) * p.n * (p.k / p.group_size);
  const size_t d_count = size_t(total_m) * p.n;
  std::mt19937 rng(20260825);
  std::uniform_real_distribution<float> adist(-0.25f, 0.25f), sdist(0.01f, 0.06f);
  std::uniform_int_distribution<int> qdist(-4, 4);
  std::vector<bf16_t> a(a_count), scales(s_count), d(d_count);
  std::vector<uint8_t> w(w_count);
  std::vector<int32_t> rows(p.experts, p.rows_per_expert);
  for (auto& x : a) x = bf16_t(adist(rng));
  for (auto& x : scales) x = bf16_t(sdist(rng));
  for (auto& x : w) {
    const auto low = static_cast<uint8_t>(qdist(rng)) & 0x0f;
    const auto high = static_cast<uint8_t>(qdist(rng)) & 0x0f;
    x = low | (high << 4);
  }
  auto* da = sycl::malloc_device<bf16_t>(a_count, queue);
  auto* dw = sycl::malloc_device<uint8_t>(w_count, queue);
  auto* ds = sycl::malloc_device<bf16_t>(s_count, queue);
  auto* dd = sycl::malloc_device<bf16_t>(d_count, queue);
  auto* dr = sycl::malloc_device<int32_t>(rows.size(), queue);
  auto* counter = sycl::malloc_device<int32_t>(1, queue);
  if (!da || !dw || !ds || !dd || !dr || !counter) throw std::runtime_error("allocation failed");
  queue.memcpy(da, a.data(), a_count * sizeof(bf16_t));
  queue.memcpy(dw, w.data(), w_count);
  queue.memcpy(ds, scales.data(), s_count * sizeof(bf16_t));
  queue.memcpy(dr, rows.data(), rows.size() * sizeof(int32_t));
  queue.memset(counter, 0, sizeof(int32_t)).wait();
  launch_w4a16<moe_w4a16::w4a16_policy, bf16_t, bf16_t, false>(
      queue, da, dw, ds, nullptr, dd, p.n, p.k, dr, p.experts, p.group_size, counter).wait();
  queue.memcpy(d.data(), dd, d_count * sizeof(bf16_t)).wait();
  float max_error = 0.0f;
  for (int e = 0; e < p.experts; ++e) for (int m = 0; m < p.rows_per_expert; ++m) {
    const int row = e * p.rows_per_expert + m;
    for (int n = 0; n < p.n; ++n) {
      float expected = 0.0f;
      for (int k = 0; k < p.k; ++k) {
        const size_t weight_index = (size_t(e) * p.n * p.k + size_t(n) * p.k + k) / 2;
        const size_t scale_index =
            (size_t(e) * p.n + n) * (p.k / p.group_size) + k / p.group_size;
        expected += static_cast<float>(a[size_t(row) * p.k + k]) *
            signed_nibble(w[weight_index], k & 1) * static_cast<float>(scales[scale_index]);
      }
      max_error = std::max(max_error, std::abs(expected - static_cast<float>(d[size_t(row) * p.n + n])));
    }
  }
  sycl::free(da, queue); sycl::free(dw, queue); sycl::free(ds, queue);
  sycl::free(dd, queue); sycl::free(dr, queue); sycl::free(counter, queue);
  std::cout << "W4A16 INT4 accuracy: E=" << p.experts << " M/expert=" << p.rows_per_expert
            << " N=" << p.n << " K=" << p.k << " max_abs=" << max_error << '\n';
  return max_error <= 0.15f;
}

int run_perf(sycl::queue& queue, const Problem& p, int warmup, int iterations) {
  const int total_m = p.experts * p.rows_per_expert;
  const size_t a_count = size_t(total_m) * p.k;
  const size_t w_count = size_t(p.experts) * p.n * p.k / 2;
  const size_t s_count = size_t(p.experts) * p.n * (p.k / p.group_size);
  const size_t d_count = size_t(total_m) * p.n;
  auto* a = sycl::malloc_device<bf16_t>(a_count, queue);
  auto* w = sycl::malloc_device<uint8_t>(w_count, queue);
  auto* s = sycl::malloc_device<bf16_t>(s_count, queue);
  auto* d = sycl::malloc_device<bf16_t>(d_count, queue);
  auto* rows = sycl::malloc_device<int32_t>(p.experts, queue);
  auto* counter = sycl::malloc_device<int32_t>(1, queue);
  if (!a || !w || !s || !d || !rows || !counter) throw std::runtime_error("allocation failed");
  std::vector<int32_t> host_rows(p.experts, p.rows_per_expert);
  queue.memset(a, 1, a_count * sizeof(bf16_t));
  queue.memset(w, 0x11, w_count);
  queue.memset(s, 0x3c, s_count * sizeof(bf16_t));
  queue.memcpy(rows, host_rows.data(), host_rows.size() * sizeof(int32_t)).wait();
  auto launch = [&] {
    queue.memset(counter, 0, sizeof(int32_t));
    return launch_w4a16<moe_w4a16::w4a16_policy, bf16_t, bf16_t, false>(
        queue, a, w, s, nullptr, d, p.n, p.k, rows, p.experts, p.group_size, counter);
  };
  for (int i = 0; i < warmup; ++i) launch().wait();
  double total_ms = 0.0;
  for (int i = 0; i < iterations; ++i) {
    auto event = launch();
    event.wait();
    total_ms += double(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                       event.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1.e-6;
  }
  const double ms = total_ms / iterations;
  const double tops = 2.0 * total_m * p.n * p.k / (ms * 1.e9);
  std::cout << std::fixed << std::setprecision(3) << "W4A16 INT4 baseline: E=" << p.experts
            << " M/expert=" << p.rows_per_expert << " N=" << p.n << " K=" << p.k
            << " device_ms=" << ms << " TOPS=" << tops << '\n';
  sycl::free(a, queue); sycl::free(w, queue); sycl::free(s, queue);
  sycl::free(d, queue); sycl::free(rows, queue); sycl::free(counter, queue);
  return 0;
}

void fill_random_bf16(sycl::queue& queue, bf16_t* data, size_t count) {
  queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
    uint32_t value = static_cast<uint32_t>(id[0]) * 2654435761u + 1013904223u;
    value ^= value >> 16;
    value *= 2246822519u;
    const float unit = static_cast<float>(value >> 8) * (1.0f / 16777216.0f);
    data[id] = bf16_t((unit - 0.5f) * 0.125f);
  }).wait();
}

void fill_random_bytes(sycl::queue& queue, uint8_t* data, size_t count) {
  queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
    uint32_t value = static_cast<uint32_t>(id[0]) * 747796405u + 2891336453u;
    value ^= value >> 16;
    value *= 2246822519u;
    data[id] = static_cast<uint8_t>(value);
  }).wait();
}

float mxfp4_e2m1_value(uint8_t packed_value, int k) {
  constexpr float kMagnitudes[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const uint8_t code = k & 1 ? packed_value >> 4 : packed_value & 0x0f;
  const float magnitude = kMagnitudes[code & 0x07];
  return code & 0x08 ? -magnitude : magnitude;
}

float mxfp4_e8m0_scale(uint8_t exponent) {
  return std::ldexp(1.0f, static_cast<int>(exponent) - 127);
}

bool run_gpt_oss_accuracy(sycl::queue& queue, const gpt_oss_120b::Workload& workload) {
  const int experts = static_cast<int>(workload.rows.size());
  const int total_m = std::accumulate(workload.rows.begin(), workload.rows.end(), 0);
  constexpr int group_size = 32;
  const size_t a_count = size_t(total_m) * workload.k;
  const size_t w_count = size_t(experts) * workload.n * workload.k / 2;
  const size_t s_count = size_t(experts) * workload.n * (workload.k / group_size);
  const size_t d_count = size_t(total_m) * workload.n;

  std::mt19937 rng(20260828);
  std::uniform_real_distribution<float> adist(-0.03125f, 0.03125f);
  std::uniform_int_distribution<int> mxfp4_dist(0, 15);
  std::uniform_int_distribution<int> e8m0_dist(124, 130);
  std::vector<bf16_t> a(a_count), d(d_count);
  std::vector<uint8_t> w(w_count), scales(s_count);
  for (auto& value : a) value = bf16_t(adist(rng));
  for (auto& value : w) {
    const auto low = static_cast<uint8_t>(mxfp4_dist(rng));
    const auto high = static_cast<uint8_t>(mxfp4_dist(rng));
    value = low | (high << 4);
  }
  for (auto& value : scales) value = static_cast<uint8_t>(e8m0_dist(rng));

  auto* da = sycl::malloc_device<bf16_t>(a_count, queue);
  auto* dw = sycl::malloc_device<uint8_t>(w_count, queue);
  auto* ds = sycl::malloc_device<uint8_t>(s_count, queue);
  auto* dd = sycl::malloc_device<bf16_t>(d_count, queue);
  auto* dr = sycl::malloc_device<int32_t>(experts, queue);
  auto* counter = sycl::malloc_device<int32_t>(1, queue);
  if (!da || !dw || !ds || !dd || !dr || !counter) {
    throw std::runtime_error("GPT-OSS W4A16 accuracy allocation failed");
  }
  queue.memcpy(da, a.data(), a_count * sizeof(bf16_t));
  queue.memcpy(dw, w.data(), w_count);
  queue.memcpy(ds, scales.data(), s_count);
  queue.memcpy(dr, workload.rows.data(), experts * sizeof(int32_t));
  queue.memset(counter, 0, sizeof(int32_t)).wait();
  launch_w4a16<moe_w4a16::w4a16_policy, uint8_t, bf16_t, false>(
      queue, da, dw, ds, nullptr, dd, workload.n, workload.k, dr, experts, group_size, counter).wait();
  queue.memcpy(d.data(), dd, d_count * sizeof(bf16_t)).wait();

  float max_abs = 0.0f;
  float max_error_ratio = 0.0f;
  int pre_rows = 0;
  for (int expert = 0; expert < experts; ++expert) {
    for (int m = 0; m < workload.rows[expert]; ++m) {
      const int row = pre_rows + m;
      for (int n = 0; n < workload.n; ++n) {
        float expected = 0.0f;
        for (int k = 0; k < workload.k; ++k) {
          const size_t weight_offset =
              (size_t(expert) * workload.n * workload.k + size_t(n) * workload.k + k) / 2;
          const size_t scale_offset =
              (size_t(expert) * workload.n + n) * (workload.k / group_size) + k / group_size;
          expected += static_cast<float>(a[size_t(row) * workload.k + k]) *
              mxfp4_e2m1_value(w[weight_offset], k) * mxfp4_e8m0_scale(scales[scale_offset]);
        }
        const float error = std::abs(expected - static_cast<float>(d[size_t(row) * workload.n + n]));
        max_abs = std::max(max_abs, error);
        max_error_ratio = std::max(max_error_ratio, error / (0.01f + 0.1f * std::abs(expected)));
      }
    }
    pre_rows += workload.rows[expert];
  }
  sycl::free(da, queue); sycl::free(dw, queue); sycl::free(ds, queue);
  sycl::free(dd, queue); sycl::free(dr, queue); sycl::free(counter, queue);
  std::cout << "W4A16 MXFP4 GPT-OSS-120B accuracy workload=" << workload.name
            << " E=" << experts << " total_M=" << total_m
            << " N=" << workload.n << " K=" << workload.k
            << " max_abs=" << max_abs << " max_error_ratio=" << max_error_ratio << '\n';
  return max_error_ratio <= 1.0f;
}

int run_gpt_oss_workload(
    sycl::queue& queue, const gpt_oss_120b::Workload& workload, int warmup, int iterations) {
  const int experts = static_cast<int>(workload.rows.size());
  const int total_m = std::accumulate(workload.rows.begin(), workload.rows.end(), 0);
  const int active_experts = std::count_if(
      workload.rows.begin(), workload.rows.end(), [](int32_t rows) { return rows != 0; });
  constexpr int group_size = 32;
  const size_t a_count = size_t(total_m) * workload.k;
  const size_t w_count = size_t(experts) * workload.n * workload.k / 2;
  const size_t s_count = size_t(experts) * workload.n * (workload.k / group_size);
  const size_t d_count = size_t(total_m) * workload.n;
  auto* a = sycl::malloc_device<bf16_t>(a_count, queue);
  auto* w = sycl::malloc_device<uint8_t>(w_count, queue);
  auto* scales = sycl::malloc_device<uint8_t>(s_count, queue);
  auto* d = sycl::malloc_device<bf16_t>(d_count, queue);
  auto* rows = sycl::malloc_device<int32_t>(experts, queue);
  auto* counter = sycl::malloc_device<int32_t>(1, queue);
  if (!a || !w || !scales || !d || !rows || !counter) {
    throw std::runtime_error("GPT-OSS W4A16 allocation failed");
  }
  // GPT-OSS production weights are MXFP4. Use the W4A16 MXFP4 path: packed
  // E2M1 values and raw E8M0 scale exponents. Random weights avoid a
  // compression-assisted timing result; initialization is outside timing.
  fill_random_bf16(queue, a, a_count);
  fill_random_bytes(queue, w, w_count);
  queue.memset(scales, 127, s_count).wait();  // E8M0 exponent 0 (scale = 1).
  queue.memcpy(rows, workload.rows.data(), experts * sizeof(int32_t)).wait();
  auto launch = [&] {
    queue.memset(counter, 0, sizeof(int32_t));
    return launch_w4a16<moe_w4a16::w4a16_policy, uint8_t, bf16_t, false>(
        queue, a, w, scales, nullptr, d, workload.n, workload.k, rows, experts, group_size, counter);
  };
  for (int i = 0; i < warmup; ++i) launch().wait();
  double total_ms = 0.0;
  for (int i = 0; i < iterations; ++i) {
    auto event = launch();
    event.wait();
    total_ms += double(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                       event.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1.e-6;
  }
  const double ms = total_ms / iterations;
  const double tops = 2.0 * total_m * workload.n * workload.k / (ms * 1.e9);
  std::cout << std::fixed << std::setprecision(3)
            << "W4A16 MXFP4 workload=" << workload.name
            << " E=" << experts << " active_E=" << active_experts
            << " total_M=" << total_m << " N=" << workload.n << " K=" << workload.k
            << " device_ms=" << ms << " TOPS=" << tops << '\n';
  sycl::free(a, queue); sycl::free(w, queue); sycl::free(scales, queue);
  sycl::free(d, queue); sycl::free(rows, queue); sycl::free(counter, queue);
  return 0;
}

}  // namespace

int main(int argc, const char** argv) {
  cutlass::CommandLine cmd(argc, argv);
  std::string mode = "accuracy";
  std::string workload;
  Problem p;
  int warmup = 5, iterations = 20;
  cmd.get_cmd_line_argument("mode", mode);
  cmd.get_cmd_line_argument("workload", workload);
  cmd.get_cmd_line_argument("experts", p.experts);
  cmd.get_cmd_line_argument("rows", p.rows_per_expert);
  cmd.get_cmd_line_argument("n", p.n);
  cmd.get_cmd_line_argument("k", p.k);
  cmd.get_cmd_line_argument("warmup", warmup);
  cmd.get_cmd_line_argument("iterations", iterations);
  if (p.experts <= 0 || p.rows_per_expert <= 0 || p.n <= 0 || p.k <= 0 ||
      p.n % 8 || p.k % p.group_size) {
    std::cerr << "E/M/N/K must be positive; N must be divisible by 8 and K by 32.\n";
    return 1;
  }
  try {
    sycl::queue queue{sycl::gpu_selector_v, sycl::property_list{
        sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    const auto workloads = gpt_oss_120b::workloads();
    if (cmd.check_cmd_line_flag("list-workloads")) {
      for (const auto& item : workloads) {
        const int total_m = std::accumulate(item.rows.begin(), item.rows.end(), 0);
        const int active = std::count_if(
            item.rows.begin(), item.rows.end(), [](int32_t rows) { return rows != 0; });
        std::cout << item.name << ": E=" << item.rows.size() << " active_E=" << active
                  << " total_M=" << total_m << " N=" << item.n << " K=" << item.k << '\n';
      }
      return 0;
    }
    if (!workload.empty()) {
      const auto found = std::find_if(
          workloads.begin(), workloads.end(), [&](const auto& item) { return item.name == workload; });
      if (found == workloads.end()) throw std::invalid_argument("unknown --workload; use --list-workloads");
      if (mode == "accuracy") return run_gpt_oss_accuracy(queue, *found) ? 0 : 1;
      if (mode == "perf") return run_gpt_oss_workload(queue, *found, warmup, iterations);
      throw std::invalid_argument("--workload requires --mode=accuracy or --mode=perf");
    }
    if (mode == "accuracy") return run_accuracy(queue, p) ? 0 : 1;
    if (mode == "perf") return run_perf(queue, p, warmup, iterations);
    std::cerr << "--mode must be accuracy or perf\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}

#undef SYCL_INTEL_TARGET
