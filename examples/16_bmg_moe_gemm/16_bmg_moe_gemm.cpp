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
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include "gpt_oss_120b_workloads.hpp"
#include "kernel/moe/xe20/w4a16/gemm_xe2_policy.hpp"
#include "kernel/moe/xe20/w4a16/grouped_gemm_xe2.hpp"

namespace moe_w4a16 {
template <typename, typename, typename, typename, bool, char, char, class>
class GemmCuteName;
}

namespace {

using bf16_t = cutlass::bfloat16_t;
using namespace cute;

// The work-group tile PR 446 replaced. It lives here, outside the kernel
// sources, so the "before" row of the PR's table stays measurable without
// adding anything to the upstream policy menu.
class legacy_policy_m_128_n_256 : public moe_w4a16::xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
};

// Policy ids 0-4 are upstream's menu; this one extends it locally.
constexpr int kLegacyPolicyId = 5;

// The W4A16 kernel issues vectorized 2D block accesses.  The PyTorch XPU
// allocator used by sgl-kernel-xpu provides cache-line aligned tensor storage;
// keep the standalone USM harness on the same contract.
template <typename T>
T* aligned_device_alloc(size_t count, sycl::queue& queue) {
  constexpr size_t kAlignment = 64;
  return sycl::aligned_alloc_device<T>(kAlignment, count, queue);
}

// The mainloop prefetches each subgroup's scale column with a 2D block message
// (prefetch_scale_group() in gemm_xe2.hpp): a (SG_N x 1) surface whose pitch is
// the per-row group count. The hardware rounds that 1-byte-wide block up to its
// minimum block width, so the message reads a little past the last scale of the
// last expert. Being a prefetch its data is discarded and results are
// unaffected, but against an exact-size USM allocation that happens to end on an
// unmapped page it faults the device with UR_RESULT_ERROR_DEVICE_LOST -- as
// GPT-OSS-120B TP=8 gemm1 (E=128, N=768, K=2880) does, whose scale buffer is
// exactly 135 MiB. sgl-kernel-xpu never sees this because PyTorch's XPU caching
// allocator sub-allocates tensors from large slabs, so the over-read always
// lands in mapped memory. 64 B of slack measured sufficient; keep a page.
constexpr size_t kScalePrefetchSlack = 4096;

template <typename T>
T* scales_device_alloc(size_t count, sycl::queue& queue) {
  return aligned_device_alloc<T>(count + (kScalePrefetchSlack + sizeof(T) - 1) / sizeof(T), queue);
}

// A direct, single-launch W4A16 baseline. The kernel sources under w4a16 are
// copied unchanged from sgl-kernel-xpu. No local bucket dispatch, heuristic,
// tile override, fused activation, diagnostic path, or compiler tuning remains.
template <
    char LayoutA,
    char LayoutB,
    bool HasZero,
    class Policy,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
sycl::event launch_w4a16_kernel(
    sycl::queue& stream,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementS* zeros,
    const ElementBI* bias,
    ElementD* output,
    int n,
    int k,
    const int32_t* rows,
    int experts,
    int group_size,
    int32_t* counter) {
  using ElementAUnqualified = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementAUnqualified>{};
  using WGTile = typename Policy::WGTile;
  using SGLayout = typename Policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;
  auto mma = MMA{};
  const int threads = size(mma);
  constexpr int kThreadsPerSm = 512;
  if (kThreadsPerSm % threads) throw std::runtime_error("invalid W4A16 workgroup size");
  const int sms = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  const sycl::range<3> local(1, 1, threads);
  const sycl::range<3> global(1, sms * kThreadsPerSm / threads, 1);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  using GmemTiledCopyA = typename Policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename Policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename Policy::GmemTiledCopyD;
  return stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<moe_w4a16::GemmCuteName<
        ElementA, ElementB, ElementS, ElementD, HasZero, LayoutA, LayoutB, Policy>>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto) {
          moe_w4a16::MoEGEMM<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, LayoutA, LayoutB, 'R', HasZero>(
              activations, weights, scales, zeros, bias,
              output, mma, rows, experts, group_size, n, k, counter, local_mem);
        });
  });
}

template <class Policy, typename ElementS, typename ElementA, bool HasZero>
sycl::event launch_w4a16(
    sycl::queue& queue,
    const ElementA* activations,
    const uint8_t* packed_weights,
    const ElementS* scales,
    const ElementS* zeros,
    ElementA* output,
    int n,
    int k,
    const int32_t* rows,
    int experts,
    int group_size,
    int32_t* counter) {
  return launch_w4a16_kernel<'R', 'C', HasZero, Policy, ElementA, uint8_t, ElementS, float, ElementA>(
      queue, activations, packed_weights, scales, zeros, static_cast<const float*>(nullptr), output,
      n, k, rows, experts, group_size, counter);
}

// GEMM shape -> tile policy, copied from select_w4a16_tile_m() /
// select_w4a16_policy_id() in sgl-kernel-xpu's GroupGemmW4A16Xe20.cpp at
// 44060731 (PR 446). Each candidate is scored by its peak throughput times how
// much of its M and N tile the GEMM actually fills.
constexpr int kW4A16TileM[] = {32, 64, 128};
constexpr int kW4A16TileN[] = {64, 128, 128};
constexpr float kW4A16TilePeakTflops[] = {49.0f, 64.0f, 68.0f};

int select_w4a16_tile_m(int avg_m, int gemm_n) {
  int best_tile_m = kW4A16TileM[0];
  float best_score = 0.0f;
  for (size_t i = 0; i < sizeof(kW4A16TileM) / sizeof(kW4A16TileM[0]); ++i) {
    const int tile_m = kW4A16TileM[i];
    const int tile_n = kW4A16TileN[i];
    const int rows_computed = ((avg_m + tile_m - 1) / tile_m) * tile_m;
    const int columns_computed = ((gemm_n + tile_n - 1) / tile_n) * tile_n;
    const float score = kW4A16TilePeakTflops[i] * static_cast<float>(avg_m) / static_cast<float>(rows_computed) *
                        static_cast<float>(gemm_n) / static_cast<float>(columns_computed);
    if (score > best_score) {
      best_score = score;
      best_tile_m = tile_m;
    }
  }
  return best_tile_m;
}

int select_w4a16_policy_id(int avg_m, int gemm_n) {
  if (avg_m <= 4) return 0;
  if (avg_m <= 8) return 1;

  const int tile_m = select_w4a16_tile_m(avg_m, gemm_n);
  if (tile_m <= 32) return 2;
  if (tile_m <= 64) return 3;
  return 4;
}

// The pre-446 selector, for the "before" row of the PR's table.
int select_w4a16_policy_id_pre_446(int avg_m) {
  if (avg_m <= 4) return 0;
  if (avg_m <= 8) return 1;
  if (avg_m <= 128) return 2;
  return kLegacyPolicyId;
}

const char* policy_name(int policy_id) {
  switch (policy_id) {
    case 0: return "w4a16_policy_m_8_n_64";
    case 1: return "w4a16_policy_m_16_n_64";
    case 2: return "w4a16_policy_m_32_n_64";
    case 3: return "w4a16_policy_m_64_n_128";
    case 4: return "w4a16_policy_m_128_n_128";
    case kLegacyPolicyId: return "legacy_policy_m_128_n_256";
    default: return "invalid";
  }
}

// How the tile is chosen. `after` is PR 446's scored selector, `before` is the
// avg_m threshold ladder it replaced; a bare policy id forces one tile.
struct TileChoice {
  bool pre_446 = false;
  int forced_policy_id = -1;

  int policy_id(int total_m, int experts, int n) const {
    if (forced_policy_id >= 0) return forced_policy_id;
    const int avg_m = total_m / experts;
    return pre_446 ? select_w4a16_policy_id_pre_446(avg_m) : select_w4a16_policy_id(avg_m, n);
  }
};

template <typename ElementS, typename ElementA, bool HasZero>
sycl::event launch_w4a16_dispatched(
    sycl::queue& queue,
    const ElementA* activations,
    const uint8_t* packed_weights,
    const ElementS* scales,
    const ElementS* zeros,
    ElementA* output,
    int total_m,
    int n,
    int k,
    const int32_t* rows,
    int experts,
    int group_size,
    int32_t* counter,
    TileChoice tile = {}) {
#define LAUNCH_W4A16(Policy)                                                                                 \
  return launch_w4a16<Policy, ElementS, ElementA, HasZero>(                                                  \
      queue, activations, packed_weights, scales, zeros, output, n, k, rows, experts, group_size, counter)

  switch (tile.policy_id(total_m, experts, n)) {
    case 0: LAUNCH_W4A16(moe_w4a16::w4a16_policy_m_8_n_64);
    case 1: LAUNCH_W4A16(moe_w4a16::w4a16_policy_m_16_n_64);
    case 2: LAUNCH_W4A16(moe_w4a16::w4a16_policy_m_32_n_64);
    case 3: LAUNCH_W4A16(moe_w4a16::w4a16_policy_m_64_n_128);
    case 4: LAUNCH_W4A16(moe_w4a16::w4a16_policy_m_128_n_128);
    case kLegacyPolicyId: LAUNCH_W4A16(legacy_policy_m_128_n_256);
    default: throw std::runtime_error("invalid W4A16 policy id");
  }
#undef LAUNCH_W4A16
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

// Perf paths fill their operands on the device with pseudo-random data. Xe
// memory compression makes a constant-filled buffer read far faster than a real
// weight tensor, so a memset baseline overstates a weight-bound launch.
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

bool run_accuracy(sycl::queue& queue, const Problem& p, TileChoice tile) {
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
  auto* da = aligned_device_alloc<bf16_t>(a_count, queue);
  auto* dw = aligned_device_alloc<uint8_t>(w_count, queue);
  auto* ds = scales_device_alloc<bf16_t>(s_count, queue);
  auto* dd = aligned_device_alloc<bf16_t>(d_count, queue);
  auto* dr = aligned_device_alloc<int32_t>(rows.size(), queue);
  auto* counter = aligned_device_alloc<int32_t>(1, queue);
  if (!da || !dw || !ds || !dd || !dr || !counter) throw std::runtime_error("allocation failed");
  queue.memcpy(da, a.data(), a_count * sizeof(bf16_t));
  queue.memcpy(dw, w.data(), w_count);
  queue.memcpy(ds, scales.data(), s_count * sizeof(bf16_t));
  queue.memcpy(dr, rows.data(), rows.size() * sizeof(int32_t));
  queue.memset(counter, 0, sizeof(int32_t)).wait();
  launch_w4a16_dispatched<bf16_t, bf16_t, false>(
      queue, da, dw, ds, nullptr, dd, total_m, p.n, p.k, dr, p.experts, p.group_size, counter, tile).wait();
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
            << " N=" << p.n << " K=" << p.k
            << " policy=" << policy_name(tile.policy_id(total_m, p.experts, p.n))
            << " max_abs=" << max_error << '\n';
  return max_error <= 0.15f;
}

// ElementS selects the quantization: bf16_t scales are signed INT4 weights,
// uint8_t scales are MXFP4 (E2M1 weights, E8M0 scale exponents). Uniform
// rows-per-expert, which is what PR 446's avg_m sweep measures.
template <typename ElementS>
int run_perf(sycl::queue& queue, const Problem& p, int warmup, int iterations, TileChoice tile) {
  constexpr bool kMxfp4 = std::is_same_v<ElementS, uint8_t>;
  const int total_m = p.experts * p.rows_per_expert;
  const size_t a_count = size_t(total_m) * p.k;
  const size_t w_count = size_t(p.experts) * p.n * p.k / 2;
  const size_t s_count = size_t(p.experts) * p.n * (p.k / p.group_size);
  const size_t d_count = size_t(total_m) * p.n;
  auto* a = aligned_device_alloc<bf16_t>(a_count, queue);
  auto* w = aligned_device_alloc<uint8_t>(w_count, queue);
  auto* s = scales_device_alloc<ElementS>(s_count, queue);
  auto* d = aligned_device_alloc<bf16_t>(d_count, queue);
  auto* rows = aligned_device_alloc<int32_t>(p.experts, queue);
  auto* counter = aligned_device_alloc<int32_t>(1, queue);
  if (!a || !w || !s || !d || !rows || !counter) throw std::runtime_error("allocation failed");
  std::vector<int32_t> host_rows(p.experts, p.rows_per_expert);
  fill_random_bf16(queue, a, a_count);
  fill_random_bytes(queue, w, w_count);
  if constexpr (kMxfp4) {
    queue.memset(s, 127, s_count).wait();  // E8M0 exponent 0 (scale = 1).
  } else {
    fill_random_bf16(queue, reinterpret_cast<bf16_t*>(s), s_count);
  }
  queue.memcpy(rows, host_rows.data(), host_rows.size() * sizeof(int32_t)).wait();
  const int policy_id = tile.policy_id(total_m, p.experts, p.n);
  auto launch = [&] {
    queue.memset(counter, 0, sizeof(int32_t));
    return launch_w4a16_dispatched<ElementS, bf16_t, false>(
        queue, a, w, s, nullptr, d, total_m, p.n, p.k, rows, p.experts, p.group_size, counter, tile);
  };
  for (int i = 0; i < warmup; ++i) launch().wait();
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    sycl::event event = launch();
    event.wait();
    samples.push_back(double(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                             event.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1.e-6);
  }
  std::sort(samples.begin(), samples.end());
  const double ms = samples[samples.size() / 2];
  const double tops = 2.0 * total_m * p.n * p.k / (ms * 1.e9);
  std::cout << std::fixed << std::setprecision(3) << "W4A16 " << (kMxfp4 ? "MXFP4" : "INT4")
            << " baseline: E=" << p.experts
            << " M/expert=" << p.rows_per_expert << " N=" << p.n << " K=" << p.k
            << " policy=" << policy_name(policy_id)
            << " device_ms=" << ms << " TOPS=" << tops << '\n';
  sycl::free(a, queue); sycl::free(w, queue); sycl::free(s, queue);
  sycl::free(d, queue); sycl::free(rows, queue); sycl::free(counter, queue);
  return 0;
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

std::vector<int> reference_indices(int extent, bool exhaustive) {
  std::vector<int> indices;
  if (exhaustive) {
    indices.resize(extent);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }
  const int candidates[] = {
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128,
      extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1};
  for (int index : candidates) {
    if (index >= 0 && index < extent &&
        std::find(indices.begin(), indices.end(), index) == indices.end()) {
      indices.push_back(index);
    }
  }
  return indices;
}

bool run_gpt_oss_accuracy(sycl::queue& queue, const gpt_oss_120b::Workload& workload, TileChoice tile) {
  const int experts = static_cast<int>(workload.rows.size());
  const int total_m = std::accumulate(workload.rows.begin(), workload.rows.end(), 0);
  constexpr int group_size = 32;
  constexpr float atol = 0.005f;
  constexpr float rtol = 0.005f;
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

  auto* da = aligned_device_alloc<bf16_t>(a_count, queue);
  auto* dw = aligned_device_alloc<uint8_t>(w_count, queue);
  auto* ds = scales_device_alloc<uint8_t>(s_count, queue);
  auto* dd = aligned_device_alloc<bf16_t>(d_count, queue);
  auto* dr = aligned_device_alloc<int32_t>(experts, queue);
  auto* counter = aligned_device_alloc<int32_t>(1, queue);
  if (!da || !dw || !ds || !dd || !dr || !counter) {
    throw std::runtime_error("GPT-OSS W4A16 accuracy allocation failed");
  }
  queue.memcpy(da, a.data(), a_count * sizeof(bf16_t));
  queue.memcpy(dw, w.data(), w_count);
  queue.memcpy(ds, scales.data(), s_count);
  queue.memcpy(dr, workload.rows.data(), experts * sizeof(int32_t));
  queue.memset(counter, 0, sizeof(int32_t)).wait();
  launch_w4a16_dispatched<uint8_t, bf16_t, false>(
      queue, da, dw, ds, nullptr, dd, total_m, workload.n, workload.k, dr, experts, group_size, counter, tile)
      .wait();
  queue.memcpy(d.data(), dd, d_count * sizeof(bf16_t)).wait();

  float max_abs = 0.0f;
  float max_relative = 0.0f;
  float max_tolerance_ratio = 0.0f;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  size_t sampled_values = 0;
  const bool exhaustive = total_m <= 16;
  const auto output_columns = reference_indices(workload.n, exhaustive);
  int pre_rows = 0;
  for (int expert = 0; expert < experts; ++expert) {
    for (int m : reference_indices(workload.rows[expert], exhaustive)) {
      const int row = pre_rows + m;
      for (int n : output_columns) {
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
        if (std::abs(expected) >= atol) max_relative = std::max(max_relative, error / std::abs(expected));
        max_tolerance_ratio = std::max(max_tolerance_ratio, error / (atol + rtol * std::abs(expected)));
        squared_error += double(error) * error;
        squared_reference += double(expected) * expected;
        ++sampled_values;
      }
    }
    pre_rows += workload.rows[expert];
  }
  sycl::free(da, queue); sycl::free(dw, queue); sycl::free(ds, queue);
  sycl::free(dd, queue); sycl::free(dr, queue); sycl::free(counter, queue);
  std::cout << "W4A16 MXFP4 GPT-OSS-120B accuracy workload=" << workload.name
            << " E=" << experts << " total_M=" << total_m
            << " N=" << workload.n << " K=" << workload.k
            << " policy=" << policy_name(tile.policy_id(total_m, experts, workload.n))
            << " sampled_values=" << sampled_values
            << " max_abs=" << max_abs
            << " max_relative=" << max_relative
            << " l2_relative=" << std::sqrt(squared_error / squared_reference)
            << " max_tolerance_ratio=" << max_tolerance_ratio << '\n';
  return max_tolerance_ratio <= 1.0f;
}

int run_gpt_oss_workload(
    sycl::queue& queue, const gpt_oss_120b::Workload& workload, int warmup, int iterations, TileChoice tile) {
  const int experts = static_cast<int>(workload.rows.size());
  const int total_m = std::accumulate(workload.rows.begin(), workload.rows.end(), 0);
  const int active_experts = std::count_if(
      workload.rows.begin(), workload.rows.end(), [](int32_t rows) { return rows != 0; });
  constexpr int group_size = 32;
  const size_t a_count = size_t(total_m) * workload.k;
  const size_t w_count = size_t(experts) * workload.n * workload.k / 2;
  const size_t s_count = size_t(experts) * workload.n * (workload.k / group_size);
  const size_t d_count = size_t(total_m) * workload.n;
  auto* a = aligned_device_alloc<bf16_t>(a_count, queue);
  auto* w = aligned_device_alloc<uint8_t>(w_count, queue);
  auto* scales = scales_device_alloc<uint8_t>(s_count, queue);
  auto* d = aligned_device_alloc<bf16_t>(d_count, queue);
  auto* rows = aligned_device_alloc<int32_t>(experts, queue);
  auto* counter = aligned_device_alloc<int32_t>(1, queue);
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
  const int policy_id = tile.policy_id(total_m, experts, workload.n);
  auto launch = [&] {
    queue.memset(counter, 0, sizeof(int32_t));
    return launch_w4a16_dispatched<uint8_t, bf16_t, false>(
        queue, a, w, scales, nullptr, d, total_m, workload.n, workload.k, rows, experts, group_size, counter,
        tile);
  };
  for (int i = 0; i < warmup; ++i) launch().wait();
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    auto event = launch();
    event.wait();
    samples.push_back(double(event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                             event.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1.e-6);
  }
  std::sort(samples.begin(), samples.end());
  const double ms = samples[samples.size() / 2];
  const double tops = 2.0 * total_m * workload.n * workload.k / (ms * 1.e9);
  std::cout << std::fixed << std::setprecision(3)
            << "W4A16 MXFP4 workload=" << workload.name
            << " E=" << experts << " active_E=" << active_experts
            << " total_M=" << total_m << " N=" << workload.n << " K=" << workload.k
            << " policy=" << policy_name(policy_id)
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
  std::string selector = "after";
  std::string quant = "int4";
  Problem p;
  int warmup = 5, iterations = 20, policy = -1;
  cmd.get_cmd_line_argument("mode", mode);
  cmd.get_cmd_line_argument("workload", workload);
  cmd.get_cmd_line_argument("selector", selector);
  cmd.get_cmd_line_argument("policy", policy);
  cmd.get_cmd_line_argument("quant", quant);
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
  if (selector != "after" && selector != "before") {
    std::cerr << "--selector must be after (PR 446) or before (the thresholds it replaced)\n";
    return 1;
  }
  if (quant != "int4" && quant != "mxfp4") {
    std::cerr << "--quant must be int4 or mxfp4\n";
    return 1;
  }
  if (policy > kLegacyPolicyId) {
    std::cerr << "--policy must be 0-" << kLegacyPolicyId << '\n';
    return 1;
  }
  const TileChoice tile{selector == "before", policy};
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
      if (mode == "accuracy") return run_gpt_oss_accuracy(queue, *found, tile) ? 0 : 1;
      if (mode == "perf") return run_gpt_oss_workload(queue, *found, warmup, iterations, tile);
      throw std::invalid_argument("--workload requires --mode=accuracy or --mode=perf");
    }
    if (mode == "accuracy") return run_accuracy(queue, p, tile) ? 0 : 1;
    if (mode == "perf") {
      return quant == "mxfp4" ? run_perf<uint8_t>(queue, p, warmup, iterations, tile)
                              : run_perf<bf16_t>(queue, p, warmup, iterations, tile);
    }
    std::cerr << "--mode must be accuracy or perf\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}

#undef SYCL_INTEL_TARGET
