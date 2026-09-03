/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling MoE / dense-MLP silu_and_mul (SwiGLU) activation for CUTLASS SYCL on BMG.
 *
 * Semantics (ported from sglang python/sglang/kernels/ops/moe/inkling_moe.py,
 * silu_and_mul_interleaved_kernel / silu_and_mul_non_interleaved_kernel, and
 * from srt/models/inkling_common/dense_mlp.py, swiglu / swiglu_contiguous):
 *
 *   gateup[M, 2N] -> out[M, N]
 *   interleaved (the model default, inference_moe_w13_interleaved=True):
 *     gate = gateup[m, 2n], up = gateup[m, 2n + 1]
 *   non-interleaved / contiguous ([gate || up], used under --enable-lora):
 *     gate = gateup[m, n], up = gateup[m, N + n]
 *
 *   float g = float(gate), u = float(up);            // widen first
 *   float v = (g * sigmoid(g)) * u;                  // fp32 silu-and-mul
 *   if (has_topk_weights) v = v * float(weight[m]);  // routing weight LAST
 *   out[m, n] = Element(v);                          // ONE rounding cast
 *
 * The single-cast / weight-last order is the whole point of this example: the
 * Inkling kernel is documented as a bitwise-identical port of the Helion
 * kernels it replaced (fp32 math, gate * sigmoid(gate) * up, weight scale
 * last, one rounding cast at the store). Casting per multiply -- as the stock
 * sglang silu_and_mul_kernel in ops/elementwise/elementwise.py does
 * ("cast down before mul to better match training") -- is a different kernel
 * and does not match Inkling.
 *
 * InklingSwiglu (srt/models/inkling_common/dense_mlp.py) is only a thin
 * nn.Module around the same math: it selects swiglu (interleaved) or
 * swiglu_contiguous by a flag and adds NO gamma, alpha, limit or clamp term
 * (the shared-expert MoeRunnerConfig sets gemm1_alpha=None and
 * gemm1_clamp_limit=None). The dense path therefore equals the routed path
 * with has_topk_weights=false; the shared-expert path
 * (InklingBatchDenseMLP._swiglu) reuses the routed kernel and passes the gate
 * gammas in the topk-weight slot, which is the has_topk_weights=true case.
 *
 * Roofline: one output element reads 2 gate/up elements and writes 1, doing a
 * handful of FP32 ops. For bf16 that is 6 bytes moved per output element with
 * ~5 FLOP, i.e. ~0.8 FLOP/B: purely memory bound. The benchmark reports
 * effective GB/s against the ~400 GB/s BMG DRAM ceiling, never TOPS.
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cutlass::examples::bmg_moe_silu_and_mul {

constexpr int kWorkGroupItems = 256;
constexpr double kBytesPerGB = 1.0e9;

// CPU-reference / host-buffer cap. Larger cases still run and are benchmarked,
// they just report verify=SKIP (mirrors 23_bmg_mel_embedding_sum).
constexpr std::size_t kVerifyMaxOutputElements = 8u << 20;

// Calibrated on one Intel Arc Pro B60 shared with another job (so several
// percent of run-to-run noise) at --iterations=50 --warmup=20; see README.md for
// the measured table. Every gate sits ~15% below the measured floor of the band
// it guards.
//
// The op moves 3 bytes-per-2-byte-element (read gate, read up, write out) and
// does ~5 FP32 ops, so DRAM-resident cases saturate: measured 385-394 GB/s
// against the ~400 GB/s BMG ceiling.
constexpr double kDramTargetGBps = 300.0;
// N/P = 48 at T = 4096 keeps the whole 7 MB working set in L2 and measures
// 544-554 GB/s, i.e. above the DRAM ceiling.
constexpr double kL2ResidentTargetGBps = 420.0;
// The same width at T = 16384 (29 MB) only partly fits: 416 GB/s.
constexpr double kSmallL2TargetGBps = 350.0;
// Decode / small-T cases are launch-latency bound (one wave does not fill the
// card), so they stay report-only rather than gated on a guessed number.
constexpr double kNoTarget = 0.0;

enum class DType { kAll, kFloat, kBf16, kFp16 };

enum class LayoutFilter { kAll, kInterleaved, kContiguous };

struct Options {
  std::string suite = "quick";
  std::string shape;
  DType dtype = DType::kAll;
  LayoutFilter layout = LayoutFilter::kAll;
  int iterations = 20;
  int warmup = 5;
  int elems_per_item = 0;  // 0 = auto
  bool verify = true;
  bool benchmark = true;
  double perf_threshold_scale = 1.0;
  bool help = false;
};

struct CaseConfig {
  std::string name;
  int m = 1;                   // routed rows: T * top_k (or T for the dense path)
  int n = 384;                 // per-rank intermediate width I / P
  bool interleaved = true;     // Inkling default
  bool has_weights = true;     // false = dense / InklingSwiglu path
  double target_gbps = kNoTarget;
  int elems_per_item = 0;      // 0 = auto; pinned by the tail-coverage cases
};

inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

inline int round_up(int x, int y) { return ceil_div(x, y) * y; }

// Pick the (rows, columns) split of a kWorkGroupItems work-group.
//
// Columns are the contiguous direction, so a wide column split is what makes a
// subgroup's 16 lanes read one contiguous span; measured, coalescing dominates
// the cost of padded work-items (N/P=48 at T=16384 loses 13% when the split
// narrows from 8 columns to 2, even though it launches 25% fewer items).
// So: take the largest power-of-two divisor of the column count when that still
// keeps a full subgroup contiguous -- which avoids padding n=3072 (384 columns)
// out to 512 -- and otherwise round the columns up to a power of two.
inline void choose_group_shape(int cols, int& rows_per_group, int& cols_per_group) {
  int divisor = 1;
  while (divisor < kWorkGroupItems && (cols % (2 * divisor)) == 0) {
    divisor *= 2;
  }
  int cols_pow2 = 1;
  while (cols_pow2 < cols && cols_pow2 < kWorkGroupItems) {
    cols_pow2 *= 2;
  }
  cols_per_group = (divisor >= 16) ? divisor : cols_pow2;
  rows_per_group = kWorkGroupItems / cols_per_group;
}

// ---------------------------------------------------------------------------
// Small device-buffer RAII helper
// ---------------------------------------------------------------------------

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
    // 8-byte alignment so the vectorized uint64 path is usable (plain
    // malloc_device only promises alignof(T)).
    ptr = sycl::aligned_alloc_device<T>(8, std::max<std::size_t>(count, 1), q);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  ~DeviceBuffer() { reset(); }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(queue, other.queue);
    std::swap(ptr, other.ptr);
    std::swap(count, other.count);
  }

  void reset() {
    if (ptr != nullptr) {
      sycl::free(ptr, *queue);
    }
    ptr = nullptr;
    queue = nullptr;
    count = 0;
  }

  T* get() const { return ptr; }

  void copy_from(std::vector<T> const& host) {
    if (host.size() > count) {
      throw std::runtime_error("copy_from exceeds device buffer");
    }
    if (!host.empty()) {
      queue->memcpy(ptr, host.data(), sizeof(T) * host.size()).wait();
    }
  }

  void copy_to(std::vector<T>& host) const {
    if (host.size() > count) {
      throw std::runtime_error("copy_to exceeds device buffer");
    }
    if (!host.empty()) {
      queue->memcpy(host.data(), ptr, sizeof(T) * host.size()).wait();
    }
  }

  // Fill the whole buffer by tiling a random host block. Used for cases too
  // large to materialize on the host: constant/zero fills would let Xe memory
  // compression report fictitious bandwidth.
  void fill_tiled(std::vector<T> const& tile) {
    if (tile.empty() || count == 0) {
      return;
    }
    for (std::size_t off = 0; off < count; off += tile.size()) {
      std::size_t n = std::min(tile.size(), count - off);
      queue->memcpy(ptr + off, tile.data(), sizeof(T) * n).wait();
    }
  }
};

// ---------------------------------------------------------------------------
// Numeric helpers (shared host/device so the reference matches bit for bit
// except for the platform exp())
// ---------------------------------------------------------------------------

template <typename Element>
CUTLASS_HOST_DEVICE float to_float(Element x) {
#if defined(__SYCL_DEVICE_ONLY__)
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    uint32_t bits = static_cast<uint32_t>(x.raw()) << 16;
    return sycl::bit_cast<float>(bits);
  } else {
    return static_cast<float>(x);
  }
#else
  return static_cast<float>(x);
#endif
}

template <typename Element>
CUTLASS_HOST_DEVICE Element from_float(float x) {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    // Round-to-nearest-even bf16, matching torch's cast.
    uint32_t bits = sycl::bit_cast<uint32_t>(x);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
      if (bits & 0x007fffffu) {
        return cutlass::bfloat16_t::bitcast(0x7fffu);
      }
      return cutlass::bfloat16_t::bitcast(static_cast<uint16_t>(bits >> 16));
    }
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7fffu + lsb;
    return cutlass::bfloat16_t::bitcast(static_cast<uint16_t>((bits + rounding_bias) >> 16));
  } else {
    return static_cast<Element>(x);
  }
}

// silu(x) * y, in the exact operation order of the Inkling kernel:
// (gate * sigmoid(gate)) * up, all in fp32.
CUTLASS_HOST_DEVICE float silu_and_mul_f32(float gate, float up) {
  float sigmoid = 1.0f / (1.0f + sycl::exp(-gate));
  return (gate * sigmoid) * up;
}

template <typename Element>
CUTLASS_DEVICE Element element_from_raw(uint64_t raw, int lane) {
  return Element::bitcast(static_cast<uint16_t>(raw >> (16 * lane)));
}

template <typename Element>
uint32_t host_bits(Element value) {
  if constexpr (std::is_same_v<Element, float>) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  } else {
    return static_cast<uint32_t>(value.raw());
  }
}

// Monotone key for a 16-bit float's raw bits, so |key(a) - key(b)| is the ulp
// distance. Built from the magnitude, not the raw word, so that the mapping is
// continuous across zero and -0 keys the same as +0 (raw 0x8000 and 0x8001 are
// one ulp apart, not 32769).
inline int ordered_raw16(uint32_t raw) {
  int magnitude = static_cast<int>(raw & 0x7fffu);
  bool negative = (raw & 0x8000u) != 0;
  return negative ? (0x8000 - magnitude) : (0x8000 + magnitude);
}

inline double event_ms(sycl::event const& event) {
  auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

inline sycl::queue make_queue() {
  return sycl::queue(
      sycl::gpu_selector_v,
      sycl::property_list{sycl::property::queue::in_order{},
                          sycl::property::queue::enable_profiling{}});
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename Element>
struct KernelParams {
  Element const* __restrict__ gateup = nullptr;
  float const* __restrict__ weights = nullptr;
  Element* __restrict__ out = nullptr;
  int m = 0;
  int n = 0;
  int cols = 0;  // ceil_div(n, ElemsPerItem)
};

template <typename Element, int ElemsPerItem, bool Interleaved, bool HasWeights>
class SiluAndMulKernel;

template <typename Element, int ElemsPerItem, bool Interleaved, bool HasWeights>
sycl::event launch_silu_and_mul(sycl::queue& queue, KernelParams<Element> params) {
  // Vectorized (8-element, 4x uint64) path only for the 16-bit types on widths
  // that divide evenly; everything else takes the scalar path.
  constexpr bool kVectorizable = (ElemsPerItem == 8) && !std::is_same_v<Element, float>;
  // The 8-byte accesses need an 8-byte-aligned base: USM is only guaranteed to
  // be aligned for the element type, so check rather than assume.
  bool const aligned = (reinterpret_cast<uintptr_t>(params.gateup) % 8 == 0) &&
                       (reinterpret_cast<uintptr_t>(params.out) % 8 == 0);
  bool const vector_ok = kVectorizable && aligned && (params.n % ElemsPerItem == 0);

  // Shape the work-group to the row width: narrow rows (N/P as small as 48)
  // would otherwise leave most of a 256-item group idle.
  int rows_per_group = 0;
  int cols_per_group = 0;
  choose_group_shape(params.cols, rows_per_group, cols_per_group);
  sycl::range<2> local(static_cast<std::size_t>(rows_per_group),
                       static_cast<std::size_t>(cols_per_group));
  sycl::range<2> global(static_cast<std::size_t>(round_up(params.m, rows_per_group)),
                        static_cast<std::size_t>(round_up(params.cols, cols_per_group)));

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SiluAndMulKernel<Element, ElemsPerItem, Interleaved, HasWeights>>(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) {
          int row = static_cast<int>(item.get_global_id(0));
          int col = static_cast<int>(item.get_global_id(1));
          if (row >= params.m || col >= params.cols) {
            return;
          }

          int n0 = col * ElemsPerItem;
          Element const* gateup_row =
              params.gateup + static_cast<int64_t>(row) * (2 * params.n);
          Element* out_row = params.out + static_cast<int64_t>(row) * params.n;
          float weight = HasWeights ? params.weights[row] : 1.0f;

          bool full_tile = (n0 + ElemsPerItem) <= params.n;

          if constexpr (kVectorizable) {
            if (vector_ok && full_tile) {
              float acc[ElemsPerItem];
              if constexpr (Interleaved) {
                // 8 outputs = 16 consecutive elements = 4x uint64; each uint64
                // holds (g[2j], u[2j], g[2j+1], u[2j+1]).
                Element const* p = gateup_row + 2 * n0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  uint64_t raw = *reinterpret_cast<uint64_t const*>(p + 4 * j);
                  acc[2 * j] = silu_and_mul_f32(to_float(element_from_raw<Element>(raw, 0)),
                                                to_float(element_from_raw<Element>(raw, 1)));
                  acc[2 * j + 1] = silu_and_mul_f32(to_float(element_from_raw<Element>(raw, 2)),
                                                    to_float(element_from_raw<Element>(raw, 3)));
                }
              } else {
                Element const* pg = gateup_row + n0;
                Element const* pu = gateup_row + params.n + n0;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                  uint64_t rg = *reinterpret_cast<uint64_t const*>(pg + 4 * j);
                  uint64_t ru = *reinterpret_cast<uint64_t const*>(pu + 4 * j);
#pragma unroll
                  for (int i = 0; i < 4; ++i) {
                    acc[4 * j + i] =
                        silu_and_mul_f32(to_float(element_from_raw<Element>(rg, i)),
                                         to_float(element_from_raw<Element>(ru, i)));
                  }
                }
              }
              if (HasWeights) {
#pragma unroll
                for (int i = 0; i < ElemsPerItem; ++i) {
                  acc[i] *= weight;
                }
              }
              uint64_t out_raw[2] = {0, 0};
#pragma unroll
              for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  out_raw[j] |= static_cast<uint64_t>(from_float<Element>(acc[4 * j + i]).raw())
                                << (16 * i);
                }
              }
              *reinterpret_cast<uint64_t*>(out_row + n0) = out_raw[0];
              *reinterpret_cast<uint64_t*>(out_row + n0 + 4) = out_raw[1];
              return;
            }
          }

#pragma unroll
          for (int i = 0; i < ElemsPerItem; ++i) {
            int n = n0 + i;
            if (n >= params.n) {
              continue;
            }
            float gate;
            float up;
            if constexpr (Interleaved) {
              gate = to_float(gateup_row[2 * n]);
              up = to_float(gateup_row[2 * n + 1]);
            } else {
              gate = to_float(gateup_row[n]);
              up = to_float(gateup_row[params.n + n]);
            }
            float value = silu_and_mul_f32(gate, up);
            if (HasWeights) {
              value *= weight;
            }
            out_row[n] = from_float<Element>(value);
          }
        });
  });
}

inline int choose_elems_per_item(CaseConfig const& cfg, Options const& options) {
  if (options.elems_per_item != 0) {
    return options.elems_per_item;
  }
  if (cfg.elems_per_item != 0) {
    // Case-level pin. The auto rule below always picks an ElemsPerItem that
    // divides N, so the partial-tile path is unreachable without a pin: the
    // quick suite pins a few cases to cover it.
    return cfg.elems_per_item;
  }
  if (cfg.n % 8 == 0) {
    return 8;
  }
  if (cfg.n % 4 == 0) {
    return 4;
  }
  return 1;
}

template <typename Element, bool Interleaved, bool HasWeights>
sycl::event dispatch_epi(
    sycl::queue& queue, KernelParams<Element> params, int elems_per_item) {
  if (elems_per_item == 8) {
    return launch_silu_and_mul<Element, 8, Interleaved, HasWeights>(queue, params);
  }
  if (elems_per_item == 4) {
    return launch_silu_and_mul<Element, 4, Interleaved, HasWeights>(queue, params);
  }
  return launch_silu_and_mul<Element, 1, Interleaved, HasWeights>(queue, params);
}

template <typename Element>
sycl::event dispatch_silu_and_mul(
    sycl::queue& queue,
    KernelParams<Element> params,
    bool interleaved,
    bool has_weights,
    int elems_per_item) {
  if (interleaved) {
    return has_weights ? dispatch_epi<Element, true, true>(queue, params, elems_per_item)
                       : dispatch_epi<Element, true, false>(queue, params, elems_per_item);
  }
  return has_weights ? dispatch_epi<Element, false, true>(queue, params, elems_per_item)
                     : dispatch_epi<Element, false, false>(queue, params, elems_per_item);
}

// ---------------------------------------------------------------------------
// Host reference and data generation
// ---------------------------------------------------------------------------

template <typename Element>
std::vector<Element> make_random_elements(std::size_t count, uint32_t seed) {
  std::vector<Element> data(count);
  std::mt19937 gen(seed);
  // Wide enough to cover silu's saturating and near-zero regimes.
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = from_float<Element>(dist(gen));
  }
  return data;
}

inline std::vector<float> make_random_weights(std::size_t count, uint32_t seed) {
  std::vector<float> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  return data;
}

template <typename Element>
std::vector<Element> reference_silu_and_mul(
    CaseConfig const& cfg,
    std::vector<Element> const& gateup,
    std::vector<float> const& weights) {
  std::vector<Element> out(static_cast<std::size_t>(cfg.m) * cfg.n);
  for (int row = 0; row < cfg.m; ++row) {
    std::size_t in_base = static_cast<std::size_t>(row) * (2 * cfg.n);
    std::size_t out_base = static_cast<std::size_t>(row) * cfg.n;
    float weight = cfg.has_weights ? weights[static_cast<std::size_t>(row)] : 1.0f;
    for (int n = 0; n < cfg.n; ++n) {
      float gate;
      float up;
      if (cfg.interleaved) {
        gate = to_float(gateup[in_base + 2 * n]);
        up = to_float(gateup[in_base + 2 * n + 1]);
      } else {
        gate = to_float(gateup[in_base + n]);
        up = to_float(gateup[in_base + cfg.n + n]);
      }
      float value = silu_and_mul_f32(gate, up);
      if (cfg.has_weights) {
        value *= weight;
      }
      out[out_base + n] = from_float<Element>(value);
    }
  }
  return out;
}

// Deliberately WRONG variant kept as a discriminator: it rounds to the output
// dtype after every multiply, the way the stock sglang
// ops/elementwise/elementwise.py silu_and_mul_kernel does ("cast down before
// mul to better match training"). It differs from the Inkling kernel by about
// one output ulp, which a 1-ulp tolerance alone would happily accept -- so the
// verifier instead requires the single-cast reference to be the better
// bit-exact match. Without this, the example could not tell the two kernels
// apart, and cast placement is the point of the example.
template <typename Element>
std::vector<Element> reference_silu_and_mul_cast_per_multiply(
    CaseConfig const& cfg,
    std::vector<Element> const& gateup,
    std::vector<float> const& weights) {
  std::vector<Element> out(static_cast<std::size_t>(cfg.m) * cfg.n);
  for (int row = 0; row < cfg.m; ++row) {
    std::size_t in_base = static_cast<std::size_t>(row) * (2 * cfg.n);
    std::size_t out_base = static_cast<std::size_t>(row) * cfg.n;
    float weight = cfg.has_weights ? weights[static_cast<std::size_t>(row)] : 1.0f;
    for (int n = 0; n < cfg.n; ++n) {
      float gate;
      float up;
      if (cfg.interleaved) {
        gate = to_float(gateup[in_base + 2 * n]);
        up = to_float(gateup[in_base + 2 * n + 1]);
      } else {
        gate = to_float(gateup[in_base + n]);
        up = to_float(gateup[in_base + cfg.n + n]);
      }
      float sigmoid = 1.0f / (1.0f + sycl::exp(-gate));
      Element act = from_float<Element>(gate * sigmoid);
      Element prod = from_float<Element>(to_float(act) * up);
      out[out_base + n] =
          cfg.has_weights ? from_float<Element>(to_float(prod) * weight) : prod;
    }
  }
  return out;
}

template <typename Element>
std::size_t count_bit_exact(
    std::vector<Element> const& got, std::vector<Element> const& expected) {
  std::size_t exact = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    exact += (host_bits(got[i]) == host_bits(expected[i])) ? 1 : 0;
  }
  return exact;
}

struct VerifyResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  int max_ulps = 0;
  std::size_t index = 0;
  uint32_t got_bits = 0;
  uint32_t expected_bits = 0;
};

template <typename Element>
VerifyResult compare_outputs(
    std::vector<Element> const& got, std::vector<Element> const& expected) {
  if (got.size() != expected.size()) {
    throw std::invalid_argument("compare_outputs size mismatch");
  }
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double g = static_cast<double>(to_float(got[i]));
    double e = static_cast<double>(to_float(expected[i]));
    double abs_err = std::abs(g - e);
    double rel_err = abs_err / std::max(1.0, std::abs(e));
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);

    bool ok;
    if constexpr (std::is_same_v<Element, float>) {
      // The device and host exp() differ by an ulp or two; the rest of the
      // chain is exact.
      ok = abs_err <= 1.0e-5 + 1.0e-5 * std::abs(e);
    } else {
      // 16-bit outputs: the fp32 chain is identical apart from exp(), so a
      // 1-ulp store difference is the most a correct kernel can show.
      int ulps = std::abs(ordered_raw16(host_bits(got[i])) - ordered_raw16(host_bits(expected[i])));
      result.max_ulps = std::max(result.max_ulps, ulps);
      ok = ulps <= 1;
    }
    if (!ok && result.passed) {
      result.passed = false;
      result.index = i;
      result.got_bits = host_bits(got[i]);
      result.expected_bits = host_bits(expected[i]);
    }
  }
  return result;
}

inline std::size_t traffic_bytes(CaseConfig const& cfg, std::size_t element_bytes) {
  std::size_t m = static_cast<std::size_t>(cfg.m);
  std::size_t n = static_cast<std::size_t>(cfg.n);
  std::size_t bytes = m * (2 * n) * element_bytes + m * n * element_bytes;
  if (cfg.has_weights) {
    bytes += m * sizeof(float);
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// Suites
//
// Inkling shapes: M = T * top_k with top_k = 6 (num_experts_per_tok), N = I / P
// for I in {384 checkpoint routed, 3072 production routed / dense} and
// P (TP) in {1, 2, 4, 8}. The dense / shared-expert MLP runs the same op with
// dense_intermediate_size = 3072 and no routing weight (top_k folded away), so
// its rows are T, not T * top_k.
// ---------------------------------------------------------------------------

inline std::vector<CaseConfig> quick_suite() {
  // Correctness-shaped: odd widths, tails, both layouts, weights on and off.
  return {
      {"tiny_interleaved_w", 3, 5, true, true, kNoTarget},
      {"tiny_contiguous_w", 3, 5, false, true, kNoTarget},
      {"tiny_interleaved_dense", 4, 7, true, false, kNoTarget},
      {"tiny_contiguous_dense", 4, 7, false, false, kNoTarget},
      {"epi4_interleaved", 17, 12, true, true, kNoTarget},
      {"epi4_contiguous", 17, 12, false, true, kNoTarget},
      {"vec8_interleaved", 33, 48, true, true, kNoTarget},
      {"vec8_contiguous", 33, 48, false, true, kNoTarget},
      {"epi4_n52_interleaved", 9, 52, true, true, kNoTarget},
      // Pinned ElemsPerItem so the partial-tile / bounds-checked path actually
      // runs: the auto rule always picks an ElemsPerItem that divides N.
      {"pin_epi8_tail_interleaved", 17, 12, true, true, kNoTarget, 8},
      {"pin_epi8_tail_contiguous", 17, 12, false, true, kNoTarget, 8},
      {"pin_epi8_tail_n52", 9, 52, true, true, kNoTarget, 8},
      {"pin_epi8_tail_dense", 5, 21, true, false, kNoTarget, 8},
      {"pin_epi4_tail_interleaved", 5, 5, true, true, kNoTarget, 4},
      {"pin_epi4_tail_contiguous", 5, 5, false, true, kNoTarget, 4},
      {"wide_row_interleaved", 6, 3072, true, true, kNoTarget},
      {"wide_row_contiguous", 6, 3072, false, true, kNoTarget},
  };
}

inline std::vector<CaseConfig> inkling_suite() {
  std::vector<CaseConfig> cases;
  int const tokens[] = {1, 9, 144, 4096};
  int const top_k = 6;
  // Checkpoint routed I=384 and production routed I=3072 over TP=1/2/4/8.
  int const routed_n[] = {384, 192, 96, 48, 3072, 1536, 768, 384};
  int const routed_i[] = {384, 384, 384, 384, 3072, 3072, 3072, 3072};
  int const routed_p[] = {1, 2, 4, 8, 1, 2, 4, 8};

  for (int t : tokens) {
    for (int k = 0; k < 8; ++k) {
      CaseConfig cfg;
      cfg.m = t * top_k;
      cfg.n = routed_n[k];
      cfg.interleaved = true;
      cfg.has_weights = true;
      cfg.name = "routed_i" + std::to_string(routed_i[k]) + "_tp" +
                 std::to_string(routed_p[k]) + "_t" + std::to_string(t);
      cases.push_back(cfg);
    }
  }

  // Non-interleaved (LoRA / de-interleaved w13) coverage on the production
  // widths, so both kernels see real shapes.
  for (int t : {9, 144, 4096}) {
    for (int k = 0; k < 4; ++k) {
      CaseConfig cfg;
      cfg.m = t * top_k;
      cfg.n = routed_n[4 + k];
      cfg.interleaved = false;
      cfg.has_weights = true;
      cfg.name = "routed_contig_i3072_tp" + std::to_string(routed_p[4 + k]) + "_t" +
                 std::to_string(t);
      cases.push_back(cfg);
    }
  }

  // Dense / shared-expert MLP: dense_intermediate_size = 3072 over TP, one row
  // per token, no routing weight (InklingSwiglu).
  for (int t : {1, 9, 144, 4096}) {
    for (int k = 0; k < 4; ++k) {
      CaseConfig cfg;
      cfg.m = t;
      cfg.n = routed_n[4 + k];
      cfg.interleaved = true;
      cfg.has_weights = false;
      cfg.name =
          "dense_i3072_tp" + std::to_string(routed_p[4 + k]) + "_t" + std::to_string(t);
      cases.push_back(cfg);
    }
  }
  // Shared experts feed the gate gammas through the topk-weight slot.
  for (int t : {9, 144}) {
    CaseConfig cfg;
    cfg.m = t * 2;  // n_shared_experts = 2
    cfg.n = 3072;
    cfg.interleaved = true;
    cfg.has_weights = true;
    cfg.name = "shared_i3072_tp1_t" + std::to_string(t);
    cases.push_back(cfg);
  }
  return cases;
}

inline std::vector<CaseConfig> perf_suite() {
  std::vector<CaseConfig> cases;
  int const top_k = 6;
  // Prefill bands: T = 4096 and T = max_prefill_tokens = 16384.
  struct Band {
    int t;
    int n;
    int intermediate;
    int p;
    double target;
  };
  Band const bands[] = {
      {4096, 384, 384, 1, kDramTargetGBps},
      {4096, 48, 384, 8, kL2ResidentTargetGBps},
      {4096, 3072, 3072, 1, kDramTargetGBps},
      {4096, 384, 3072, 8, kDramTargetGBps},
      {16384, 384, 384, 1, kDramTargetGBps},
      {16384, 48, 384, 8, kSmallL2TargetGBps},
      {16384, 3072, 3072, 1, kDramTargetGBps},
      {16384, 384, 3072, 8, kDramTargetGBps},
  };
  for (Band const& band : bands) {
    CaseConfig cfg;
    cfg.m = band.t * top_k;
    cfg.n = band.n;
    cfg.interleaved = true;
    cfg.has_weights = true;
    cfg.target_gbps = band.target;
    cfg.name = "perf_routed_i" + std::to_string(band.intermediate) + "_tp" +
               std::to_string(band.p) + "_t" + std::to_string(band.t);
    cases.push_back(cfg);
    // Same shape, non-interleaved: the two-stream read is the interesting
    // bandwidth comparison (it measures within noise of the interleaved read).
    CaseConfig contig = cfg;
    contig.interleaved = false;
    contig.name = "perf_contig_i" + std::to_string(band.intermediate) + "_tp" +
                  std::to_string(band.p) + "_t" + std::to_string(band.t);
    cases.push_back(contig);
  }
  // Dense path at the production width, no routing weight (InklingSwiglu).
  for (int t : {4096, 16384}) {
    CaseConfig cfg;
    cfg.m = t;
    cfg.n = 3072;
    cfg.interleaved = true;
    cfg.has_weights = false;
    cfg.target_gbps = kDramTargetGBps;
    cfg.name = "perf_dense_i3072_tp1_t" + std::to_string(t);
    cases.push_back(cfg);
  }
  return cases;
}

inline std::vector<CaseConfig> make_suite(std::string const& suite) {
  if (suite == "quick") {
    return quick_suite();
  }
  if (suite == "inkling") {
    return inkling_suite();
  }
  if (suite == "perf") {
    return perf_suite();
  }
  return {};
}

// ---------------------------------------------------------------------------
// Option / shape parsing
// ---------------------------------------------------------------------------

inline char const* dtype_text(DType dtype) {
  switch (dtype) {
    case DType::kAll:
      return "all";
    case DType::kFloat:
      return "float";
    case DType::kBf16:
      return "bf16";
    case DType::kFp16:
      return "fp16";
  }
  return "unknown";
}

inline bool parse_dtype(std::string const& text, DType& dtype) {
  if (text == "all") {
    dtype = DType::kAll;
  } else if (text == "float" || text == "fp32") {
    dtype = DType::kFloat;
  } else if (text == "bf16") {
    dtype = DType::kBf16;
  } else if (text == "fp16" || text == "half") {
    dtype = DType::kFp16;
  } else {
    return false;
  }
  return true;
}

inline char const* layout_text(LayoutFilter layout) {
  switch (layout) {
    case LayoutFilter::kAll:
      return "all";
    case LayoutFilter::kInterleaved:
      return "interleaved";
    case LayoutFilter::kContiguous:
      return "contiguous";
  }
  return "unknown";
}

inline bool parse_layout(std::string const& text, LayoutFilter& layout) {
  if (text == "all") {
    layout = LayoutFilter::kAll;
  } else if (text == "interleaved") {
    layout = LayoutFilter::kInterleaved;
  } else if (text == "contiguous" || text == "non_interleaved") {
    layout = LayoutFilter::kContiguous;
  } else {
    return false;
  }
  return true;
}

template <typename Element>
std::string element_dtype_text() {
  if constexpr (std::is_same_v<Element, float>) {
    return "float";
  } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  } else if constexpr (std::is_same_v<Element, cutlass::half_t>) {
    return "fp16";
  } else {
    return "unknown";
  }
}

inline bool parse_bool_text(std::string const& value, bool& out) {
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    out = false;
    return true;
  }
  return false;
}

inline bool parse_shape_impl(std::string const& text, CaseConfig& cfg) {
  if (text.empty()) {
    return true;
  }
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto eq = item.find('=');
    if (eq == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, eq);
    std::string value = item.substr(eq + 1);
    if (key == "name") {
      cfg.name = value;
    } else if (key == "m" || key == "rows") {
      cfg.m = std::stoi(value);
    } else if (key == "n" || key == "inter") {
      cfg.n = std::stoi(value);
    } else if (key == "t" || key == "tokens") {
      cfg.m = std::stoi(value) * 6;  // top_k = num_experts_per_tok
    } else if (key == "interleaved") {
      if (!parse_bool_text(value, cfg.interleaved)) {
        return false;
      }
    } else if (key == "weights" || key == "topk_weights") {
      if (!parse_bool_text(value, cfg.has_weights)) {
        return false;
      }
    } else if (key == "epi" || key == "elems_per_item") {
      cfg.elems_per_item = std::stoi(value);
    } else if (key == "target" || key == "target_gbps") {
      cfg.target_gbps = std::stod(value);
    } else {
      return false;
    }
  }
  return true;
}

// std::stoi / std::stod throw on garbage input; keep that inside the parser so
// callers get the same clean "invalid --shape" path as a structural error.
inline bool parse_shape(std::string const& text, CaseConfig& cfg) {
  try {
    return parse_shape_impl(text, cfg);
  } catch (std::exception const&) {
    return false;
  }
}

inline void validate_case(CaseConfig& cfg) {
  if (cfg.m <= 0 || cfg.n <= 0) {
    throw std::invalid_argument("case has non-positive shape");
  }
  // Only these three are instantiated; anything else would make the launch's
  // column count disagree with the kernel's ElemsPerItem.
  if (!(cfg.elems_per_item == 0 || cfg.elems_per_item == 1 || cfg.elems_per_item == 4 ||
        cfg.elems_per_item == 8)) {
    throw std::invalid_argument("case elems_per_item must be 0, 1, 4, or 8");
  }
  if (cfg.name.empty()) {
    cfg.name = "custom";
  }
}

inline void print_usage(char const* name) {
  std::cout
      << "Usage: " << name << " [options]\n\n"
      << "Options:\n"
      << "  --suite=quick|inkling|perf     Built-in suite (default quick)\n"
      << "  --shape=m=<int>,n=<int>[,t=<int>][,interleaved=0|1][,weights=0|1][,epi=1|4|8][,target=<f>]\n"
      << "  --dtype=all|float|bf16|fp16    Element dtype (default all: bf16 then fp16)\n"
      << "  --layout=all|interleaved|contiguous  Filter suite cases by gate/up layout\n"
      << "  --elems-per-item=0|1|4|8       Outputs per work-item, 0 = auto (default 0)\n"
      << "  --iterations=<int>             Timed iterations (default 20)\n"
      << "  --warmup=<int>                 Warmup launches (default 5)\n"
      << "  --verify=0|1                   CPU reference comparison (default 1)\n"
      << "  --benchmark=0|1                Profiling-event timing (default 1)\n"
      << "  --perf-threshold-scale=<f>     Scale perf gates; 0 disables them\n";
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

template <typename Element>
bool run_case_for_dtype(sycl::queue& queue, CaseConfig cfg, Options const& options) {
  validate_case(cfg);
  int elems_per_item = choose_elems_per_item(cfg, options);
  std::size_t input_count = static_cast<std::size_t>(cfg.m) * (2 * cfg.n);
  std::size_t output_count = static_cast<std::size_t>(cfg.m) * cfg.n;
  bool do_verify = options.verify && output_count <= kVerifyMaxOutputElements;

  DeviceBuffer<Element> d_gateup(queue, input_count);
  DeviceBuffer<float> d_weights(queue, static_cast<std::size_t>(cfg.m));
  DeviceBuffer<Element> d_out(queue, output_count);

  std::vector<Element> host_gateup;
  std::vector<float> host_weights = make_random_weights(static_cast<std::size_t>(cfg.m), 31);
  d_weights.copy_from(host_weights);
  if (do_verify) {
    host_gateup = make_random_elements<Element>(input_count, 17);
    d_gateup.copy_from(host_gateup);
  } else {
    // Random tile, replicated: constant fills would let Xe memory compression
    // report bandwidth the DRAM never delivered.
    d_gateup.fill_tiled(make_random_elements<Element>(std::min<std::size_t>(input_count, 1u << 20), 17));
  }

  KernelParams<Element> params;
  params.gateup = d_gateup.get();
  params.weights = d_weights.get();
  params.out = d_out.get();
  params.m = cfg.m;
  params.n = cfg.n;
  params.cols = ceil_div(cfg.n, elems_per_item);

  bool passed = true;
  std::cout << "  [" << element_dtype_text<Element>() << "] " << cfg.name
            << " m=" << cfg.m << " n=" << cfg.n
            << " layout=" << (cfg.interleaved ? "interleaved" : "contiguous")
            << " weights=" << (cfg.has_weights ? "1" : "0")
            << " epi=" << elems_per_item << "\n";

  if (do_verify) {
    dispatch_silu_and_mul(queue, params, cfg.interleaved, cfg.has_weights, elems_per_item)
        .wait();
    std::vector<Element> host_out(output_count);
    d_out.copy_to(host_out);
    std::vector<Element> expected = reference_silu_and_mul(cfg, host_gateup, host_weights);
    VerifyResult result = compare_outputs(host_out, expected);
    if (!result.passed) {
      std::cerr << "    verify=FAIL index=" << result.index << " got_bits=0x" << std::hex
                << result.got_bits << " expected_bits=0x" << result.expected_bits << std::dec
                << " max_abs=" << result.max_abs << " max_rel=" << result.max_rel
                << " max_ulps=" << result.max_ulps << "\n";
      passed = false;
    } else {
      std::cout << "    verify=PASS max_abs=" << result.max_abs
                << " max_rel=" << result.max_rel << " max_ulps=" << result.max_ulps << "\n";
    }
    if constexpr (!std::is_same_v<Element, float>) {
      std::vector<Element> alt =
          reference_silu_and_mul_cast_per_multiply(cfg, host_gateup, host_weights);
      std::size_t exact_single = count_bit_exact(host_out, expected);
      std::size_t exact_alt = count_bit_exact(host_out, alt);
      std::cout << "    cast_order: bit_exact_single_cast=" << exact_single << "/"
                << host_out.size() << " bit_exact_cast_per_multiply=" << exact_alt << "\n";
      if (exact_alt > exact_single) {
        std::cerr << "    cast_order=FAIL kernel matches the cast-per-multiply variant, "
                     "not Inkling's single rounding cast\n";
        passed = false;
      }
    }
  } else if (options.verify) {
    std::cout << "    verify=SKIP output too large for the CPU reference\n";
  }

  if (options.benchmark) {
    for (int i = 0; i < options.warmup; ++i) {
      dispatch_silu_and_mul(queue, params, cfg.interleaved, cfg.has_weights, elems_per_item)
          .wait();
    }
    double total_ms = 0.0;
    int timing_iterations = std::max(options.iterations, 1);
    for (int i = 0; i < timing_iterations; ++i) {
      sycl::event event = dispatch_silu_and_mul(
          queue, params, cfg.interleaved, cfg.has_weights, elems_per_item);
      event.wait();
      total_ms += event_ms(event);
    }
    double avg_ms = total_ms / static_cast<double>(timing_iterations);
    std::size_t bytes = traffic_bytes(cfg, sizeof(Element));
    double gbps = (static_cast<double>(bytes) / kBytesPerGB) / (avg_ms * 1.0e-3);
    double target = cfg.target_gbps * options.perf_threshold_scale;
    // Formatted through a local stream so the global cout flags/precision are
    // left alone (the verify prints below and above want full precision).
    std::ostringstream line;
    line << std::fixed << std::setprecision(4) << "    avg_ms=" << avg_ms
         << std::setprecision(6) << " GB=" << (static_cast<double>(bytes) / kBytesPerGB)
         << std::setprecision(3) << " effective_GBps=" << gbps;
    if (target > 0.0) {
      line << " target_GBps=" << target;
    }
    std::cout << line.str() << "\n";
    if (target > 0.0 && gbps < target) {
      std::cerr << "    perf=FAIL target_GBps=" << target << "\n";
      passed = false;
    }
  }

  return passed;
}

template <typename Element>
bool run_cases_for_dtype(
    sycl::queue& queue, std::vector<CaseConfig> const& cases, Options const& options) {
  bool all_passed = true;
  for (CaseConfig cfg : cases) {
    all_passed &= run_case_for_dtype<Element>(queue, std::move(cfg), options);
  }
  return all_passed;
}

inline std::vector<CaseConfig> filter_layout(
    std::vector<CaseConfig> cases, LayoutFilter layout) {
  if (layout == LayoutFilter::kAll) {
    return cases;
  }
  bool want_interleaved = layout == LayoutFilter::kInterleaved;
  std::vector<CaseConfig> out;
  for (CaseConfig& cfg : cases) {
    if (cfg.interleaved == want_interleaved) {
      out.push_back(std::move(cfg));
    }
  }
  return out;
}

}  // namespace cutlass::examples::bmg_moe_silu_and_mul

int main(int argc, char const** argv) {
  namespace silu = cutlass::examples::bmg_moe_silu_and_mul;

  cutlass::CommandLine cmd(argc, argv);
  silu::Options options;
  if (cmd.check_cmd_line_flag("help")) {
    silu::print_usage(argv[0]);
    return 0;
  }
  cmd.get_cmd_line_argument("suite", options.suite, options.suite);
  cmd.get_cmd_line_argument("shape", options.shape, options.shape);
  cmd.get_cmd_line_argument("iterations", options.iterations, options.iterations);
  cmd.get_cmd_line_argument("warmup", options.warmup, options.warmup);
  cmd.get_cmd_line_argument("elems-per-item", options.elems_per_item, options.elems_per_item);
  cmd.get_cmd_line_argument(
      "perf-threshold-scale", options.perf_threshold_scale, options.perf_threshold_scale);
  int verify = options.verify ? 1 : 0;
  cmd.get_cmd_line_argument("verify", verify, verify);
  options.verify = verify != 0;
  int benchmark = options.benchmark ? 1 : 0;
  cmd.get_cmd_line_argument("benchmark", benchmark, benchmark);
  options.benchmark = benchmark != 0;

  std::string dtype_arg = silu::dtype_text(options.dtype);
  cmd.get_cmd_line_argument("dtype", dtype_arg, dtype_arg);
  if (!silu::parse_dtype(dtype_arg, options.dtype)) {
    std::cerr << "Unknown dtype: " << dtype_arg << "\n";
    silu::print_usage(argv[0]);
    return -1;
  }
  std::string layout_arg = silu::layout_text(options.layout);
  cmd.get_cmd_line_argument("layout", layout_arg, layout_arg);
  if (!silu::parse_layout(layout_arg, options.layout)) {
    std::cerr << "Unknown layout: " << layout_arg << "\n";
    silu::print_usage(argv[0]);
    return -1;
  }
  if (options.iterations < 0 || options.warmup < 0) {
    std::cerr << "iterations and warmup must be non-negative\n";
    return -1;
  }
  if (!(options.elems_per_item == 0 || options.elems_per_item == 1 ||
        options.elems_per_item == 4 || options.elems_per_item == 8)) {
    std::cerr << "--elems-per-item must be 0, 1, 4, or 8\n";
    return -1;
  }

  std::vector<silu::CaseConfig> cases;
  if (!options.shape.empty()) {
    silu::CaseConfig cfg;
    cfg.name = "custom";
    if (!silu::parse_shape(options.shape, cfg)) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
    }
    cases.push_back(cfg);
  } else {
    cases = silu::filter_layout(silu::make_suite(options.suite), options.layout);
    if (cases.empty()) {
      std::cerr << "Unknown suite (or empty after --layout filter): " << options.suite << "\n";
      return -1;
    }
  }

  try {
    sycl::queue queue = silu::make_queue();
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "24_bmg_moe_silu_and_mul: out[m,n] = Element(silu(gate)*up*weight[m]), fp32 math, one cast\n";
    std::cout << "Suite=" << options.suite << " dtype=" << silu::dtype_text(options.dtype)
              << " layout=" << silu::layout_text(options.layout)
              << " iterations=" << options.iterations << " warmup=" << options.warmup
              << " verify=" << (options.verify ? "true" : "false")
              << " benchmark=" << (options.benchmark ? "true" : "false")
              << " elems_per_item=" << options.elems_per_item
              << " perf_threshold_scale=" << options.perf_threshold_scale << "\n";

    bool all_passed = true;
    if (options.dtype == silu::DType::kAll || options.dtype == silu::DType::kBf16) {
      all_passed &= silu::run_cases_for_dtype<cutlass::bfloat16_t>(queue, cases, options);
    }
    if (options.dtype == silu::DType::kAll || options.dtype == silu::DType::kFp16) {
      all_passed &= silu::run_cases_for_dtype<cutlass::half_t>(queue, cases, options);
    }
    if (options.dtype == silu::DType::kFloat) {
      all_passed &= silu::run_cases_for_dtype<float>(queue, cases, options);
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
