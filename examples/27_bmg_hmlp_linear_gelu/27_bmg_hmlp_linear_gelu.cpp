/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Inkling HMLP (vision patch encoder) per-layer compute for CUTLASS SYCL on BMG.
 *
 * Model reference (sglang):
 *   python/sglang/srt/models/inkling_common/hmlp.py  -- HMLPPatchEncoder, plan_out_scales
 *   python/sglang/srt/models/inkling_common/norm.py  -- RMSNorm(hidden_size, eps=1e-6)
 *   python/sglang/srt/configs/inkling.py             -- InklingVisionConfig
 *
 * Semantics mirrored here, per HMLP layer i of n_layers:
 *
 *   x = fold_timespace_to_depth(x, t_fold, hw_fold)   <- example 22 (data movement only)
 *   x = Linear(start_c * shuffle_mult, out, bias=False)(x)
 *   if i < n_layers - 1:                              <- POST-linear norm, no norm on last layer
 *       x = RMSNorm(out)(x)                           <- weighted, eps = 1e-6
 *       x = F.gelu(x)                                 <- EXACT erf GELU (approximate='none')
 *   # the last layer maps to decoder_dmodel and has no in-layer norm and no GELU
 *   if use_vision_norm:
 *       x = RMSNorm(decoder_dmodel)(x)                <- final norm after the last layer
 *
 * This example covers exactly the compute that example 22 (fold_timespace_to_depth)
 * leaves untested: the Linear, the RMSNorm, and the GELU. The fold itself is NOT
 * re-implemented here; between layers this example reinterprets the previous
 * layer's [rows, out] output as [rows / shuffle_mult, out * shuffle_mult], which
 * has the identical shape/arithmetic of the folded tensor but not its permutation.
 * The permutation is example 22's subject.
 *
 * Layer widths are DERIVED from plan_out_scales() (transcribed below), not
 * hard-coded, so a config change moves the shapes here as well. For the shipped
 * checkpoint (temporal_patch_size=2, patch_size=40, n_layers=4, n_channels=3,
 * decoder_dmodel=768, use_vision_norm=true) the unfiltered ladder of channel
 * widths is 3 -> 128 -> 320 -> 1216 -> 4800 -> 9600 and the assignment selects
 * scales (1,1,1,3), (1,5,5,128), (1,10,10,320), (1,40,40,4800), (2,40,40,9600),
 * giving the layers
 *
 *   layer 0:   75 ->  128  + RMSNorm(128)  + GELU
 *   layer 1:  512 ->  320  + RMSNorm(320)  + GELU
 *   layer 2: 5120 -> 4800  + RMSNorm(4800) + GELU
 *   layer 3: 9600 ->  768  (decoder_dmodel), then final RMSNorm(768)
 *
 * and for InklingVisionConfig defaults (temporal_patch_size=1, patch_size=16,
 * n_layers=1) a single layer 768 -> decoder_dmodel with no GELU.
 *
 * dtype: bf16 activations and weights with fp32 accumulate, matching the shipped
 * checkpoint (torch_dtype bfloat16). The Linear is delegated to oneMKL
 * (bfloat16, bfloat16, bfloat16, float) row-major GEMM -- the same choice as
 * example 20 -- because the weight can then be consumed in its exact nn.Linear
 * [out_features, in_features] layout via transb, and because an in-repo GEMM is
 * not what this example is testing. RMSNorm + GELU is a plain SYCL kernel (no
 * ESIMD). The norm cannot be folded into the GEMM epilogue: it reduces over the
 * full output row, so it needs the completed GEMM row.
 *
 * Roofline summary:
 *   The Linear is compute-bound at large row counts (2*M*K*N FLOPs against
 *   (M*K + K*N + M*N) * 2 bytes) and launch/bandwidth-bound at decode-like row
 *   counts. The RMSNorm+GELU pass is pure streaming: it reads each element twice
 *   (sum-of-squares pass, then the scaled write) and writes it once, i.e. about
 *   3 * M * N * 2 bytes for ~6 FLOPs per element, so it is memory-bound and the
 *   useful metric there is GB/s. Both are reported per case.
 **************************************************************************************************/

#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cutlass::examples::bmg_hmlp_linear_gelu {

// RMSNorm(hidden_size, eps=1e-6) from inkling_common/norm.py.
constexpr float kRmsNormEps = 1e-6f;
// Maximum work-group size used by the RMSNorm+GELU kernel.
constexpr int kMaxNormThreads = 256;
constexpr int kSubgroupSize = 16;
constexpr double kBytesPerGB = 1.0e9;
// Largest ladder width the brute-force scale assignment will handle. The width
// is 1 + Omega(patch_size) + Omega(temporal_patch_size), so 20 covers any patch
// size up to 2^19; the branch-and-bound search below is additionally capped by a
// node budget so a pathological cost matrix fails loudly instead of hanging.
constexpr int kMaxScaleColumns = 20;
constexpr int64_t kAssignmentNodeBudget = 20000000;

enum class DType { kBf16, kFp16 };

inline char const* dtype_text(DType dtype) {
  return dtype == DType::kBf16 ? "bf16" : "fp16";
}

inline bool parse_dtype(std::string const& text, DType& dtype) {
  if (text == "bf16" || text == "bfloat16") {
    dtype = DType::kBf16;
    return true;
  }
  if (text == "fp16" || text == "half" || text == "f16") {
    dtype = DType::kFp16;
    return true;
  }
  return false;
}

inline char const* bool_text(bool value) {
  return value ? "true" : "false";
}

inline int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// ---------------------------------------------------------------------------
// plan_out_scales: transcription of hmlp.py:37-93.
// ---------------------------------------------------------------------------

struct Scale {
  int t = 1;
  int h = 1;
  int w = 1;
  int c = 1;
};

inline std::vector<int> prime_factors(int n) {
  if (n < 1) {
    throw std::invalid_argument("prime_factors requires a positive integer");
  }
  std::vector<int> factors;
  while (n % 2 == 0) {
    factors.push_back(2);
    n /= 2;
  }
  for (int p = 3; static_cast<int64_t>(p) * p <= n; p += 2) {
    while (n % p == 0) {
      factors.push_back(p);
      n /= p;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

// hmlp.py's local _round_up: ceil(x / 64) * 64.
inline int round_up_64(int64_t x) {
  return static_cast<int>(ceil_div(x, 64) * 64);
}

// Rectangular minimum-cost assignment (scipy.optimize.linear_sum_assignment).
// The ladder is short (<= kMaxScaleColumns), so an exhaustive search with
// bounding is both exact and instant. The two shipped Inkling configs have a
// unique optimum, so this matches scipy without needing its tie-break rule.
inline std::vector<int> linear_sum_assignment(
    std::vector<std::vector<double>> const& cost) {
  int rows = static_cast<int>(cost.size());
  int cols = rows > 0 ? static_cast<int>(cost[0].size()) : 0;
  if (rows > cols) {
    throw std::invalid_argument("assignment requires rows <= cols");
  }
  if (cols > kMaxScaleColumns) {
    throw std::invalid_argument("scale ladder too wide for exhaustive assignment");
  }

  std::vector<int> current(rows, -1);
  std::vector<int> best;
  std::vector<bool> used(static_cast<std::size_t>(cols), false);
  double best_cost = std::numeric_limits<double>::infinity();
  int64_t nodes = 0;

  auto search = [&](auto&& self, int row, double accumulated) -> void {
    if (accumulated >= best_cost) {
      return;
    }
    if (++nodes > kAssignmentNodeBudget) {
      throw std::runtime_error("scale assignment exceeded its search budget");
    }
    if (row == rows) {
      best_cost = accumulated;
      best = current;
      return;
    }
    for (int col = 0; col < cols; ++col) {
      if (used[static_cast<std::size_t>(col)]) {
        continue;
      }
      used[static_cast<std::size_t>(col)] = true;
      current[static_cast<std::size_t>(row)] = col;
      self(self, row + 1, accumulated + cost[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
      used[static_cast<std::size_t>(col)] = false;
    }
  };
  search(search, 0, 0.0);

  if (best.empty()) {
    throw std::runtime_error("assignment failed");
  }
  return best;
}

inline std::vector<Scale> plan_out_scales(
    int temporal_patch_size,
    int patch_size,
    int n_layers,
    int n_channels) {
  if (patch_size <= 1) {
    throw std::invalid_argument("patch_size must be greater than 1");
  }
  if (n_layers < 1) {
    throw std::invalid_argument("n_layers must be positive");
  }

  std::vector<Scale> scales;
  scales.push_back(Scale{1, 1, 1, n_channels});

  std::vector<int> hw_factors = prime_factors(patch_size);
  std::reverse(hw_factors.begin(), hw_factors.end());
  int last_h_scale = 1;
  for (int p : hw_factors) {
    last_h_scale *= p;
    int64_t width = static_cast<int64_t>(last_h_scale) * last_h_scale * n_channels;
    scales.push_back(Scale{1, last_h_scale, last_h_scale, round_up_64(width)});
  }

  std::vector<int> t_factors = prime_factors(temporal_patch_size);
  std::reverse(t_factors.begin(), t_factors.end());
  int last_t_scale = 1;
  for (int p : t_factors) {
    last_t_scale *= p;
    int64_t width =
        static_cast<int64_t>(last_h_scale) * last_h_scale * n_channels * last_t_scale;
    scales.push_back(Scale{last_t_scale, last_h_scale, last_h_scale, round_up_64(width)});
  }

  int cols = static_cast<int>(scales.size());
  std::vector<double> log_size_reduction(static_cast<std::size_t>(cols));
  for (int c = 0; c < cols; ++c) {
    Scale const& s = scales[static_cast<std::size_t>(c)];
    double reduction = static_cast<double>(s.t) * s.h * s.w;
    log_size_reduction[static_cast<std::size_t>(c)] = std::log(reduction);
  }

  int rows = n_layers + 1;
  double log_total = std::log(
      static_cast<double>(patch_size) * patch_size * temporal_patch_size * n_channels);
  std::vector<std::vector<double>> cost(
      static_cast<std::size_t>(rows), std::vector<double>(static_cast<std::size_t>(cols), 0.0));
  for (int r = 0; r < rows; ++r) {
    // np.linspace(0, log_total, n_layers + 1)
    double ideal = rows == 1 ? 0.0 : log_total * static_cast<double>(r) / static_cast<double>(rows - 1);
    for (int c = 0; c < cols; ++c) {
      cost[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
          std::abs(ideal - log_size_reduction[static_cast<std::size_t>(c)]);
    }
  }

  std::vector<int> idxs;
  if (n_layers >= cols) {
    idxs.resize(static_cast<std::size_t>(rows));
    for (int r = 0; r < rows; ++r) {
      int argmin = 0;
      for (int c = 1; c < cols; ++c) {
        if (cost[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] <
            cost[static_cast<std::size_t>(r)][static_cast<std::size_t>(argmin)]) {
          argmin = c;
        }
      }
      idxs[static_cast<std::size_t>(r)] = argmin;
    }
  } else {
    idxs = linear_sum_assignment(cost);
  }

  idxs.front() = 0;
  idxs.back() = cols - 1;

  std::vector<Scale> selected;
  selected.reserve(idxs.size());
  for (int idx : idxs) {
    selected.push_back(scales[static_cast<std::size_t>(idx)]);
  }
  return selected;
}

// ---------------------------------------------------------------------------
// HMLP layer plan
// ---------------------------------------------------------------------------

struct VisionConfig {
  int temporal_patch_size = 2;
  int patch_size = 40;
  int n_channels = 3;
  int n_layers = 4;
  int decoder_dmodel = 768;
  bool use_vision_norm = true;
};

struct HmlpLayer {
  int in_features = 0;
  int out_features = 0;
  bool has_norm = false;   // in-layer RMSNorm (all but the last layer)
  bool has_gelu = false;   // exact erf GELU, same layers as has_norm
  bool final_norm = false; // use_vision_norm RMSNorm after the last layer
  int64_t rows_per_patch = 1;
  int t_fold = 1;
  int hw_fold = 1;
  int shuffle_mult = 1;
};

inline std::vector<HmlpLayer> plan_hmlp_layers(VisionConfig const& cfg) {
  std::vector<Scale> scales = plan_out_scales(
      cfg.temporal_patch_size, cfg.patch_size, cfg.n_layers, cfg.n_channels);
  std::vector<HmlpLayer> layers;
  layers.reserve(scales.size() - 1);

  for (std::size_t i = 0; i + 1 < scales.size(); ++i) {
    Scale const& start = scales[i];
    Scale const& end = scales[i + 1];
    HmlpLayer layer;
    layer.t_fold = end.t / start.t;
    layer.hw_fold = end.h / start.h;
    layer.shuffle_mult = (end.t / start.t) * (end.h / start.h) * (end.w / start.w);
    layer.in_features = start.c * layer.shuffle_mult;
    bool is_last = (static_cast<int>(i) == cfg.n_layers - 1);
    layer.out_features = is_last ? cfg.decoder_dmodel : end.c;
    layer.has_norm = !is_last;
    layer.has_gelu = !is_last;
    layer.final_norm = is_last && cfg.use_vision_norm;
    // Output cells per input patch at this layer's end scale.
    layer.rows_per_patch = static_cast<int64_t>(cfg.temporal_patch_size / end.t) *
        (cfg.patch_size / end.h) * (cfg.patch_size / end.w);
    layers.push_back(layer);
  }
  return layers;
}

// ---------------------------------------------------------------------------
// Element helpers
// ---------------------------------------------------------------------------

template <typename Element>
struct MklTraits;

template <>
struct MklTraits<cutlass::bfloat16_t> {
  using type = oneapi::mkl::bfloat16;
  using scalar = float;
};

template <>
struct MklTraits<cutlass::half_t> {
  using type = sycl::half;
  using scalar = sycl::half;
};

template <typename Element>
std::string element_dtype_text() {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  } else {
    return "fp16";
  }
}

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
    // Round-to-nearest-even fp32 -> bf16, NaN preserving.
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

template <typename Element>
uint32_t host_bits(Element value) {
  return static_cast<uint32_t>(value.raw());
}

// Monotone key over the 16-bit float encoding, so |key(a) - key(b)| is the ulp
// distance. Sign-magnitude is mapped to a signed count so that -0 and +0 collide
// at 0 and a pair straddling zero reports its true (small) distance instead of
// ~32768.
inline int ordered_raw16(uint32_t raw) {
  int value = static_cast<int>(raw & 0xffffu);
  int magnitude = value & 0x7fff;
  return (value & 0x8000) ? -magnitude : magnitude;
}

// F.gelu(x) with approximate='none': 0.5 * x * (1 + erf(x / sqrt(2))).
CUTLASS_HOST_DEVICE float gelu_exact(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return 0.5f * x * (1.0f + sycl::erf(x * 0.70710678118654752440f));
#else
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
#endif
}

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
    ptr = sycl::malloc_device<T>(std::max<std::size_t>(count, 1), q);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept {
    queue = other.queue;
    ptr = other.ptr;
    count = other.count;
    other.queue = nullptr;
    other.ptr = nullptr;
    other.count = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      queue = other.queue;
      ptr = other.ptr;
      count = other.count;
      other.queue = nullptr;
      other.ptr = nullptr;
      other.count = 0;
    }
    return *this;
  }

  ~DeviceBuffer() {
    reset();
  }

  void reset() {
    if (ptr != nullptr) {
      sycl::free(ptr, *queue);
    }
    ptr = nullptr;
    queue = nullptr;
    count = 0;
  }

  T* get() const {
    return ptr;
  }

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
};

// ---------------------------------------------------------------------------
// RMSNorm (+ optional exact GELU) kernel
// ---------------------------------------------------------------------------

template <typename Element>
struct NormGeluParams {
  Element const* __restrict__ x = nullptr;
  Element const* __restrict__ weight = nullptr;
  Element* __restrict__ out = nullptr;
  int64_t rows = 0;
  int cols = 0;
  float eps = kRmsNormEps;
};

template <typename Element, bool ApplyGelu>
class RmsNormGeluKernel;

// One work-group per row. The row is read twice (sum-of-squares, then the
// scaled write) rather than staged in local memory: out_features reaches 4800
// here, which no reasonable SLM budget holds, and the second read is L2-hot.
template <typename Element, bool ApplyGelu>
sycl::event launch_rms_norm(sycl::queue& queue, NormGeluParams<Element> params) {
  if (params.rows <= 0 || params.cols <= 0) {
    return {};
  }
  int threads = std::min(
      kMaxNormThreads,
      static_cast<int>(ceil_div(params.cols, kSubgroupSize) * kSubgroupSize));
  threads = std::max(threads, kSubgroupSize);
  float inv_cols = 1.0f / static_cast<float>(params.cols);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<RmsNormGeluKernel<Element, ApplyGelu>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(params.rows) * static_cast<std::size_t>(threads)),
            sycl::range<1>(static_cast<std::size_t>(threads))),
        [=](sycl::nd_item<1> item) {
          int64_t row = static_cast<int64_t>(item.get_group(0));
          int lane = static_cast<int>(item.get_local_id(0));
          int stride = static_cast<int>(item.get_local_range(0));
          Element const* row_in = params.x + row * params.cols;
          Element* row_out = params.out + row * params.cols;

          float partial = 0.0f;
          for (int col = lane; col < params.cols; col += stride) {
            float v = to_float(row_in[col]);
            partial += v * v;
          }
          float sum_sq = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
          float scale = sycl::rsqrt(sum_sq * inv_cols + params.eps);

          for (int col = lane; col < params.cols; col += stride) {
            float v = to_float(row_in[col]) * scale * to_float(params.weight[col]);
            if constexpr (ApplyGelu) {
              v = gelu_exact(v);
            }
            row_out[col] = from_float<Element>(v);
          }
        });
  });
}

template <typename Element>
sycl::event launch_rms_norm(sycl::queue& queue, NormGeluParams<Element> params, bool apply_gelu) {
  return apply_gelu ? launch_rms_norm<Element, true>(queue, params)
                    : launch_rms_norm<Element, false>(queue, params);
}

// ---------------------------------------------------------------------------
// Linear: oneMKL row-major GEMM, weight consumed as nn.Linear [out, in]
// ---------------------------------------------------------------------------

template <typename Element>
void launch_linear(
    sycl::queue& queue,
    Element const* activation,  // [rows, in_features] row-major
    Element const* weight,      // [out_features, in_features] row-major (nn.Linear)
    Element* out,               // [rows, out_features] row-major
    int64_t rows,
    int in_features,
    int out_features) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0) {
    return;
  }
  using MklElement = typename MklTraits<Element>::type;
  using MklScalar = typename MklTraits<Element>::scalar;
  oneapi::mkl::blas::row_major::gemm(
      queue,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans,
      rows,
      out_features,
      in_features,
      MklScalar(1),
      reinterpret_cast<MklElement const*>(activation),
      in_features,
      reinterpret_cast<MklElement const*>(weight),
      in_features,
      MklScalar(0),
      reinterpret_cast<MklElement*>(out),
      out_features);
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

struct CaseConfig {
  std::string name;
  VisionConfig vision;
  // layer_index < 0 runs the whole stack for `patches` patches; otherwise it
  // runs exactly one layer with `rows` rows.
  int layer_index = -1;
  int64_t patches = 1;
  int64_t rows = 0;
  bool allow_verify = true;
  // Perf gates are report-only (0.0). BMG numbers here are dominated by the
  // oneMKL GEMM, which this repo does not own, so a hard gate would be a gate
  // on a third-party library version. See README.
  double target_tops = 0.0;
  double target_gbps = 0.0;
};

// The shipped Inkling checkpoint vision_config, verbatim from the snapshot's
// config.json ("vision_encoder_type": "hmlp", patch_size 40,
// temporal_patch_size 2, n_channels 3, n_layers 4, decoder_dmodel 768,
// use_vision_norm true) -- not the InklingVisionConfig dataclass defaults, which
// are patch_size 16 / temporal_patch_size 1 / n_layers 1 and are covered
// separately by default_vision() below.
inline VisionConfig shipped_vision(int decoder_dmodel) {
  VisionConfig cfg;
  cfg.temporal_patch_size = 2;
  cfg.patch_size = 40;
  cfg.n_channels = 3;
  cfg.n_layers = 4;
  cfg.decoder_dmodel = decoder_dmodel;
  cfg.use_vision_norm = true;
  return cfg;
}

// InklingVisionConfig defaults (patch_size=16, temporal_patch_size=1,
// n_layers=1, use_vision_norm=False).
inline VisionConfig default_vision(int decoder_dmodel) {
  VisionConfig cfg;
  cfg.temporal_patch_size = 1;
  cfg.patch_size = 16;
  cfg.n_channels = 3;
  cfg.n_layers = 1;
  cfg.decoder_dmodel = decoder_dmodel;
  cfg.use_vision_norm = false;
  return cfg;
}

inline VisionConfig tiny_vision(int patch_size, int n_layers, int decoder_dmodel, bool final_norm) {
  VisionConfig cfg;
  cfg.temporal_patch_size = 1;
  cfg.patch_size = patch_size;
  cfg.n_channels = 3;
  cfg.n_layers = n_layers;
  cfg.decoder_dmodel = decoder_dmodel;
  cfg.use_vision_norm = final_norm;
  return cfg;
}

inline std::vector<CaseConfig> quick_suite() {
  std::vector<CaseConfig> cases;
  // Small synthetic configs so the CPU reference is instant, but still driven
  // through plan_out_scales rather than hard-coded widths.
  cases.push_back({"tiny_p4_l1_d64_full", tiny_vision(4, 1, 64, true), -1, 2, 0, true});
  cases.push_back({"tiny_p6_l2_d64_full", tiny_vision(6, 2, 64, false), -1, 1, 0, true});
  // Real shipped layers at row counts that exercise the odd/1-row edges.
  cases.push_back({"shipped_l0_rows1", shipped_vision(768), 0, 1, 1, true});
  cases.push_back({"shipped_l0_rows9", shipped_vision(768), 0, 1, 9, true});
  cases.push_back({"shipped_l3_rows1", shipped_vision(768), 3, 1, 1, true});
  cases.push_back({"default_p16_l1_rows9", default_vision(768), 0, 1, 9, true});
  return cases;
}

inline std::vector<CaseConfig> inkling_suite() {
  // The HMLP tower is replicated per rank (it is not tensor-parallel sharded in
  // sglang: HMLPPatchEncoder uses plain nn.Linear, not a Column/RowParallel
  // layer), so every rank runs these exact shapes for TP=1/2/4/8.
  //
  // decoder_dmodel comes from the multimodal config: the shipped checkpoint
  // pins it to 768 (== text hidden_size), while the production 6144 text
  // hidden_size gives the second variant. Both are covered.
  std::vector<CaseConfig> cases;
  for (int dmodel : {768, 6144}) {
    VisionConfig cfg = shipped_vision(dmodel);
    std::vector<HmlpLayer> layers = plan_hmlp_layers(cfg);
    std::string tag = "shipped_d" + std::to_string(dmodel) + "_";
    for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
      std::string li = "l" + std::to_string(i) + "_";
      // Decode-like single row, the odd 9-row band, and the natural row count
      // this layer sees for a single 40x40x2 patch.
      cases.push_back({tag + li + "rows1", cfg, i, 1, 1, true});
      cases.push_back({tag + li + "rows9", cfg, i, 1, 9, true});
      cases.push_back(
          {tag + li + "rows_1patch", cfg, i, 1, layers[static_cast<std::size_t>(i)].rows_per_patch, true});
    }
    cases.push_back({tag + "full_p1", cfg, -1, 1, 0, true});
    cases.push_back({tag + "full_p4", cfg, -1, 4, 0, true});
  }
  for (int dmodel : {768, 6144}) {
    VisionConfig cfg = default_vision(dmodel);
    std::string tag = "default_p16_d" + std::to_string(dmodel) + "_";
    cases.push_back({tag + "rows1", cfg, 0, 1, 1, true});
    cases.push_back({tag + "rows9", cfg, 0, 1, 9, true});
    cases.push_back({tag + "full_p16", cfg, -1, 16, 0, true});
  }
  return cases;
}

inline std::vector<CaseConfig> perf_suite() {
  // Rows spanning the prefill bands. Verification is off: the CPU reference for
  // e.g. 16384x5120x4800 is minutes of scalar work.
  std::vector<CaseConfig> cases;
  for (int dmodel : {768, 6144}) {
    VisionConfig cfg = shipped_vision(dmodel);
    std::vector<HmlpLayer> layers = plan_hmlp_layers(cfg);
    std::string tag = "perf_shipped_d" + std::to_string(dmodel) + "_";
    for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
      std::string li = "l" + std::to_string(i) + "_";
      cases.push_back({tag + li + "rows4096", cfg, i, 1, 4096, false});
      cases.push_back({tag + li + "rows16384", cfg, i, 1, 16384, false});
    }
    // Full stacks. patches=128 puts layer 0 at 16384 rows.
    cases.push_back({tag + "full_p32", cfg, -1, 32, 0, false});
    cases.push_back({tag + "full_p128", cfg, -1, 128, 0, false});
  }
  for (int dmodel : {768, 6144}) {
    VisionConfig cfg = default_vision(dmodel);
    std::string tag = "perf_default_p16_d" + std::to_string(dmodel) + "_";
    cases.push_back({tag + "rows4096", cfg, 0, 1, 4096, false});
    cases.push_back({tag + "rows16384", cfg, 0, 1, 16384, false});
  }
  return cases;
}

// Strict integer parse: std::stoi("16k") silently yields 16, which would run a
// different shape than the user asked for.
inline int64_t parse_strict_int(std::string const& value) {
  std::size_t consumed = 0;
  int64_t parsed = std::stoll(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("expected an integer, got '" + value + "'");
  }
  return parsed;
}

inline bool parse_single_shape(std::string const& text, CaseConfig& cfg) {
  cfg.name = "custom";
  cfg.vision = shipped_vision(768);
  cfg.layer_index = -1;
  cfg.patches = 1;
  cfg.rows = 0;
  cfg.allow_verify = true;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto eq = item.find('=');
    if (eq == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, eq);
    std::string value = item.substr(eq + 1);
    try {
      if (key == "name") {
        cfg.name = value;
      } else if (key == "temporal_patch_size" || key == "t") {
        cfg.vision.temporal_patch_size = static_cast<int>(parse_strict_int(value));
      } else if (key == "patch_size" || key == "p") {
        cfg.vision.patch_size = static_cast<int>(parse_strict_int(value));
      } else if (key == "n_channels" || key == "c") {
        cfg.vision.n_channels = static_cast<int>(parse_strict_int(value));
      } else if (key == "n_layers" || key == "n") {
        cfg.vision.n_layers = static_cast<int>(parse_strict_int(value));
      } else if (key == "decoder_dmodel" || key == "d") {
        cfg.vision.decoder_dmodel = static_cast<int>(parse_strict_int(value));
      } else if (key == "use_vision_norm" || key == "final_norm") {
        cfg.vision.use_vision_norm = parse_strict_int(value) != 0;
      } else if (key == "layer") {
        cfg.layer_index = static_cast<int>(parse_strict_int(value));
      } else if (key == "patches") {
        cfg.patches = parse_strict_int(value);
      } else if (key == "rows") {
        cfg.rows = parse_strict_int(value);
      } else if (key == "verify") {
        cfg.allow_verify = parse_strict_int(value) != 0;
      } else {
        return false;
      }
    } catch (std::exception const&) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Execution plan for one case
// ---------------------------------------------------------------------------

struct StageShape {
  int layer_index = 0;
  int in_features = 0;
  int out_features = 0;
  int64_t rows = 0;
  bool has_norm = false;
  bool has_gelu = false;
};

// Expands a case into the concrete (rows, in, out) stages actually launched.
// The stack case chains stages: stage i+1's activation is stage i's output
// reinterpreted as [rows / shuffle_mult, in_features], which is the shape the
// fold produces (example 22 owns the permutation).
inline std::vector<StageShape> plan_stages(CaseConfig const& cfg, std::vector<HmlpLayer> const& layers) {
  std::vector<StageShape> stages;
  if (cfg.layer_index >= 0) {
    if (cfg.layer_index >= static_cast<int>(layers.size())) {
      throw std::invalid_argument("layer index out of range for this vision config");
    }
    HmlpLayer const& layer = layers[static_cast<std::size_t>(cfg.layer_index)];
    int64_t rows = cfg.rows > 0 ? cfg.rows : layer.rows_per_patch * cfg.patches;
    stages.push_back({cfg.layer_index, layer.in_features, layer.out_features, rows,
                      layer.has_norm, layer.has_gelu});
    return stages;
  }

  // A chained case is sized in patches, because every stage's row count is tied
  // to the same patch count by the folds. `rows` is still accepted and is taken
  // as the first stage's row count, which must therefore be a whole number of
  // patches; silently ignoring it would hand back the wrong shape.
  int64_t patches = cfg.patches;
  if (cfg.rows > 0) {
    int64_t rows_per_patch = layers.front().rows_per_patch;
    if (cfg.rows % rows_per_patch != 0) {
      throw std::invalid_argument(
          "full-stack rows must be a multiple of the first layer's rows_per_patch");
    }
    patches = cfg.rows / rows_per_patch;
  }
  if (patches <= 0) {
    throw std::invalid_argument("patches must be positive for a full-stack case");
  }
  for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
    HmlpLayer const& layer = layers[static_cast<std::size_t>(i)];
    int64_t rows = layer.rows_per_patch * patches;
    if (rows <= 0) {
      throw std::invalid_argument("full-stack case produced a zero-row layer");
    }
    stages.push_back({i, layer.in_features, layer.out_features, rows, layer.has_norm, layer.has_gelu});
  }
  return stages;
}

struct VerifyResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  int max_ulps = 0;
  std::size_t index = 0;
  std::string where;
};

// Tolerance rationale. Both sides accumulate in fp32 and round the result to
// the activation dtype, but oneMKL's accumulation order differs from the scalar
// host loop, so a value sitting on a rounding boundary can land one dtype ulp
// apart. One bf16 ulp is at most 2^-8 relative (fp16: 2^-11), hence kRelTol =
// 2 ulps of relative spacing. kAbsTolScale covers outputs near zero, where a
// cancelling dot product or GELU(x<0) makes the relative measure meaningless
// (a few 1e-5 either side of zero is thousands of "ulps" but a ~1e-5 error);
// it is taken relative to the largest magnitude in the same block, so it scales
// with the data instead of being an absolute magic number.
template <typename Element>
double rel_tolerance() {
  return std::is_same_v<Element, cutlass::bfloat16_t> ? 2.0 / 256.0 : 2.0 / 2048.0;
}

constexpr double kAbsTolScale = 1.0e-3;

template <typename Element>
void compare_block(
    std::vector<Element> const& got,
    std::vector<Element> const& expected,
    std::string const& where,
    VerifyResult& result) {
  if (got.size() != expected.size()) {
    throw std::invalid_argument("compare_block size mismatch");
  }
  double block_scale = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    block_scale = std::max(block_scale, std::abs(static_cast<double>(to_float(expected[i]))));
  }
  double rel_tol = rel_tolerance<Element>();
  double abs_tol = kAbsTolScale * std::max(block_scale, 1.0e-6);
  double rel_floor = std::max(kAbsTolScale * block_scale, 1.0e-12);

  for (std::size_t i = 0; i < got.size(); ++i) {
    double g = static_cast<double>(to_float(got[i]));
    double e = static_cast<double>(to_float(expected[i]));
    double abs_err = std::abs(g - e);
    double rel_err = abs_err / std::max(rel_floor, std::abs(e));
    int ulps = std::abs(ordered_raw16(host_bits(got[i])) - ordered_raw16(host_bits(expected[i])));
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);
    result.max_ulps = std::max(result.max_ulps, ulps);
    // A NaN/Inf in the device output must fail: every comparison against NaN is
    // false, so without the explicit finite check it would report as a pass.
    bool bad = !std::isfinite(g) || abs_err > abs_tol + rel_tol * std::abs(e);
    if (bad && result.passed) {
      result.passed = false;
      result.index = i;
      result.where = where;
    }
  }
}

// CPU reference for one stage, mirroring torch semantics exactly:
// Linear (fp32 accumulate, result rounded to the activation dtype) then, when
// the layer has them, RMSNorm (fp32 reduction over the rounded Linear output)
// and the exact erf GELU.
template <typename Element>
void reference_stage(
    std::vector<Element> const& activation,
    std::vector<Element> const& weight,
    std::vector<Element> const& norm_weight,
    StageShape const& stage,
    std::vector<Element>& linear_out,
    std::vector<Element>& norm_out) {
  int64_t rows = stage.rows;
  int in_features = stage.in_features;
  int out_features = stage.out_features;
  linear_out.assign(static_cast<std::size_t>(rows) * out_features, Element{});

  for (int64_t row = 0; row < rows; ++row) {
    Element const* a = activation.data() + row * in_features;
    for (int n = 0; n < out_features; ++n) {
      Element const* w = weight.data() + static_cast<int64_t>(n) * in_features;
      float acc = 0.0f;
      for (int k = 0; k < in_features; ++k) {
        acc += to_float(a[k]) * to_float(w[k]);
      }
      linear_out[static_cast<std::size_t>(row) * out_features + n] = from_float<Element>(acc);
    }
  }

  if (!stage.has_norm) {
    norm_out.clear();
    return;
  }

  norm_out.assign(static_cast<std::size_t>(rows) * out_features, Element{});
  for (int64_t row = 0; row < rows; ++row) {
    Element const* x = linear_out.data() + row * out_features;
    float sum_sq = 0.0f;
    for (int n = 0; n < out_features; ++n) {
      float v = to_float(x[n]);
      sum_sq += v * v;
    }
    float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(out_features) + kRmsNormEps);
    for (int n = 0; n < out_features; ++n) {
      float v = to_float(x[n]) * scale * to_float(norm_weight[static_cast<std::size_t>(n)]);
      if (stage.has_gelu) {
        v = gelu_exact(v);
      }
      norm_out[static_cast<std::size_t>(row) * out_features + n] = from_float<Element>(v);
    }
  }
}

template <typename Element>
std::vector<Element> make_random(std::size_t count, uint32_t seed, float amplitude) {
  // Random (not constant) data: Xe memory compression inflates bandwidth
  // numbers on constant or zero buffers.
  std::vector<Element> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-amplitude, amplitude);
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = from_float<Element>(dist(gen));
  }
  return data;
}

struct CaseCost {
  double gemm_flops = 0.0;
  double gemm_bytes = 0.0;
  double norm_bytes = 0.0;
};

inline CaseCost case_cost(
    std::vector<StageShape> const& stages, std::size_t element_bytes, int final_norm_width) {
  CaseCost cost;
  for (StageShape const& stage : stages) {
    double rows = static_cast<double>(stage.rows);
    double in_f = static_cast<double>(stage.in_features);
    double out_f = static_cast<double>(stage.out_features);
    cost.gemm_flops += 2.0 * rows * in_f * out_f;
    cost.gemm_bytes +=
        (rows * in_f + in_f * out_f + rows * out_f) * static_cast<double>(element_bytes);
    if (stage.has_norm) {
      // Two reads (sum-of-squares pass, scaled pass) and one write.
      cost.norm_bytes += 3.0 * rows * out_f * static_cast<double>(element_bytes);
    }
  }
  if (final_norm_width > 0 && !stages.empty()) {
    // The use_vision_norm pass after the last layer.
    cost.norm_bytes += 3.0 * static_cast<double>(stages.back().rows) *
        static_cast<double>(final_norm_width) * static_cast<double>(element_bytes);
  }
  return cost;
}

// ---------------------------------------------------------------------------
// Case runner
// ---------------------------------------------------------------------------

template <typename Element>
struct StageBuffers {
  DeviceBuffer<Element> activation;   // stage input (owned only by stage 0 of a chain)
  DeviceBuffer<Element> weight;       // [out_features, in_features]
  DeviceBuffer<Element> norm_weight;  // [out_features]
  DeviceBuffer<Element> linear_out;   // [rows, out_features]
  DeviceBuffer<Element> norm_out;     // [rows, out_features], only when has_norm
  std::vector<Element> host_activation;
  std::vector<Element> host_weight;
  std::vector<Element> host_norm_weight;
};

template <typename Element>
bool run_case(
    sycl::queue& q,
    CaseConfig const& cfg,
    int iterations,
    int warmup,
    bool verify,
    bool benchmark,
    double target_tops,
    double target_gbps,
    double threshold_scale) {
  std::vector<HmlpLayer> layers = plan_hmlp_layers(cfg.vision);
  std::vector<StageShape> stages = plan_stages(cfg, layers);
  bool chained = cfg.layer_index < 0;
  // use_vision_norm's RMSNorm(decoder_dmodel) belongs to the last layer, so it
  // runs whenever the last layer is in the plan -- including a single-layer case
  // that selects it (`--shape=...,layer=<n_layers-1>`), not just the full stack.
  bool final_norm = cfg.vision.use_vision_norm &&
      layers[static_cast<std::size_t>(stages.back().layer_index)].final_norm;
  int final_norm_width = final_norm ? stages.back().out_features : 0;

  std::vector<StageBuffers<Element>> buffers(stages.size());
  uint32_t seed = 1234;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    StageShape const& stage = stages[i];
    StageBuffers<Element>& buf = buffers[i];
    std::size_t weight_count =
        static_cast<std::size_t>(stage.in_features) * static_cast<std::size_t>(stage.out_features);
    std::size_t out_count =
        static_cast<std::size_t>(stage.rows) * static_cast<std::size_t>(stage.out_features);

    // 1/sqrt(fan_in)-ish weights keep the pre-norm activations in a sane range.
    float weight_amp = 1.0f / std::sqrt(static_cast<float>(stage.in_features));
    buf.host_weight = make_random<Element>(weight_count, seed + 11, weight_amp * 3.0f);
    buf.weight = DeviceBuffer<Element>(q, weight_count);
    buf.weight.copy_from(buf.host_weight);

    if (stage.has_norm) {
      // RMSNorm weights are initialised to ones but are learned; use values
      // around 1 so a dropped weight multiply is still visible.
      buf.host_norm_weight.resize(static_cast<std::size_t>(stage.out_features));
      std::mt19937 gen(seed + 23);
      std::uniform_real_distribution<float> dist(0.5f, 1.5f);
      for (int n = 0; n < stage.out_features; ++n) {
        buf.host_norm_weight[static_cast<std::size_t>(n)] = from_float<Element>(dist(gen));
      }
      buf.norm_weight = DeviceBuffer<Element>(q, buf.host_norm_weight.size());
      buf.norm_weight.copy_from(buf.host_norm_weight);
      buf.norm_out = DeviceBuffer<Element>(q, out_count);
    }

    buf.linear_out = DeviceBuffer<Element>(q, out_count);

    bool needs_own_activation = !chained || i == 0;
    if (needs_own_activation) {
      std::size_t act_count =
          static_cast<std::size_t>(stage.rows) * static_cast<std::size_t>(stage.in_features);
      buf.host_activation = make_random<Element>(act_count, seed + 7, 1.0f);
      buf.activation = DeviceBuffer<Element>(q, act_count);
      buf.activation.copy_from(buf.host_activation);
    }
    seed += 101;
  }

  // The final use_vision_norm RMSNorm, applied to the last stage's Linear output.
  DeviceBuffer<Element> final_norm_weight;
  DeviceBuffer<Element> final_norm_out;
  std::vector<Element> host_final_norm_weight;
  if (final_norm) {
    host_final_norm_weight.resize(static_cast<std::size_t>(final_norm_width));
    std::mt19937 gen(seed + 31);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (int n = 0; n < final_norm_width; ++n) {
      host_final_norm_weight[static_cast<std::size_t>(n)] = from_float<Element>(dist(gen));
    }
    final_norm_weight = DeviceBuffer<Element>(q, host_final_norm_weight.size());
    final_norm_weight.copy_from(host_final_norm_weight);
    final_norm_out = DeviceBuffer<Element>(
        q,
        static_cast<std::size_t>(stages.back().rows) *
            static_cast<std::size_t>(stages.back().out_features));
  }

  auto stage_input = [&](std::size_t i) -> Element const* {
    if (!chained || i == 0) {
      return buffers[i].activation.get();
    }
    // Reinterpret the previous stage's output rows: the fold regroups
    // shuffle_mult consecutive cells into one row of the next Linear.
    StageBuffers<Element> const& prev = buffers[i - 1];
    return stages[i - 1].has_norm ? prev.norm_out.get() : prev.linear_out.get();
  };

  auto launch_all = [&]() {
    for (std::size_t i = 0; i < stages.size(); ++i) {
      StageShape const& stage = stages[i];
      StageBuffers<Element>& buf = buffers[i];
      launch_linear<Element>(
          q,
          stage_input(i),
          buf.weight.get(),
          buf.linear_out.get(),
          stage.rows,
          stage.in_features,
          stage.out_features);
      if (stage.has_norm) {
        NormGeluParams<Element> params;
        params.x = buf.linear_out.get();
        params.weight = buf.norm_weight.get();
        params.out = buf.norm_out.get();
        params.rows = stage.rows;
        params.cols = stage.out_features;
        launch_rms_norm<Element>(q, params, stage.has_gelu);
      }
    }
    if (final_norm) {
      NormGeluParams<Element> params;
      params.x = buffers.back().linear_out.get();
      params.weight = final_norm_weight.get();
      params.out = final_norm_out.get();
      params.rows = stages.back().rows;
      params.cols = final_norm_width;
      launch_rms_norm<Element>(q, params, /*apply_gelu=*/false);
    }
  };

  launch_all();
  q.wait_and_throw();

  bool passed = true;
  VerifyResult vr;
  bool verified = verify && cfg.allow_verify;
  if (verified) {
    // Each stage is checked against the reference computed from the stage's own
    // device-side input, so no error accumulates across the chain and a tight
    // ulp bound stays meaningful.
    for (std::size_t i = 0; i < stages.size() && vr.passed; ++i) {
      StageShape const& stage = stages[i];
      StageBuffers<Element>& buf = buffers[i];
      std::size_t act_count =
          static_cast<std::size_t>(stage.rows) * static_cast<std::size_t>(stage.in_features);
      std::vector<Element> host_input(act_count);
      q.memcpy(host_input.data(), stage_input(i), sizeof(Element) * act_count).wait();

      std::vector<Element> ref_linear;
      std::vector<Element> ref_norm;
      reference_stage<Element>(
          host_input, buf.host_weight, buf.host_norm_weight, stage, ref_linear, ref_norm);

      std::vector<Element> got(ref_linear.size());
      buf.linear_out.copy_to(got);
      compare_block<Element>(
          got, ref_linear, "stage" + std::to_string(stage.layer_index) + ".linear", vr);

      if (stage.has_norm && vr.passed) {
        std::vector<Element> got_norm(ref_norm.size());
        buf.norm_out.copy_to(got_norm);
        compare_block<Element>(
            got_norm, ref_norm, "stage" + std::to_string(stage.layer_index) + ".norm_gelu", vr);
      }
    }

    if (final_norm && vr.passed) {
      StageShape const& stage = stages.back();
      std::size_t out_count =
          static_cast<std::size_t>(stage.rows) * static_cast<std::size_t>(final_norm_width);
      std::vector<Element> host_linear(out_count);
      buffers.back().linear_out.copy_to(host_linear);
      // Reference the final norm directly from the device Linear output.
      std::vector<Element> ref_norm(out_count);
      for (int64_t row = 0; row < stage.rows; ++row) {
        Element const* x = host_linear.data() + row * final_norm_width;
        float sum_sq = 0.0f;
        for (int n = 0; n < final_norm_width; ++n) {
          float v = to_float(x[n]);
          sum_sq += v * v;
        }
        float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(final_norm_width) + kRmsNormEps);
        for (int n = 0; n < final_norm_width; ++n) {
          ref_norm[static_cast<std::size_t>(row) * final_norm_width + n] = from_float<Element>(
              to_float(x[n]) * scale * to_float(host_final_norm_weight[static_cast<std::size_t>(n)]));
        }
      }
      std::vector<Element> got(out_count);
      final_norm_out.copy_to(got);
      compare_block<Element>(got, ref_norm, "final_norm", vr);
    }
    passed = vr.passed;
  }

  double avg_s = 0.0;
  CaseCost cost = case_cost(stages, sizeof(Element), final_norm_width);
  if (benchmark) {
    for (int i = 0; i < std::max(warmup, 0); ++i) {
      launch_all();
    }
    q.wait_and_throw();

    int timing_iterations = std::max(1, iterations);
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
      launch_all();
    }
    q.wait_and_throw();
    avg_s = timer.seconds() / static_cast<double>(timing_iterations);
  }

  double tops = avg_s > 0.0 ? cost.gemm_flops / avg_s / 1.0e12 : 0.0;
  double gbps = avg_s > 0.0 ? ((cost.gemm_bytes + cost.norm_bytes) / kBytesPerGB) / avg_s : 0.0;

  std::cout << std::left << std::setw(34) << cfg.name << std::right
            << " dtype=" << element_dtype_text<Element>()
            << " stages=" << stages.size();
  if (stages.size() == 1) {
    std::cout << " rows=" << std::setw(6) << stages[0].rows
              << " " << std::setw(5) << stages[0].in_features << "->" << std::setw(5)
              << stages[0].out_features
              << " norm=" << (stages[0].has_norm ? 1 : 0)
              << " gelu=" << (stages[0].has_gelu ? 1 : 0)
              << " fnorm=" << (final_norm ? 1 : 0);
  } else {
    std::cout << " patches=" << (stages[0].rows / layers.front().rows_per_patch) << " [";
    for (std::size_t i = 0; i < stages.size(); ++i) {
      std::cout << (i ? " " : "") << stages[i].rows << "x" << stages[i].in_features << "->"
                << stages[i].out_features << (stages[i].has_gelu ? "g" : "");
    }
    std::cout << "] fnorm=" << (final_norm ? 1 : 0);
  }

  if (benchmark) {
    std::cout << std::fixed << std::setprecision(4)
              << "  " << (avg_s * 1000.0) << " ms"
              << std::setprecision(3)
              << "  " << tops << " TOPS"
              << "  " << gbps << " GB/s" << std::defaultfloat;
  }

  if (verified) {
    std::cout << "  " << (passed ? "passed" : "failed")
              << " max_abs=" << vr.max_abs
              << " max_rel=" << vr.max_rel
              << " max_ulps=" << vr.max_ulps;
    if (!passed) {
      std::cout << " at=" << vr.where << " index=" << vr.index;
    }
  } else if (verify) {
    std::cout << "  verify=SKIP (perf case; use quick/inkling for the CPU reference)";
  } else {
    std::cout << "  verification skipped";
  }
  std::cout << "\n";

  double tops_gate = (target_tops > 0.0 ? target_tops : cfg.target_tops) * threshold_scale;
  double gbps_gate = (target_gbps > 0.0 ? target_gbps : cfg.target_gbps) * threshold_scale;
  if (benchmark && tops_gate > 0.0 && tops < tops_gate) {
    std::cerr << "    perf=FAIL " << cfg.name << " TOPS " << tops << " < " << tops_gate << "\n";
    passed = false;
  }
  if (benchmark && gbps_gate > 0.0 && gbps < gbps_gate) {
    std::cerr << "    perf=FAIL " << cfg.name << " GB/s " << gbps << " < " << gbps_gate << "\n";
    passed = false;
  }
  return passed;
}

struct Options {
  bool help = false;
  bool valid = true;
  bool verify = true;
  bool benchmark = true;
  int iterations = 20;
  int warmup = 5;
  std::string suite = "quick";
  std::string shape;
  std::string dtype_name = "bf16";
  DType dtype = DType::kBf16;
  double target_tops = 0.0;
  double target_gbps = 0.0;
  double perf_threshold_scale = 1.0;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    int verify_int = 1;
    cmd.get_cmd_line_argument("verify", verify_int, 1);
    verify = verify_int != 0;
    int benchmark_int = 1;
    cmd.get_cmd_line_argument("benchmark", benchmark_int, 1);
    benchmark = benchmark_int != 0;
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("warmup", warmup, 5);
    cmd.get_cmd_line_argument("suite", suite, std::string("quick"));
    cmd.get_cmd_line_argument("shape", shape, std::string(""));
    cmd.get_cmd_line_argument("dtype", dtype_name, std::string("bf16"));
    cmd.get_cmd_line_argument("target-tops", target_tops, 0.0);
    cmd.get_cmd_line_argument("target-gbps", target_gbps, 0.0);
    cmd.get_cmd_line_argument("perf-threshold-scale", perf_threshold_scale, 1.0);
    if (!parse_dtype(dtype_name, dtype)) {
      valid = false;
    }
    if (iterations < 0 || warmup < 0 || perf_threshold_scale < 0.0) {
      valid = false;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Inkling BMG HMLP Linear + RMSNorm + GELU Example\n\n"
        << "Options:\n"
        << "  --help                          Print this message\n"
        << "  --suite=<quick|inkling|perf>    Built-in shape suite (default: quick)\n"
        << "  --shape=<k=v,...>               Run one custom case instead of a suite\n"
        << "                                  Keys: name,t,p,c,n,d,use_vision_norm,layer,patches,rows,verify\n"
        << "                                  (t=temporal_patch_size, p=patch_size, c=n_channels,\n"
        << "                                   n=n_layers, d=decoder_dmodel, layer=-1 for the full stack)\n"
        << "  --dtype=<bf16|fp16>             Activation/weight dtype (default: bf16, the shipped dtype)\n"
        << "  --iterations=<int>              Timed iterations (default 20)\n"
        << "  --warmup=<int>                  Warmup iterations (default 5; BMG needs ~2 s to clock up)\n"
        << "  --verify=<0|1>                  Run the CPU reference where the case permits\n"
        << "  --benchmark=<0|1>               Run timing\n"
        << "  --target-tops=<float>           Fail below this Linear TOPS (0 = report only)\n"
        << "  --target-gbps=<float>           Fail below this effective GB/s (0 = report only)\n"
        << "  --perf-threshold-scale=<float>  Scale all perf gates (default 1.0)\n\n"
        << "Examples:\n"
        << "  ./27_bmg_hmlp_linear_gelu --suite=quick --verify=1\n"
        << "  ./27_bmg_hmlp_linear_gelu --suite=inkling --verify=1\n"
        << "  ./27_bmg_hmlp_linear_gelu --suite=perf --verify=0 --iterations=50\n"
        << "  ./27_bmg_hmlp_linear_gelu --shape=t=2,p=40,n=4,d=6144,layer=2,rows=16384,verify=0\n";
    return out;
  }
};

}  // namespace cutlass::examples::bmg_hmlp_linear_gelu

int main(int argc, char const** argv) {
  using namespace cutlass::examples::bmg_hmlp_linear_gelu;

  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }
  if (!options.valid) {
    std::cerr << "Invalid options (dtype=" << options.dtype_name << ")\n";
    options.print_usage(std::cerr);
    return 1;
  }

  std::vector<CaseConfig> cases;
  try {
    if (!options.shape.empty()) {
      CaseConfig cfg;
      if (!parse_single_shape(options.shape, cfg)) {
        std::cerr << "Invalid --shape argument: " << options.shape << "\n";
        options.print_usage(std::cerr);
        return 1;
      }
      cases.push_back(cfg);
    } else if (options.suite == "quick") {
      cases = quick_suite();
    } else if (options.suite == "inkling") {
      cases = inkling_suite();
    } else if (options.suite == "perf") {
      cases = perf_suite();
    } else {
      std::cerr << "Unknown suite: " << options.suite << "\n";
      options.print_usage(std::cerr);
      return 1;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to build the case list: " << e.what() << "\n";
    return 1;
  }

  try {
    sycl::queue q = compat::get_default_queue();
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "27_bmg_hmlp_linear_gelu: per-HMLP-layer Linear(no bias) -> RMSNorm -> GELU\n";
    std::cout << "Suite: " << (options.shape.empty() ? options.suite : "custom")
              << ", cases=" << cases.size()
              << ", dtype=" << dtype_text(options.dtype)
              << ", iterations=" << options.iterations
              << ", warmup=" << options.warmup
              << ", verify=" << bool_text(options.verify)
              << ", benchmark=" << bool_text(options.benchmark) << "\n";

    bool all_passed = true;
    for (CaseConfig const& cfg : cases) {
      if (options.dtype == DType::kBf16) {
        all_passed &= run_case<cutlass::bfloat16_t>(
            q, cfg, options.iterations, options.warmup, options.verify, options.benchmark,
            options.target_tops, options.target_gbps, options.perf_threshold_scale);
      } else {
        all_passed &= run_case<cutlass::half_t>(
            q, cfg, options.iterations, options.warmup, options.verify, options.benchmark,
            options.target_tops, options.target_gbps, options.perf_threshold_scale);
      }
    }
    return all_passed ? 0 : 2;
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
  } catch (std::exception const& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  return 1;
}
