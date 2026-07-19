/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Inkling decode-only fused short-convolution + cache update example for CUTLASS SYCL.

    Semantics match modeltune/inkling/01_sconv/01_03_fused_decode_update:

      y[t,d] = activation(sum_{iw<W-1} cache[slot,iw,d] * cache_mask[t] * weight[d,iw]
                           + x[t,d] * weight[d,W-1])
               + x[t,d] if residual is enabled

      cache[slot] is shifted left and x[t] is appended for non-pad slots. When prefix-cache
      tracking is enabled, the same post-update window is copied to cache[track_indices[t]]
      wherever track_mask[t] is true.

    Roofline: the production W=4 bf16/fp16 decode path performs roughly 8 useful FLOPs per
    channel and streams about 20 bytes without tracking from x/cache/y/update traffic
    (weight is tiny and cache-resident), so arithmetic intensity is about 0.4 FLOP/B.
    This is memory-bound; performance reporting emphasizes conservative effective GB/s.
*/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cutlass::examples::sconv {

constexpr int kPadSlot = -1;
constexpr int kThreads = 256;
constexpr int kScalarVec = 1;
constexpr int kW4FastVec = 4;
constexpr int kW4WideVec = 8;
constexpr double kMinSustainedTargetBytes = 512.0 * 1024.0 * 1024.0;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

template <typename Element_>
struct FusedDecodeUpdateParams {
  using Element = Element_;

  Element const* __restrict__ x;
  Element* __restrict__ cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ cache_mask;
  Element const* __restrict__ weight;
  Element* __restrict__ y;
  uint8_t const* __restrict__ track_mask;
  int64_t const* __restrict__ track_indices;
  int T;
  int D;
  int W;
  int cache_stride_slot;
  int cache_stride_w;
  int pad_slot_id;
};

template <typename Element, int StaticW, bool UseSilu, bool UseResidual, bool DoTrack, int Vec>
class FusedDecodeUpdateSconvKernel;

template <typename Element>
CUTLASS_HOST_DEVICE
float to_float(Element x) {
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

std::string dtype_text(DType dtype) {
  switch (dtype) {
    case DType::kAll:
      return "all";
    case DType::kBf16:
      return "bf16";
    case DType::kFp16:
      return "fp16";
  }
  return "unknown";
}

bool parse_dtype(std::string const& text, DType& dtype) {
  if (text == "all") {
    dtype = DType::kAll;
    return true;
  }
  if (text == "bf16") {
    dtype = DType::kBf16;
    return true;
  }
  if (text == "fp16") {
    dtype = DType::kFp16;
    return true;
  }
  return false;
}

template <typename Element>
char const* element_dtype_text() {
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    return "bf16";
  } else {
    return "fp16";
  }
}

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

std::string bool_text(bool value) {
  return value ? "true" : "false";
}

template <typename Element>
CUTLASS_DEVICE
Element element_from_raw(uint64_t raw, int lane) {
  return Element::bitcast(static_cast<uint16_t>(raw >> (16 * lane)));
}

template <typename Element>
CUTLASS_DEVICE
uint64_t load_pack4(Element const* ptr) {
  return *reinterpret_cast<uint64_t const*>(ptr);
}

template <typename Element>
CUTLASS_DEVICE
void store_pack4(Element* ptr, uint64_t raw) {
  *reinterpret_cast<uint64_t*>(ptr) = raw;
}

template <typename Element, int Vec>
CUTLASS_DEVICE
uint64_t pack4_from_floats(float const (&values)[Vec], int base) {
  uint64_t raw = 0;
#pragma unroll
  for (int v = 0; v < kW4FastVec; ++v) {
    raw |= static_cast<uint64_t>(Element(values[base + v]).raw()) << (16 * v);
  }
  return raw;
}

template <typename Element, int StaticW, bool UseSilu, bool UseResidual, bool DoTrack, int Vec>
CUTLASS_DEVICE
void run_fused_decode_update_kernel(
    FusedDecodeUpdateParams<Element> const& params,
    sycl::nd_item<1> item) {
  static_assert(Vec == kScalarVec || Vec == kW4FastVec || Vec == kW4WideVec, "unsupported fused decode vector width");
  static_assert(Vec == kScalarVec || StaticW == 4, "Vec=4 path is specialized for W=4");
  constexpr bool kPackedChannels = Vec != kScalarVec;
  constexpr int kPackCount = kPackedChannels ? (Vec / kW4FastVec) : 1;

  int channel_blocks = (params.D + Vec - 1) / Vec;
  int linear = static_cast<int>(item.get_global_linear_id());
  int total = params.T * channel_blocks;
  if (linear >= total) {
    return;
  }

  int channel_block = linear % channel_blocks;
  int t = linear / channel_blocks;
  int d0 = channel_block * Vec;
  int W = StaticW > 0 ? StaticW : params.W;
  int width_minus_one = W - 1;

  int ci = params.cache_indices[t];
  bool valid = ci != params.pad_slot_id;
  int slot = valid ? ci : 0;
  bool mask = params.cache_mask[t] != 0;
  int cache_base = slot * params.cache_stride_slot + d0;

  float acc[Vec];
  Element x_values[Vec];
  bool channel_valid[Vec];
  uint64_t x_pack[kPackCount];

  if constexpr (kPackedChannels) {
#pragma unroll
    for (int p = 0; p < kPackCount; ++p) {
      x_pack[p] = load_pack4(params.x + static_cast<std::size_t>(t) * params.D + d0 + p * kW4FastVec);
    }
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      channel_valid[v] = true;
      x_values[v] = element_from_raw<Element>(x_pack[v / kW4FastVec], v % kW4FastVec);
      acc[v] = 0.0f;
    }
  } else {
    channel_valid[0] = d0 < params.D;
    x_values[0] = channel_valid[0] ? params.x[static_cast<std::size_t>(t) * params.D + d0] : Element(0.0f);
    acc[0] = 0.0f;
  }

  uint64_t weight_pack[Vec];
  if constexpr (kPackedChannels) {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      weight_pack[v] = load_pack4(params.weight + static_cast<std::size_t>(d0 + v) * StaticW);
    }
  }

  for (int iw = 0; iw < width_minus_one; ++iw) {
    uint64_t history_pack[kPackCount];
    if constexpr (kPackedChannels) {
#pragma unroll
      for (int p = 0; p < kPackCount; ++p) {
        history_pack[p] = mask ? load_pack4(params.cache + cache_base + iw * params.cache_stride_w + p * kW4FastVec) : 0;
      }
    }
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      if (channel_valid[v]) {
        Element history = kPackedChannels
            ? element_from_raw<Element>(history_pack[v / kW4FastVec], v % kW4FastVec)
            : params.cache[cache_base + iw * params.cache_stride_w + v];
        Element w = kPackedChannels
            ? element_from_raw<Element>(weight_pack[v], iw)
            : params.weight[static_cast<std::size_t>(d0 + v) * W + iw];
        acc[v] += (mask ? to_float(history) : 0.0f) * to_float(w);
      }
    }
  }

#pragma unroll
  for (int v = 0; v < Vec; ++v) {
    if (channel_valid[v]) {
      Element w = kPackedChannels
          ? element_from_raw<Element>(weight_pack[v], width_minus_one)
          : params.weight[static_cast<std::size_t>(d0 + v) * W + width_minus_one];
      acc[v] += to_float(x_values[v]) * to_float(w);

      if constexpr (UseSilu) {
        acc[v] = acc[v] / (1.0f + sycl::exp(-acc[v]));
      }
      if constexpr (UseResidual) {
        acc[v] += to_float(x_values[v]);
      }
    }
  }

  if constexpr (kPackedChannels) {
#pragma unroll
    for (int p = 0; p < kPackCount; ++p) {
      store_pack4(
          params.y + static_cast<std::size_t>(t) * params.D + d0 + p * kW4FastVec,
          pack4_from_floats<Element, Vec>(acc, p * kW4FastVec));
    }
  } else {
    if (channel_valid[0]) {
      params.y[static_cast<std::size_t>(t) * params.D + d0] = Element(acc[0]);
    }
  }

  bool track = false;
  int track_base = 0;
  if constexpr (DoTrack) {
    track = params.track_mask[t] != 0;
    if (track) {
      track_base = static_cast<int>(params.track_indices[t]) * params.cache_stride_slot + d0;
    }
  }

  if (!valid) {
    return;
  }

  for (int iw = 0; iw < width_minus_one; ++iw) {
    if constexpr (kPackedChannels) {
#pragma unroll
      for (int p = 0; p < kPackCount; ++p) {
        uint64_t next_pack = (iw < width_minus_one - 1)
            ? (mask ? load_pack4(params.cache + cache_base + (iw + 1) * params.cache_stride_w + p * kW4FastVec) : 0)
            : x_pack[p];
        store_pack4(params.cache + cache_base + iw * params.cache_stride_w + p * kW4FastVec, next_pack);
        if constexpr (DoTrack) {
          if (track) {
            store_pack4(params.cache + track_base + iw * params.cache_stride_w + p * kW4FastVec, next_pack);
          }
        }
      }
    } else {
      Element next = (iw < width_minus_one - 1)
          ? (mask ? params.cache[cache_base + (iw + 1) * params.cache_stride_w] : Element(0.0f))
          : x_values[0];
      params.cache[cache_base + iw * params.cache_stride_w] = next;
      if constexpr (DoTrack) {
        if (track) {
          params.cache[track_base + iw * params.cache_stride_w] = next;
        }
      }
    }
  }
}

template <typename Element, int StaticW, bool UseSilu, bool UseResidual, bool DoTrack, int Vec>
sycl::event launch_fused_decode_update_static(
    sycl::queue& q,
    FusedDecodeUpdateParams<Element> const& params) {
  if (params.T == 0 || params.D == 0) {
    return sycl::event{};
  }

  int channel_blocks = (params.D + Vec - 1) / Vec;
  int total = params.T * channel_blocks;
  int global = ((total + kThreads - 1) / kThreads) * kThreads;
  return q.parallel_for<FusedDecodeUpdateSconvKernel<Element, StaticW, UseSilu, UseResidual, DoTrack, Vec>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)),
      [=](sycl::nd_item<1> item) {
        run_fused_decode_update_kernel<Element, StaticW, UseSilu, UseResidual, DoTrack, Vec>(params, item);
      });
}

template <typename Element, int StaticW, bool UseSilu, bool UseResidual, int Vec>
sycl::event launch_track_selected(
    sycl::queue& q,
    FusedDecodeUpdateParams<Element> const& params,
    bool do_track) {
  if (do_track) {
    return launch_fused_decode_update_static<Element, StaticW, UseSilu, UseResidual, true, Vec>(q, params);
  }
  return launch_fused_decode_update_static<Element, StaticW, UseSilu, UseResidual, false, Vec>(q, params);
}

template <typename Element, int StaticW, bool UseSilu, int Vec>
sycl::event launch_residual_selected(
    sycl::queue& q,
    FusedDecodeUpdateParams<Element> const& params,
    bool use_residual,
    bool do_track) {
  if (use_residual) {
    return launch_track_selected<Element, StaticW, UseSilu, true, Vec>(q, params, do_track);
  }
  return launch_track_selected<Element, StaticW, UseSilu, false, Vec>(q, params, do_track);
}

template <typename Element, int StaticW, int Vec>
sycl::event launch_activation_selected(
    sycl::queue& q,
    FusedDecodeUpdateParams<Element> const& params,
    bool use_silu,
    bool use_residual,
    bool do_track) {
  if (use_silu) {
    return launch_residual_selected<Element, StaticW, true, Vec>(q, params, use_residual, do_track);
  }
  return launch_residual_selected<Element, StaticW, false, Vec>(q, params, use_residual, do_track);
}

template <typename Element>
sycl::event launch_fused_decode_update(
    sycl::queue& q,
    FusedDecodeUpdateParams<Element> const& params,
    bool use_silu,
    bool use_residual,
    bool do_track) {
  switch (params.W) {
    case 2:
      return launch_activation_selected<Element, 2, kScalarVec>(q, params, use_silu, use_residual, do_track);
    case 3:
      return launch_activation_selected<Element, 3, kScalarVec>(q, params, use_silu, use_residual, do_track);
    case 4:
      if (params.D <= 512 && params.D % kW4WideVec == 0) {
        return launch_activation_selected<Element, 4, kW4WideVec>(q, params, use_silu, use_residual, do_track);
      }
      if (params.D % kW4FastVec == 0) {
        return launch_activation_selected<Element, 4, kW4FastVec>(q, params, use_silu, use_residual, do_track);
      }
      return launch_activation_selected<Element, 4, kScalarVec>(q, params, use_silu, use_residual, do_track);
    case 8:
      return launch_activation_selected<Element, 8, kScalarVec>(q, params, use_silu, use_residual, do_track);
    default:
      return launch_activation_selected<Element, 0, kScalarVec>(q, params, use_silu, use_residual, do_track);
  }
}

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
    if (count == 0) {
      return;
    }
    ptr = sycl::malloc_device<T>(count, q);
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
    if (!host.empty()) {
      queue->memcpy(ptr, host.data(), sizeof(T) * host.size()).wait();
    }
  }

  void copy_to(std::vector<T>& host) const {
    if (!host.empty()) {
      queue->memcpy(host.data(), ptr, sizeof(T) * host.size()).wait();
    }
  }
};

struct CaseConfig {
  std::string name;
  int T = 1;
  int D = 128;
  int W = 4;
  bool include_pad = false;
  bool include_masked = false;
  bool use_silu = true;
  bool use_residual = true;
  bool do_track = false;
  unsigned seed = 0;
};

CaseConfig make_case(
    std::string const& name,
    int T,
    int D,
    int W,
    bool include_pad,
    bool include_masked,
    bool use_silu,
    bool use_residual,
    bool do_track,
    unsigned seed = 0) {
  return CaseConfig{name, T, D, W, include_pad, include_masked, use_silu, use_residual, do_track, seed};
}

template <typename Element_>
struct HostTensors {
  using Element = Element_;

  std::vector<Element> x;
  std::vector<Element> cache;
  std::vector<Element> weight;
  std::vector<Element> y;
  std::vector<Element> ref_y;
  std::vector<Element> ref_cache;
  std::vector<int32_t> cache_indices;
  std::vector<uint8_t> cache_mask;
  std::vector<uint8_t> track_mask;
  std::vector<int64_t> track_indices;
  int slots = 0;
};

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
  h.slots = cfg.do_track ? (cfg.T * 2 + 16) : (cfg.T + 8);
  h.x.resize(static_cast<std::size_t>(cfg.T) * cfg.D);
  h.y.resize(static_cast<std::size_t>(cfg.T) * cfg.D);
  h.ref_y.resize(static_cast<std::size_t>(cfg.T) * cfg.D);
  h.cache.resize(static_cast<std::size_t>(h.slots) * (cfg.W - 1) * cfg.D);
  h.ref_cache.resize(h.cache.size());
  h.weight.resize(static_cast<std::size_t>(cfg.D) * cfg.W);
  h.cache_indices.resize(cfg.T);
  h.cache_mask.resize(cfg.T);
  h.track_mask.resize(cfg.T);
  h.track_indices.resize(cfg.T);

  unsigned seed = cfg.seed == 0
      ? 20260718u + static_cast<unsigned>(cfg.T * 11 + cfg.D * 7 + cfg.W * 3)
      : cfg.seed;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> x_dist(-0.9f, 0.9f);
  std::uniform_real_distribution<float> w_dist(-0.20f, 0.20f);
  std::uniform_real_distribution<float> c_dist(-0.10f, 0.10f);

  for (auto& v : h.x) {
    v = Element(x_dist(gen));
  }
  for (auto& v : h.weight) {
    v = Element(w_dist(gen));
  }
  for (auto& v : h.cache) {
    v = Element(c_dist(gen));
  }
  h.ref_cache = h.cache;

  for (int t = 0; t < cfg.T; ++t) {
    bool pad = cfg.include_pad && (t % 7 == 1);
    // Use a non-identity working-slot mapping so validation covers boundary
    // cache positions without introducing inter-token write races.
    h.cache_indices[t] = pad ? kPadSlot : (cfg.T - 1 - t);
    h.cache_mask[t] = static_cast<uint8_t>(pad ? 0 : 1);
    if (!pad && cfg.include_masked && (t % 11 == 3)) {
      h.cache_mask[t] = 0;
    }

    h.track_mask[t] = static_cast<uint8_t>(cfg.do_track && (t % 3 != 1));
    h.track_indices[t] = cfg.do_track ? static_cast<int64_t>(cfg.T + 8 + t) : 0;
  }

  return h;
}

template <typename Element>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  h.ref_cache = h.cache;
  int cache_stride_slot = (cfg.W - 1) * cfg.D;
  int cache_stride_w = cfg.D;

  for (int t = 0; t < cfg.T; ++t) {
    int ci = h.cache_indices[t];
    bool valid = ci != kPadSlot;
    int slot = valid ? ci : 0;
    bool mask = h.cache_mask[t] != 0;
    int cache_base = slot * cache_stride_slot;

    for (int d = 0; d < cfg.D; ++d) {
      float acc = 0.0f;
      for (int iw = 0; iw < cfg.W - 1; ++iw) {
        std::size_t cache_offset = static_cast<std::size_t>(cache_base)
            + static_cast<std::size_t>(iw) * cache_stride_w + d;
        acc += (mask ? to_float(h.cache[cache_offset]) : 0.0f)
            * to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + iw]);
      }
      acc += to_float(h.x[static_cast<std::size_t>(t) * cfg.D + d])
          * to_float(h.weight[static_cast<std::size_t>(d) * cfg.W + (cfg.W - 1)]);
      if (cfg.use_silu) {
        acc = silu(acc);
      }
      if (cfg.use_residual) {
        acc += to_float(h.x[static_cast<std::size_t>(t) * cfg.D + d]);
      }
      h.ref_y[static_cast<std::size_t>(t) * cfg.D + d] = Element(acc);
    }

    if (!valid) {
      continue;
    }

    bool track = cfg.do_track && h.track_mask[t] != 0;
    int track_base = track ? static_cast<int>(h.track_indices[t]) * cache_stride_slot : 0;
    for (int d = 0; d < cfg.D; ++d) {
      for (int iw = 0; iw < cfg.W - 1; ++iw) {
        Element next = (iw < cfg.W - 2)
            ? (mask ? h.cache[static_cast<std::size_t>(cache_base)
                              + static_cast<std::size_t>(iw + 1) * cache_stride_w + d]
                    : Element(0.0f))
            : h.x[static_cast<std::size_t>(t) * cfg.D + d];
        h.ref_cache[static_cast<std::size_t>(cache_base) + static_cast<std::size_t>(iw) * cache_stride_w + d] = next;
        if (track) {
          h.ref_cache[static_cast<std::size_t>(track_base) + static_cast<std::size_t>(iw) * cache_stride_w + d] = next;
        }
      }
    }
  }
}

struct VerifyResult {
  bool passed = true;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  int index = 0;
  float got = 0.0f;
  float expected = 0.0f;
};

template <typename Element>
VerifyResult verify_output(std::vector<Element> const& got, std::vector<Element> const& ref, bool use_silu) {
  VerifyResult result;
  float atol = std::is_same_v<Element, cutlass::bfloat16_t> ? 6.0e-2f : 5.0e-3f;
  float rtol = std::is_same_v<Element, cutlass::bfloat16_t> ? 2.0e-2f : 5.0e-3f;
  if (use_silu) {
    atol *= 1.5f;
    rtol *= 1.5f;
  }

  for (std::size_t i = 0; i < got.size(); ++i) {
    float g = to_float(got[i]);
    float r = to_float(ref[i]);
    float abs = std::abs(g - r);
    float rel = abs / std::max(std::abs(r), 1.0e-6f);
    if (abs > result.max_abs) {
      result.max_abs = abs;
      result.max_rel = rel;
      result.index = static_cast<int>(i);
      result.got = g;
      result.expected = r;
    }
    if (abs > atol + rtol * std::abs(r)) {
      result.passed = false;
    }
  }
  return result;
}

template <typename Element>
VerifyResult verify_cache(std::vector<Element> const& got, std::vector<Element> const& ref) {
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    float g = to_float(got[i]);
    float r = to_float(ref[i]);
    if (g != r) {
      result.passed = false;
      result.index = static_cast<int>(i);
      result.got = g;
      result.expected = r;
      result.max_abs = std::abs(g - r);
      result.max_rel = result.max_abs / std::max(std::abs(r), 1.0e-6f);
      return result;
    }
  }
  return result;
}

double gops_for(CaseConfig const& cfg) {
  return (2.0 * static_cast<double>(cfg.T) * cfg.D * cfg.W) / 1.0e9;
}

template <typename Element>
double effective_bytes(CaseConfig const& cfg, HostTensors<Element> const& h) {
  double element_bytes = static_cast<double>(sizeof(Element));
  double bytes = 0.0;

  bytes += static_cast<double>(cfg.T) * cfg.D * element_bytes;       // x
  bytes += static_cast<double>(cfg.D) * cfg.W * element_bytes;       // weight, counted once
  bytes += static_cast<double>(cfg.T) * cfg.D * (cfg.W - 1) * element_bytes; // cache history
  bytes += static_cast<double>(cfg.T) * cfg.D * element_bytes;       // y

  for (int t = 0; t < cfg.T; ++t) {
    if (h.cache_indices[t] == kPadSlot) {
      continue;
    }
    bytes += static_cast<double>(cfg.D) * (cfg.W - 1) * element_bytes; // cache stores
    if (h.cache_mask[t] != 0 && cfg.W > 2) {
      bytes += static_cast<double>(cfg.D) * (cfg.W - 2) * element_bytes; // shifted cache reads
    }
    if (cfg.do_track && h.track_mask[t] != 0) {
      bytes += static_cast<double>(cfg.D) * (cfg.W - 1) * element_bytes; // track stores
    }
  }

  return bytes;
}

template <typename Element>
bool run_case(
    sycl::queue& q,
    CaseConfig const& cfg,
    int iterations,
    bool verify,
    double target_tops,
    double target_gbps) {
  HostTensors<Element> h = initialize_case<Element>(cfg);
  if (verify) {
    reference_case<Element>(cfg, h);
  }

  DeviceBuffer<Element> d_x(q, h.x.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<Element> d_weight(q, h.weight.size());
  DeviceBuffer<Element> d_y(q, h.y.size());
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<uint8_t> d_cache_mask(q, h.cache_mask.size());
  DeviceBuffer<uint8_t> d_track_mask(q, h.track_mask.size());
  DeviceBuffer<int64_t> d_track_indices(q, h.track_indices.size());

  d_x.copy_from(h.x);
  d_cache.copy_from(h.cache);
  d_weight.copy_from(h.weight);
  d_cache_indices.copy_from(h.cache_indices);
  d_cache_mask.copy_from(h.cache_mask);
  d_track_mask.copy_from(h.track_mask);
  d_track_indices.copy_from(h.track_indices);

  FusedDecodeUpdateParams<Element> params{
      d_x.get(),
      d_cache.get(),
      d_cache_indices.get(),
      d_cache_mask.get(),
      d_weight.get(),
      d_y.get(),
      d_track_mask.get(),
      d_track_indices.get(),
      cfg.T,
      cfg.D,
      cfg.W,
      (cfg.W - 1) * cfg.D,
      cfg.D,
      kPadSlot};

  auto launch = [&]() {
    return launch_fused_decode_update<Element>(
        q, params, cfg.use_silu, cfg.use_residual, cfg.do_track);
  };

  launch();
  q.wait_and_throw();

  bool passed = true;
  VerifyResult y_result;
  VerifyResult cache_result;
  if (verify) {
    d_y.copy_to(h.y);
    std::vector<Element> got_cache(h.cache.size());
    d_cache.copy_to(got_cache);
    y_result = verify_output<Element>(h.y, h.ref_y, cfg.use_silu);
    cache_result = verify_cache<Element>(got_cache, h.ref_cache);
    passed = y_result.passed && cache_result.passed;
  }

  int timing_iterations = std::max(1, iterations);
  int warmup_iterations = std::min(10, std::max(2, timing_iterations));
  for (int i = 0; i < warmup_iterations; ++i) {
    launch();
  }
  q.wait_and_throw();

  std::vector<sycl::event> events;
  events.reserve(timing_iterations);
  for (int i = 0; i < timing_iterations; ++i) {
    events.push_back(launch());
  }
  q.wait_and_throw();

  double total_ns = 0.0;
  for (auto const& event : events) {
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_ns += static_cast<double>(end - start);
  }
  double avg_s = total_ns * 1.0e-9 / static_cast<double>(events.size());
  double useful_tops = gops_for(cfg) / avg_s / 1000.0;
  double bytes = effective_bytes<Element>(cfg, h);
  double gbps = (bytes / 1.0e9) / avg_s;

  bool applies_gbps_target = target_gbps > 0.0 && bytes >= kMinSustainedTargetBytes;
  if (target_tops > 0.0 && useful_tops < target_tops) {
    passed = false;
  }
  if (applies_gbps_target && gbps < target_gbps) {
    passed = false;
  }

  std::cout << std::left << std::setw(34) << cfg.name
            << " dtype=" << std::setw(4) << element_dtype_text<Element>()
            << " T=" << std::setw(6) << cfg.T
            << " D=" << std::setw(5) << cfg.D
            << " W=" << cfg.W
            << " track=" << bool_text(cfg.do_track)
            << " residual=" << bool_text(cfg.use_residual)
            << " silu=" << bool_text(cfg.use_silu)
            << "  " << std::fixed << std::setprecision(3)
            << (avg_s * 1000.0) << " ms"
            << "  " << useful_tops << " useful_TOPS"
            << "  " << gbps << " GB/s";

  if (target_tops > 0.0) {
    std::cout << "  target=" << target_tops << " useful_TOPS";
  }
  if (applies_gbps_target) {
    std::cout << "  target=" << target_gbps << " GB/s";
  } else if (target_gbps > 0.0) {
    std::cout << "  target=skipped-cache-smoke";
  }

  if (verify) {
    std::cout << "  " << (passed ? "passed" : "failed")
              << " y_abs=" << y_result.max_abs
              << " y_rel=" << y_result.max_rel
              << " y_index=" << y_result.index;
    if (!cache_result.passed) {
      std::cout << " cache_index=" << cache_result.index
                << " got=" << cache_result.got
                << " expected=" << cache_result.expected;
    }
  } else {
    std::cout << "  verification skipped";
  }
  std::cout << "\n";

  return passed;
}

std::vector<CaseConfig> quick_suite() {
  return {
      make_case("tiny_w2_d7_pad_track", 5, 7, 2, true, false, true, true, true),
      make_case("decode_b32_w4_d128", 32, 128, 4, false, false, false, true, false),
      make_case("masked_w4_d257_silu_track", 17, 257, 4, true, true, true, true, true),
      make_case("no_residual_w3_d33", 9, 33, 3, false, true, false, false, false),
      make_case("dynamic_w5_d65_masked", 13, 65, 5, true, true, true, true, true),
      make_case("inkling_decode_b128_d1536", 128, 1536, 4, true, true, true, true, true),
      make_case("inkling_kv_decode_b128_d512", 128, 512, 4, false, false, true, true, false),
  };
}

std::vector<CaseConfig> stress_suite() {
  return {
      make_case("stress_w2_t3_d1", 3, 1, 2, true, true, true, true, true, 1000u),
      make_case("stress_w3_t11_d31", 11, 31, 3, true, true, false, true, true, 1001u),
      make_case("stress_w4_t19_d129", 19, 129, 4, true, true, true, false, true, 1002u),
      make_case("stress_w5_t23_d257", 23, 257, 5, true, true, true, true, false, 1003u),
      make_case("stress_w8_t29_d770", 29, 770, 8, true, true, false, true, true, 1004u),
  };
}

std::vector<CaseConfig> perf_suite() {
  // Sustained-bandwidth production decode cases. T is increased for smaller D
  // so every case has hundreds of MiB to GiB of x/cache/y traffic; repeated
  // timing iterations should not be mistaken for small fixed-address cache hits.
  return {
      make_case("perf_decode_t65536_d1536", 65536, 1536, 4, false, false, true, true, false),
      make_case("perf_track_t65536_d1536", 65536, 1536, 4, false, false, true, true, true),
      make_case("perf_scattered_t65536_d768", 65536, 768, 4, false, false, true, true, false),
      make_case("perf_scattered_t131072_d384", 131072, 384, 4, false, false, true, true, false),
      make_case("perf_scattered_t262144_d192", 262144, 192, 4, false, false, true, true, false),
      make_case("perf_kv_t131072_d512", 131072, 512, 4, false, false, true, true, false),
      make_case("perf_kv_t131072_d256", 131072, 256, 4, false, false, true, true, false),
      make_case("perf_kv_t262144_d128", 262144, 128, 4, false, false, true, true, false),
  };
}

bool parse_single_shape(std::string const& text, CaseConfig& cfg) {
  if (text.empty()) {
    return false;
  }

  cfg = make_case("custom", 32, 128, 4, false, false, true, true, false);
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto pos = item.find('=');
    if (pos == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, pos);
    std::string value = item.substr(pos + 1);
    if (key == "name") {
      cfg.name = value;
    } else if (key == "T") {
      cfg.T = std::stoi(value);
    } else if (key == "D") {
      cfg.D = std::stoi(value);
    } else if (key == "W") {
      cfg.W = std::stoi(value);
    } else if (key == "pad") {
      cfg.include_pad = std::stoi(value) != 0;
    } else if (key == "masked") {
      cfg.include_masked = std::stoi(value) != 0;
    } else if (key == "silu") {
      cfg.use_silu = std::stoi(value) != 0;
    } else if (key == "residual") {
      cfg.use_residual = std::stoi(value) != 0;
    } else if (key == "track") {
      cfg.do_track = std::stoi(value) != 0;
    } else if (key == "seed") {
      cfg.seed = static_cast<unsigned>(std::stoul(value));
    } else {
      return false;
    }
  }

  return cfg.T > 0 && cfg.D > 0 && cfg.W >= 2 && cfg.W <= 16;
}

struct Options {
  bool help = false;
  bool valid = true;
  bool verify = true;
  int iterations = 20;
  std::string suite = "quick";
  std::string shape;
  std::string dtype_name = "all";
  DType dtype = DType::kAll;
  double target_tops = 0.0;
  double target_gbps = 0.0;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    int verify_int = 1;
    cmd.get_cmd_line_argument("verify", verify_int, 1);
    verify = verify_int != 0;
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("suite", suite, std::string("quick"));
    cmd.get_cmd_line_argument("shape", shape, std::string(""));
    cmd.get_cmd_line_argument("dtype", dtype_name, std::string("all"));
    cmd.get_cmd_line_argument("target-tops", target_tops, 0.0);
    cmd.get_cmd_line_argument("target-gbps", target_gbps, 0.0);
    if (!parse_dtype(dtype_name, dtype)) {
      valid = false;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Inkling BMG Fused Decode SConv Update Example\n\n"
        << "Options:\n"
        << "  --help                         Print this message\n"
        << "  --suite=<quick|stress|perf>     Built-in shape suite (default: quick)\n"
        << "  --shape=<k=v,...>               Run one custom shape instead of a suite\n"
        << "                                  Keys: name,T,D,W,pad,masked,silu,residual,track,seed\n"
        << "  --dtype=<all|bf16|fp16>         Input/cache/output dtype (default: all)\n"
        << "  --iterations=<int>              Timed kernel iterations\n"
        << "  --verify=<0|1>                  Run CPU reference comparison\n"
        << "  --target-tops=<float>           Fail if any timed case is below this useful TOPS\n"
        << "  --target-gbps=<float>           Fail if large working-set cases are below this effective GB/s\n\n"
        << "Examples:\n"
        << "  ./examples/14_bmg_sconv/14_bmg_fused_decode_update_sconv --suite=quick\n"
        << "  ./examples/14_bmg_sconv/14_bmg_fused_decode_update_sconv --suite=quick --dtype=fp16\n"
        << "  ./examples/14_bmg_sconv/14_bmg_fused_decode_update_sconv --suite=perf --verify=0 --iterations=100\n"
        << "  ./examples/14_bmg_sconv/14_bmg_fused_decode_update_sconv --shape=T=128,D=1536,W=4,silu=1,residual=1,track=1\n";
    return out;
  }
};

}  // namespace cutlass::examples::sconv

int main(int argc, char const** argv) {
  using namespace cutlass::examples::sconv;

  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }
  if (!options.valid) {
    std::cerr << "Unsupported dtype: " << options.dtype_name << "\n";
    options.print_usage(std::cerr);
    return 1;
  }

  std::vector<CaseConfig> cases;
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
  } else if (options.suite == "stress") {
    cases = stress_suite();
  } else if (options.suite == "perf") {
    cases = perf_suite();
  } else {
    std::cerr << "Unknown suite: " << options.suite << "\n";
    options.print_usage(std::cerr);
    return 1;
  }

  try {
    sycl::queue base_queue = compat::get_default_queue();
    sycl::queue q(
        base_queue.get_context(),
        base_queue.get_device(),
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Suite: " << (options.shape.empty() ? options.suite : "custom")
              << ", cases=" << cases.size()
              << ", dtype=" << dtype_text(options.dtype)
              << ", iterations=" << options.iterations
              << ", verify=" << bool_text(options.verify) << "\n";

    bool all_passed = true;
    for (auto const& cfg : cases) {
      if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
        all_passed &= run_case<cutlass::bfloat16_t>(
            q, cfg, options.iterations, options.verify, options.target_tops, options.target_gbps);
      }
      if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
        all_passed &= run_case<cutlass::half_t>(
            q, cfg, options.iterations, options.verify, options.target_tops, options.target_gbps);
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
