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
    \brief Inkling packed causal depthwise short-convolution example for CUTLASS SYCL.

    Semantics match the reference in modeltune/inkling/01_sconv/01_01_causal_conv1d:

      history = concat(cache[sequence_slot], packed_sequence_tokens)
      out[t,d] = activation(sum_iw history[prefix + t_in_sequence - iw, d] * weight[d,iw] + residual[t,d])

    The production Inkling metadata form is used directly: packed tokens [T,D],
    cache [slots,W-1,D], weight [D,W], sequence starts cu, per-token sequence ids,
    safe cache slot ids, and raw bool cache masks. Accumulation is fp32 and the
    output is bf16.
*/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cutlass::examples::sconv {

constexpr int kThreads = 384;
constexpr int kBlockT = 4;
constexpr int kRegularBlockT = 4;
constexpr int kVec = 1;

enum class DType {
  kBf16,
  kFp16
};

template <typename Element_>
struct DeviceParams {
  using Element = Element_;

  Element const* __restrict__ x;
  Element const* __restrict__ cache;
  int64_t const* __restrict__ safe_idx;
  uint8_t const* __restrict__ cache_mask;
  Element const* __restrict__ weight;
  Element const* __restrict__ residual;
  int64_t const* __restrict__ cu;
  int32_t const* __restrict__ seq_idx;
  Element* __restrict__ y;
  int T;
  int D;
  int cache_stride_slot;
  int cache_stride_w;
  int regular_tokens_per_seq_log2;
};

template <
    typename Element,
    int W,
    bool UseSilu,
    bool UseResidual,
    bool IsDecode,
    int Vec,
    int BlockT,
    bool RegularSequenceFastPath,
    bool PairFastPath>
class CausalSconvKernel;

template <typename T>
struct DeviceBuffer {
  sycl::queue* queue = nullptr;
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;

  DeviceBuffer(sycl::queue& q, std::size_t n) : queue(&q), count(n) {
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
    queue->memcpy(ptr, host.data(), sizeof(T) * host.size()).wait();
  }

  void copy_to(std::vector<T>& host) const {
    queue->memcpy(host.data(), ptr, sizeof(T) * host.size()).wait();
  }
};

struct CaseConfig {
  std::string name;
  int T = 0;
  int D = 0;
  int W = 4;
  int batch = 1;
  int tokens_per_seq = 1;
  bool varied_lengths = false;
  bool use_silu = false;
  bool use_residual = true;
  bool is_decode = false;
};

template <typename Element_>
struct HostTensors {
  using Element = Element_;

  std::vector<Element> x;
  std::vector<Element> cache;
  std::vector<Element> weight;
  std::vector<Element> residual;
  std::vector<Element> y;
  std::vector<Element> ref;
  std::vector<int64_t> safe_idx;
  std::vector<uint8_t> cache_mask;
  std::vector<int64_t> cu;
  std::vector<int32_t> seq_idx;
  int slots = 0;
};

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
  return dtype == DType::kBf16 ? "bf16" : "fp16";
}

bool parse_dtype(std::string const& text, DType& dtype) {
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

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

std::string bool_text(bool value) {
  return value ? "true" : "false";
}

int log2_if_power_of_two(int value) {
  if (value <= 0 || (value & (value - 1)) != 0) {
    return -1;
  }
  int result = 0;
  while ((1 << result) != value) {
    ++result;
  }
  return result;
}

std::vector<int> make_lengths(CaseConfig const& cfg) {
  std::vector<int> lengths(cfg.batch, cfg.tokens_per_seq);
  if (cfg.varied_lengths && cfg.batch > 1) {
    for (int b = 0; b < cfg.batch; ++b) {
      int delta = (b % 5) - 2;
      lengths[b] = std::max(1, cfg.tokens_per_seq + delta);
    }
    int sum = std::accumulate(lengths.begin(), lengths.end(), 0);
    int diff = cfg.T - sum;
    for (int b = 0; diff != 0; b = (b + 1) % cfg.batch) {
      if (diff > 0) {
        ++lengths[b];
        --diff;
      } else if (lengths[b] > 1) {
        --lengths[b];
        ++diff;
      }
    }
  }
  return lengths;
}

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
  h.slots = std::max(cfg.batch + 7, 8);
  h.x.resize(static_cast<std::size_t>(cfg.T) * cfg.D);
  h.y.resize(static_cast<std::size_t>(cfg.T) * cfg.D);
  h.ref.resize(static_cast<std::size_t>(cfg.T) * cfg.D);
  h.cache.resize(static_cast<std::size_t>(h.slots) * (cfg.W - 1) * cfg.D);
  h.weight.resize(static_cast<std::size_t>(cfg.D) * cfg.W);
  h.residual.resize(static_cast<std::size_t>(cfg.T) * cfg.D);
  h.safe_idx.resize(cfg.batch);
  h.cache_mask.resize(cfg.batch);
  h.cu.resize(cfg.batch + 1);
  h.seq_idx.resize(cfg.T);

  std::mt19937 gen(20260717u + static_cast<unsigned>(cfg.T * 3 + cfg.D * 5 + cfg.W));
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> w_dist(-0.20f, 0.20f);
  std::uniform_real_distribution<float> c_dist(-0.10f, 0.10f);
  std::uniform_real_distribution<float> r_dist(-0.05f, 0.05f);

  for (auto& v : h.x) {
    v = Element(x_dist(gen));
  }
  for (auto& v : h.weight) {
    v = Element(w_dist(gen));
  }
  for (auto& v : h.cache) {
    v = Element(c_dist(gen));
  }
  for (auto& v : h.residual) {
    v = Element(r_dist(gen));
  }

  if (cfg.name == "reference_w3_tiny") {
    h.cu = {0, 3, 7};
    h.safe_idx = {0, 1};
    h.cache_mask = {1, 1};
    for (int t = 0; t < 3; ++t) {
      h.seq_idx[t] = 0;
    }
    for (int t = 3; t < 7; ++t) {
      h.seq_idx[t] = 1;
    }
    return h;
  }

  auto lengths = make_lengths(cfg);
  h.cu[0] = 0;
  for (int b = 0; b < cfg.batch; ++b) {
    h.safe_idx[b] = b;
    h.cache_mask[b] = static_cast<uint8_t>(cfg.is_decode || (b % 7 != 3));
    h.cu[b + 1] = h.cu[b] + lengths[b];
    for (int t = static_cast<int>(h.cu[b]); t < static_cast<int>(h.cu[b + 1]); ++t) {
      h.seq_idx[t] = b;
    }
  }
  return h;
}

template <typename Element, int W>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  for (int t = 0; t < cfg.T; ++t) {
    int seq = h.seq_idx[t];
    int bos = static_cast<int>(h.cu[seq]);
    int slot = static_cast<int>(h.safe_idx[seq]);
    bool mask = h.cache_mask[seq] != 0;

    for (int d = 0; d < cfg.D; ++d) {
      float acc = 0.0f;
      for (int iw = 0; iw < W; ++iw) {
        int shifted = t - iw;
        float tap = 0.0f;
        if (shifted >= bos) {
          tap = to_float(h.x[static_cast<std::size_t>(shifted) * cfg.D + d]);
        } else {
          int prefix_pos = shifted - bos + (W - 1);
          if (prefix_pos >= 0 && prefix_pos < W - 1) {
            std::size_t offset = static_cast<std::size_t>(slot) * (W - 1) * cfg.D
                + static_cast<std::size_t>(prefix_pos) * cfg.D + d;
            tap = (cfg.is_decode || mask) ? to_float(h.cache[offset]) : 0.0f;
          }
        }
        acc += tap * to_float(h.weight[static_cast<std::size_t>(d) * W + iw]);
      }
      if (cfg.use_residual) {
        acc += to_float(h.residual[static_cast<std::size_t>(t) * cfg.D + d]);
      }
      if (cfg.use_silu) {
        acc = silu(acc);
      }
      h.ref[static_cast<std::size_t>(t) * cfg.D + d] = Element(acc);
    }
  }
}

template <int Vec>
CUTLASS_DEVICE
void set_channel_valid(int c0, int D, bool (&channel_valid)[Vec]) {
#pragma unroll
  for (int v = 0; v < Vec; ++v) {
    channel_valid[v] = c0 + v < D;
  }
}

template <typename Element, int W, int Vec>
CUTLASS_DEVICE
void load_sconv_weights(
    DeviceParams<Element> const& params,
    int c0,
    bool const (&channel_valid)[Vec],
    float (&weights)[W][Vec]) {
#pragma unroll
  for (int iw = 0; iw < W; ++iw) {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      int c = c0 + v;
      weights[iw][v] = channel_valid[v]
          ? to_float(params.weight[static_cast<std::size_t>(c) * W + iw])
          : 0.0f;
    }
  }
}

template <typename Element, int W, int Vec, int BlockT, bool RegularSequenceFastPath, bool PairFastPath>
CUTLASS_DEVICE
void load_sconv_window(
    DeviceParams<Element> const& params,
    int c0,
    int t0,
    bool const (&channel_valid)[Vec],
    uint64_t (&x_raw)[BlockT + W - 1],
    float (&x_window)[BlockT + W - 1][Vec]) {
#pragma unroll
  for (int r = 0; r < BlockT + W - 1; ++r) {
    int row = t0 - (W - 1) + r;
    if constexpr (RegularSequenceFastPath) {
      if (row >= 0 && row < params.T) {
        auto base = params.x + static_cast<std::size_t>(row) * params.D + c0;
        x_raw[r] = *reinterpret_cast<uint64_t const*>(base);
      } else {
        x_raw[r] = 0;
      }
    } else if constexpr (PairFastPath) {
      if (row >= 0 && row < params.T) {
        auto base = params.x + static_cast<std::size_t>(row) * params.D + c0;
        if (channel_valid[1]) {
          uint32_t raw = *reinterpret_cast<uint32_t const*>(base);
          x_window[r][0] = to_float(Element::bitcast(static_cast<uint16_t>(raw & 0xffffu)));
          x_window[r][1] = to_float(Element::bitcast(static_cast<uint16_t>(raw >> 16)));
        } else {
          x_window[r][0] = to_float(*base);
          x_window[r][1] = 0.0f;
        }
      } else {
        x_window[r][0] = 0.0f;
        x_window[r][1] = 0.0f;
      }
    } else {
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        int c = c0 + v;
        x_window[r][v] = (row >= 0 && row < params.T && c < params.D)
            ? to_float(params.x[static_cast<std::size_t>(row) * params.D + c])
            : 0.0f;
      }
    }
  }
}

template <typename Element, int W, bool IsDecode, bool RegularSequenceFastPath>
CUTLASS_DEVICE
void get_sconv_sequence_state(
    DeviceParams<Element> const& params,
    int t,
    int sequence_mask,
    int& bos,
    int& slot,
    bool& mask) {
  slot = 0;
  mask = true;
  if constexpr (RegularSequenceFastPath) {
    int local_t = t & sequence_mask;
    if (local_t < W - 1) {
      int seq = t >> params.regular_tokens_per_seq_log2;
      bos = t - local_t;
      slot = static_cast<int>(params.safe_idx[seq]);
      mask = params.cache_mask[seq] != 0;
    } else {
      bos = t - (W - 1);
    }
  } else {
    int seq = params.seq_idx[t];
    bos = static_cast<int>(params.cu[seq]);
    slot = static_cast<int>(params.safe_idx[seq]);
    if constexpr (!IsDecode) {
      mask = params.cache_mask[seq] != 0;
    }
  }
}

template <typename Element, int W, int Vec, int BlockT, bool IsDecode, bool RegularSequenceFastPath, bool PairFastPath>
CUTLASS_DEVICE
void prepare_sconv_taps(
    DeviceParams<Element> const& params,
    int c0,
    int j,
    int t,
    int bos,
    int slot,
    bool mask,
    bool const (&channel_valid)[Vec],
    uint64_t const (&x_raw)[BlockT + W - 1],
    float const (&x_window)[BlockT + W - 1][Vec],
    float (&tap_values)[W][Vec]) {
#pragma unroll
  for (int iw = 0; iw < W; ++iw) {
    if constexpr (RegularSequenceFastPath) {
      uint64_t raw = x_raw[j + (W - 1 - iw)];
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        tap_values[iw][v] = to_float(Element::bitcast(static_cast<uint16_t>(raw >> (16 * v))));
      }
    } else {
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        tap_values[iw][v] = x_window[j + (W - 1 - iw)][v];
      }
    }
  }

  if constexpr (RegularSequenceFastPath) {
    if (t - (W - 1) < bos) {
#pragma unroll
      for (int iw = 1; iw < W; ++iw) {
        int shifted = t - iw;
        int prefix_pos = shifted - bos + (W - 1);
        bool in_prefix = shifted < bos && prefix_pos >= 0 && prefix_pos < W - 1;
        if (shifted < bos) {
#pragma unroll
          for (int v = 0; v < Vec; ++v) {
            tap_values[iw][v] = 0.0f;
          }
          if (in_prefix && mask) {
            std::size_t cache_offset = static_cast<std::size_t>(slot) * params.cache_stride_slot
                + static_cast<std::size_t>(prefix_pos) * params.cache_stride_w + c0;
            auto base = params.cache + cache_offset;
            uint64_t raw = *reinterpret_cast<uint64_t const*>(base);
#pragma unroll
            for (int v = 0; v < Vec; ++v) {
              tap_values[iw][v] = to_float(Element::bitcast(static_cast<uint16_t>(raw >> (16 * v))));
            }
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int iw = 0; iw < W; ++iw) {
      int shifted = t - iw;
      int prefix_pos = shifted - bos + (W - 1);
      bool in_prefix = shifted < bos && prefix_pos >= 0 && prefix_pos < W - 1;
      if (shifted < bos) {
#pragma unroll
        for (int v = 0; v < Vec; ++v) {
          tap_values[iw][v] = 0.0f;
        }
        if (in_prefix && (IsDecode || mask)) {
          std::size_t cache_offset = static_cast<std::size_t>(slot) * params.cache_stride_slot
              + static_cast<std::size_t>(prefix_pos) * params.cache_stride_w + c0;
          auto base = params.cache + cache_offset;
          if constexpr (PairFastPath) {
            if (channel_valid[1]) {
              uint32_t raw = *reinterpret_cast<uint32_t const*>(base);
              tap_values[iw][0] = to_float(Element::bitcast(static_cast<uint16_t>(raw & 0xffffu)));
              tap_values[iw][1] = to_float(Element::bitcast(static_cast<uint16_t>(raw >> 16)));
            } else {
              tap_values[iw][0] = to_float(*base);
            }
          } else {
#pragma unroll
            for (int v = 0; v < Vec; ++v) {
              int c = c0 + v;
              if (c < params.D) {
                tap_values[iw][v] = to_float(params.cache[cache_offset + v]);
              }
            }
          }
        }
      }
    }
  }
}

template <int W, int Vec, bool RegularSequenceFastPath, bool PairFastPath>
CUTLASS_DEVICE
void accumulate_sconv(
    bool const (&channel_valid)[Vec],
    float const (&weights)[W][Vec],
    float const (&tap_values)[W][Vec],
    float (&acc)[Vec]) {
#pragma unroll
  for (int v = 0; v < Vec; ++v) {
    acc[v] = 0.0f;
  }

#pragma unroll
  for (int iw = 0; iw < W; ++iw) {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      if constexpr (RegularSequenceFastPath || PairFastPath) {
        acc[v] += tap_values[iw][v] * weights[iw][v];
      } else if (channel_valid[v]) {
        acc[v] += tap_values[iw][v] * weights[iw][v];
      }
    }
  }
}

template <typename Element, int Vec, bool UseResidual, bool RegularSequenceFastPath, bool PairFastPath>
CUTLASS_DEVICE
void add_sconv_residual(
    DeviceParams<Element> const& params,
    int c0,
    int t,
    bool const (&channel_valid)[Vec],
    float (&out_values)[Vec]) {
  if constexpr (UseResidual) {
    auto residual_base = params.residual + static_cast<std::size_t>(t) * params.D + c0;
    if constexpr (RegularSequenceFastPath) {
      uint64_t raw = *reinterpret_cast<uint64_t const*>(residual_base);
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        out_values[v] += to_float(Element::bitcast(static_cast<uint16_t>(raw >> (16 * v))));
      }
    } else if constexpr (PairFastPath) {
      if (channel_valid[1]) {
        uint32_t raw = *reinterpret_cast<uint32_t const*>(residual_base);
        out_values[0] += to_float(Element::bitcast(static_cast<uint16_t>(raw & 0xffffu)));
        out_values[1] += to_float(Element::bitcast(static_cast<uint16_t>(raw >> 16)));
      } else {
        out_values[0] += to_float(*residual_base);
      }
    } else {
#pragma unroll
      for (int v = 0; v < Vec; ++v) {
        int c = c0 + v;
        if (c < params.D) {
          out_values[v] += to_float(params.residual[static_cast<std::size_t>(t) * params.D + c]);
        }
      }
    }
  }
}

template <int Vec, bool UseSilu, bool RegularSequenceFastPath, bool PairFastPath>
CUTLASS_DEVICE
void apply_sconv_silu(bool const (&channel_valid)[Vec], float (&out_values)[Vec]) {
  if constexpr (UseSilu) {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      if constexpr (RegularSequenceFastPath || PairFastPath) {
        out_values[v] = out_values[v] / (1.0f + sycl::exp(-out_values[v]));
      } else if (channel_valid[v]) {
        out_values[v] = out_values[v] / (1.0f + sycl::exp(-out_values[v]));
      }
    }
  }
}

template <typename Element, int Vec, bool RegularSequenceFastPath, bool PairFastPath>
CUTLASS_DEVICE
void store_sconv_output(
    DeviceParams<Element> const& params,
    int c0,
    int t,
    bool const (&channel_valid)[Vec],
    float const (&out_values)[Vec]) {
  auto out = params.y + static_cast<std::size_t>(t) * params.D + c0;
  if constexpr (RegularSequenceFastPath) {
    uint64_t raw = 0;
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      raw |= static_cast<uint64_t>(Element(out_values[v]).raw()) << (16 * v);
    }
    *reinterpret_cast<uint64_t*>(out) = raw;
  } else if constexpr (PairFastPath) {
    if (channel_valid[1]) {
      uint32_t raw = static_cast<uint32_t>(Element(out_values[0]).raw())
          | (static_cast<uint32_t>(Element(out_values[1]).raw()) << 16);
      *reinterpret_cast<uint32_t*>(out) = raw;
    } else {
      *out = Element(out_values[0]);
    }
  } else {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      int c = c0 + v;
      if (c < params.D) {
        params.y[static_cast<std::size_t>(t) * params.D + c] = Element(out_values[v]);
      }
    }
  }
}

template <
    typename Element,
    int W,
    bool UseSilu,
    bool UseResidual,
    bool IsDecode,
    int Vec = kVec,
    int BlockT = kBlockT,
    bool RegularSequenceFastPath = false,
    bool PairFastPath = false>
void run_sconv_kernel(DeviceParams<Element> const& params, sycl::nd_item<2> item) {
  static_assert(!(RegularSequenceFastPath && PairFastPath), "fast path template switches are mutually exclusive");
  if constexpr (RegularSequenceFastPath) {
    static_assert(W == 4, "regular sequence fast path requires W=4");
    static_assert(Vec == 4, "regular sequence fast path requires Vec=4");
    static_assert(UseResidual, "regular sequence fast path requires residual");
    static_assert(!UseSilu, "regular sequence fast path does not support silu");
    static_assert(!IsDecode, "regular sequence fast path is for prefill only");
  }
  if constexpr (PairFastPath) {
    static_assert(W == 4, "pair fast path requires W=4");
    static_assert(Vec == 2, "pair fast path requires Vec=2");
    static_assert(UseResidual, "pair fast path requires residual");
    static_assert(!UseSilu, "pair fast path does not support silu");
  }

  int channel_blocks = RegularSequenceFastPath ? params.D / Vec : (params.D + Vec - 1) / Vec;
  int cb = static_cast<int>(item.get_global_id(0));
  int tb = static_cast<int>(item.get_global_id(1));
  if (cb >= channel_blocks) {
    return;
  }

  int c0 = cb * Vec;
  int t0 = tb * BlockT;
  bool channel_valid[Vec];
  set_channel_valid<Vec>(c0, params.D, channel_valid);

  float weights[W][Vec];
  load_sconv_weights<Element, W, Vec>(params, c0, channel_valid, weights);

  uint64_t x_raw[BlockT + W - 1];
  float x_window[BlockT + W - 1][Vec];
  load_sconv_window<Element, W, Vec, BlockT, RegularSequenceFastPath, PairFastPath>(
      params, c0, t0, channel_valid, x_raw, x_window);

  int sequence_mask = 0;
  if constexpr (RegularSequenceFastPath) {
    sequence_mask = (1 << params.regular_tokens_per_seq_log2) - 1;
  }

#pragma unroll
  for (int j = 0; j < BlockT; ++j) {
    int t = t0 + j;
    if (t >= params.T) {
      return;
    }

    int bos;
    int slot;
    bool mask;
    get_sconv_sequence_state<Element, W, IsDecode, RegularSequenceFastPath>(
        params, t, sequence_mask, bos, slot, mask);

    float tap_values[W][Vec];
    prepare_sconv_taps<Element, W, Vec, BlockT, IsDecode, RegularSequenceFastPath, PairFastPath>(
        params, c0, j, t, bos, slot, mask, channel_valid, x_raw, x_window, tap_values);

    float acc[Vec];
    accumulate_sconv<W, Vec, RegularSequenceFastPath, PairFastPath>(
        channel_valid, weights, tap_values, acc);

    float out_values[Vec];
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      out_values[v] = acc[v];
    }

    add_sconv_residual<Element, Vec, UseResidual, RegularSequenceFastPath, PairFastPath>(
        params, c0, t, channel_valid, out_values);
    apply_sconv_silu<Vec, UseSilu, RegularSequenceFastPath, PairFastPath>(
        channel_valid, out_values);

    store_sconv_output<Element, Vec, RegularSequenceFastPath, PairFastPath>(
        params, c0, t, channel_valid, out_values);
  }
}

template <
    typename Element,
    int W,
    bool UseSilu,
    bool UseResidual,
    bool IsDecode,
    int Vec = kVec,
    int BlockT = kBlockT,
    bool RegularSequenceFastPath = false,
    bool PairFastPath = false>
void launch_sconv(sycl::queue& q, DeviceParams<Element> const& params) {
  int channel_blocks = (params.D + Vec - 1) / Vec;
  int token_blocks = (params.T + BlockT - 1) / BlockT;
  int rounded_channel_blocks = ((channel_blocks + kThreads - 1) / kThreads) * kThreads;

  sycl::range<2> local(kThreads, 1);
  sycl::range<2> global(rounded_channel_blocks, token_blocks);

  q.parallel_for<CausalSconvKernel<Element, W, UseSilu, UseResidual, IsDecode, Vec, BlockT, RegularSequenceFastPath, PairFastPath>>(
      sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
        run_sconv_kernel<Element, W, UseSilu, UseResidual, IsDecode, Vec, BlockT, RegularSequenceFastPath, PairFastPath>(
            params, item);
      });
}

template <typename Element, int W, bool UseSilu, bool UseResidual, bool IsDecode>
void launch_selected(sycl::queue& q, DeviceParams<Element> const& params) {
  launch_sconv<Element, W, UseSilu, UseResidual, IsDecode>(q, params);
}

template <typename Element, int W, bool UseSilu, bool UseResidual>
void launch_decode_selected(sycl::queue& q, DeviceParams<Element> const& params, bool is_decode) {
  if (is_decode) {
    launch_selected<Element, W, UseSilu, UseResidual, true>(q, params);
  } else {
    launch_selected<Element, W, UseSilu, UseResidual, false>(q, params);
  }
}

template <typename Element, int W, bool UseSilu>
void launch_residual_selected(sycl::queue& q, DeviceParams<Element> const& params, bool use_residual, bool is_decode) {
  if (use_residual) {
    launch_decode_selected<Element, W, UseSilu, true>(q, params, is_decode);
  } else {
    launch_decode_selected<Element, W, UseSilu, false>(q, params, is_decode);
  }
}

template <typename Element, int W>
void launch_runtime(sycl::queue& q, DeviceParams<Element> const& params, bool use_silu, bool use_residual, bool is_decode) {
  if constexpr (W == 4) {
    if (!use_silu && use_residual && params.D % 2 == 0) {
      if (!is_decode && params.regular_tokens_per_seq_log2 >= 0) {
        if (params.D % 4 == 0) {
          launch_sconv<Element, 4, false, true, false, 4, kRegularBlockT, true>(q, params);
        } else {
          launch_sconv<Element, 4, false, true, false, 2, kBlockT, false, true>(q, params);
        }
        return;
      }
      if (is_decode) {
        launch_sconv<Element, 4, false, true, true, 2, kBlockT, false, true>(q, params);
      } else {
        launch_sconv<Element, 4, false, true, false, 2, kBlockT, false, true>(q, params);
      }
      return;
    }
  }
  if (use_silu) {
    launch_residual_selected<Element, W, true>(q, params, use_residual, is_decode);
  } else {
    launch_residual_selected<Element, W, false>(q, params, use_residual, is_decode);
  }
}

struct VerifyResult {
  bool passed = true;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  int index = 0;
};

template <typename Element>
VerifyResult verify_output(std::vector<Element> const& got, std::vector<Element> const& ref, bool use_silu) {
  VerifyResult result;
  float atol = use_silu ? 2.5e-2f : 1.0e-2f;
  float rtol = use_silu ? 2.5e-2f : 1.0e-2f;
  for (std::size_t i = 0; i < got.size(); ++i) {
    float g = to_float(got[i]);
    float r = to_float(ref[i]);
    float abs = std::abs(g - r);
    float rel = abs / std::max(std::abs(r), 1.0e-6f);
    if (abs > result.max_abs) {
      result.max_abs = abs;
      result.max_rel = rel;
      result.index = static_cast<int>(i);
    }
    if (abs > atol + rtol * std::abs(r)) {
      result.passed = false;
    }
  }
  return result;
}

double gops_for(CaseConfig const& cfg) {
  return (2.0 * static_cast<double>(cfg.T) * cfg.D * cfg.W) / 1.0e9;
}

template <typename Element>
double minimum_streaming_bytes(CaseConfig const& cfg, int slots) {
  double bytes = static_cast<double>(cfg.T) * cfg.D * sizeof(Element) * 2.0
      + static_cast<double>(cfg.D) * cfg.W * sizeof(Element)
      + static_cast<double>(slots) * (cfg.W - 1) * cfg.D * sizeof(Element);
  if (cfg.use_residual) {
    bytes += static_cast<double>(cfg.T) * cfg.D * sizeof(Element);
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
    if (cfg.W == 3) {
      reference_case<Element, 3>(cfg, h);
    } else if (cfg.W == 4) {
      reference_case<Element, 4>(cfg, h);
    } else {
      std::cerr << "Unsupported W=" << cfg.W << " in reference\n";
      return false;
    }
  }

  DeviceBuffer<Element> d_x(q, h.x.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<Element> d_weight(q, h.weight.size());
  DeviceBuffer<Element> d_residual(q, h.residual.size());
  DeviceBuffer<Element> d_y(q, h.y.size());
  DeviceBuffer<int64_t> d_safe_idx(q, h.safe_idx.size());
  DeviceBuffer<uint8_t> d_cache_mask(q, h.cache_mask.size());
  DeviceBuffer<int64_t> d_cu(q, h.cu.size());
  DeviceBuffer<int32_t> d_seq_idx(q, h.seq_idx.size());

  d_x.copy_from(h.x);
  d_cache.copy_from(h.cache);
  d_weight.copy_from(h.weight);
  d_residual.copy_from(h.residual);
  d_safe_idx.copy_from(h.safe_idx);
  d_cache_mask.copy_from(h.cache_mask);
  d_cu.copy_from(h.cu);
  d_seq_idx.copy_from(h.seq_idx);

  DeviceParams<Element> params{
      d_x.get(),
      d_cache.get(),
      d_safe_idx.get(),
      d_cache_mask.get(),
      d_weight.get(),
      d_residual.get(),
      d_cu.get(),
      d_seq_idx.get(),
      d_y.get(),
      cfg.T,
      cfg.D,
      (cfg.W - 1) * cfg.D,
      cfg.D,
      (!cfg.is_decode && !cfg.varied_lengths && cfg.T == cfg.batch * cfg.tokens_per_seq)
          ? log2_if_power_of_two(cfg.tokens_per_seq)
          : -1};

  auto launch = [&]() {
    if (cfg.W == 3) {
      launch_runtime<Element, 3>(q, params, cfg.use_silu, cfg.use_residual, cfg.is_decode);
    } else if (cfg.W == 4) {
      launch_runtime<Element, 4>(q, params, cfg.use_silu, cfg.use_residual, cfg.is_decode);
    } else {
      throw std::runtime_error("unsupported W");
    }
  };

  launch();
  q.wait_and_throw();

  bool passed = true;
  VerifyResult vr;
  if (verify) {
    d_y.copy_to(h.y);
    vr = verify_output<Element>(h.y, h.ref, cfg.use_silu);
    passed = vr.passed;
  }

  int timing_iterations = std::max(1, iterations);
  int warmup_iterations = std::min(10, std::max(2, timing_iterations));
  for (int i = 0; i < warmup_iterations; ++i) {
    launch();
  }
  q.wait_and_throw();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    launch();
  }
  q.wait_and_throw();
  double avg_s = timer.seconds() / timing_iterations;
  double useful_tops = gops_for(cfg) / avg_s / 1000.0;
  double bytes = minimum_streaming_bytes<Element>(cfg, h.slots);
  double gbps = (bytes / 1.0e9) / avg_s;

  std::cout << std::left << std::setw(28) << cfg.name
            << " T=" << std::setw(7) << cfg.T
            << " D=" << std::setw(5) << cfg.D
            << " W=" << cfg.W
            << " decode=" << bool_text(cfg.is_decode)
            << " residual=" << bool_text(cfg.use_residual)
            << " silu=" << bool_text(cfg.use_silu)
            << "  " << std::fixed << std::setprecision(3)
            << (avg_s * 1000.0) << " ms"
            << "  " << useful_tops << " useful_TOPS"
            << "  " << gbps << " GB/s";

  if (target_tops > 0.0) {
    double target_s = gops_for(cfg) / (target_tops * 1000.0);
    double required_tbps = (bytes / target_s) / 1.0e12;
    std::cout << "  target=" << target_tops << " useful_TOPS"
              << " needs=" << (target_s * 1000.0) << " ms/"
              << required_tbps << " TB/s";
  }
  if (target_gbps > 0.0) {
    std::cout << "  target=" << target_gbps << " GB/s";
  }

  if (verify) {
    std::cout << "  " << (passed ? "passed" : "failed")
              << " max_abs=" << vr.max_abs
              << " max_rel=" << vr.max_rel
              << " index=" << vr.index;
  } else {
    std::cout << "  verification skipped";
  }
  std::cout << "\n";

  if (target_tops > 0.0 && useful_tops < target_tops) {
    passed = false;
  }
  if (target_gbps > 0.0 && gbps < target_gbps) {
    passed = false;
  }

  return passed;
}

std::vector<CaseConfig> quick_suite() {
  return {
      {"reference_w3_tiny", 7, 4, 3, 2, 1, false, true, true, false},
      {"decode_b32_d128", 32, 128, 4, 32, 1, false, false, true, true},
      {"extend_b4_l64_d512", 256, 512, 4, 4, 64, true, false, true, false},
      {"verify_b16_m8_d1536", 128, 1536, 4, 16, 8, false, false, true, false},
      {"silu_residual_b4_l16_d256", 64, 256, 4, 4, 16, true, true, true, false},
      {"edge_varied_b5_l13_d257", 65, 257, 4, 5, 13, true, false, true, false},
      {"edge_pair_b3_l32_d770", 96, 770, 4, 3, 32, false, false, true, false},
  };
}

std::vector<CaseConfig> inkling_suite() {
  return {
      {"reference_w3_tiny", 7, 4, 3, 2, 1, false, true, true, false},
      {"decode_b64_d128", 64, 128, 4, 64, 1, false, false, true, true},
      {"decode_b128_d512", 128, 512, 4, 128, 1, false, false, true, true},
      {"decode_b128_d1536", 128, 1536, 4, 128, 1, false, false, true, true},
      {"extend_b8_l128_d128", 1024, 128, 4, 8, 128, true, false, true, false},
      {"extend_b8_l512_d512", 4096, 512, 4, 8, 512, true, false, true, false},
      {"extend_b8_l1024_d1536", 8192, 1536, 4, 8, 1024, true, false, true, false},
      {"verify_b32_m8_d1536", 256, 1536, 4, 32, 8, false, false, true, false},
      {"scattered_tp8_b16_l128_d192", 2048, 192, 4, 16, 128, true, false, true, false},
      {"silu_residual_b8_l128_d256", 1024, 256, 4, 8, 128, true, true, true, false},
  };
}

std::vector<CaseConfig> perf_suite() {
  return {
      {"perf_extend_t65536_d1536", 65536, 1536, 4, 64, 1024, false, false, true, false},
      {"perf_extend_t262144_d1536", 262144, 1536, 4, 128, 2048, false, false, true, false},
  };
}

bool parse_single_shape(std::string const& text, CaseConfig& cfg) {
  if (text.empty()) {
    return false;
  }

  cfg = {"custom", 0, 0, 4, 1, 1, false, false, true, false};
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
    } else if (key == "B") {
      cfg.batch = std::stoi(value);
    } else if (key == "L") {
      cfg.tokens_per_seq = std::stoi(value);
    } else if (key == "varied") {
      cfg.varied_lengths = std::stoi(value) != 0;
    } else if (key == "silu") {
      cfg.use_silu = std::stoi(value) != 0;
    } else if (key == "residual") {
      cfg.use_residual = std::stoi(value) != 0;
    } else if (key == "decode") {
      cfg.is_decode = std::stoi(value) != 0;
    } else {
      return false;
    }
  }

  if (cfg.is_decode) {
    cfg.T = cfg.batch;
    cfg.tokens_per_seq = 1;
  }

  return cfg.T > 0 && cfg.D > 0 && (cfg.W == 3 || cfg.W == 4) &&
      cfg.batch > 0 && cfg.tokens_per_seq > 0 &&
      (!cfg.is_decode || cfg.T == cfg.batch) &&
      (cfg.is_decode || cfg.T == cfg.batch * cfg.tokens_per_seq || cfg.varied_lengths);
}

struct Options {
  bool help = false;
  bool valid = true;
  bool verify = true;
  int iterations = 20;
  std::string suite = "inkling";
  std::string shape;
  std::string dtype_name = "bf16";
  DType dtype = DType::kBf16;
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
    cmd.get_cmd_line_argument("suite", suite, std::string("inkling"));
    cmd.get_cmd_line_argument("shape", shape, std::string(""));
    cmd.get_cmd_line_argument("dtype", dtype_name, std::string("bf16"));
    cmd.get_cmd_line_argument("target-tops", target_tops, 0.0);
    cmd.get_cmd_line_argument("target-gbps", target_gbps, 0.0);
    if (!parse_dtype(dtype_name, dtype)) {
      valid = false;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Inkling BMG SConv Example\n\n"
        << "Options:\n"
        << "  --help                         Print this message\n"
        << "  --suite=<quick|inkling|perf>    Built-in shape suite (default: inkling)\n"
        << "  --shape=<k=v,...>               Run one custom shape instead of a suite\n"
        << "                                  Keys: name,T,D,W,B,L,varied,silu,residual,decode\n"
        << "  --dtype=<bf16|fp16>             Input/output dtype (default: bf16)\n"
        << "  --iterations=<int>              Timed kernel iterations\n"
        << "  --verify=<0|1>                  Run CPU dtype reference comparison\n"
        << "  --target-tops=<float>           Fail if any timed case is below this useful TOPS\n"
        << "  --target-gbps=<float>           Fail if any timed case is below this effective GB/s\n\n"
        << "Examples:\n"
        << "  ./examples/14_bmg_sconv/14_bmg_sconv --suite=quick\n"
        << "  ./examples/14_bmg_sconv/14_bmg_sconv --suite=quick --dtype=fp16\n"
        << "  ./examples/14_bmg_sconv/14_bmg_sconv --suite=perf --verify=0 --iterations=100 --target-gbps=350\n"
        << "  ./examples/14_bmg_sconv/14_bmg_sconv --shape=T=8192,D=1536,W=4,B=8,L=1024,residual=1\n";
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
  } else if (options.suite == "inkling") {
    cases = inkling_suite();
  } else if (options.suite == "perf") {
    cases = perf_suite();
  } else {
    std::cerr << "Unknown suite: " << options.suite << "\n";
    options.print_usage(std::cerr);
    return 1;
  }

  try {
    sycl::queue q = compat::get_default_queue();
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Suite: " << (options.shape.empty() ? options.suite : "custom")
              << ", cases=" << cases.size()
              << ", dtype=" << dtype_text(options.dtype)
              << ", iterations=" << options.iterations
              << ", verify=" << bool_text(options.verify) << "\n";

    bool all_passed = true;
    for (auto const& cfg : cases) {
      if (options.dtype == DType::kBf16) {
        all_passed &= run_case<cutlass::bfloat16_t>(
            q, cfg, options.iterations, options.verify, options.target_tops, options.target_gbps);
      } else {
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
