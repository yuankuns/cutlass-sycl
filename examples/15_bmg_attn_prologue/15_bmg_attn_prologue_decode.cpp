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
    \brief Inkling decode attention prologue for CUTLASS SYCL on BMG.

    This standalone example mirrors the decode branch of
    inkling_attn_prologue_decode:

      q_out = per-head RMSNorm(q)
      k_work/v_work = k/v decode short-conv using W-1 cached history taps
      k_out = per-head RMSNorm(round_to_dtype(k_work))
      v_out = round_to_dtype(v_work)
      k_sconv_cache/v_sconv_cache are shift-updated in place
      optional track slots receive the same updated cache window
      k_buf/v_buf optionally receive the final K/V rows at loc[t]

    The production CUDA path stores K/V short-conv weights as [oldest ... current] for
    each channel. This example follows that layout rather than the tiny Python
    pedagogical cases that index tap 0 as current.

    Roofline: for the common W=4 bf16 decode case, each K/V channel performs
    about 8 useful conv FLOPs plus a few norm/residual operations while reading
    qkvr/cache/weight/gamma values and writing output, updated conv cache, and
    optional KV rows. Arithmetic intensity is well under 1 FLOP/B for production
    decode batches, so this kernel is memory-bound. Performance reporting
    emphasizes effective GB/s; target gates should use sustained working sets
    large enough to avoid measuring cache-hit behavior.
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
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cutlass::examples::attn_prologue_decode {

constexpr int kPadSlot = -1;
constexpr int kVecElems = 8;
constexpr int kHeadDim = 128;
constexpr int kHeadLanes = kHeadDim / kVecElems;
constexpr int kMaxLanes = 1024;
constexpr int kMinLocalSize = 32;
constexpr double kMinSustainedTargetBytes = 32.0 * 1024.0 * 1024.0;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

struct CaseConfig {
  std::string name;
  int tokens = 1;
  int dq = kHeadDim;
  int dkv = kHeadDim;
  int W = 4;
  int slice_gap = 0;
  int row_padding = 0;
  int cache_padding = 0;
  int kv_padding = 0;
  int extra_slots = 8;
  bool use_silu = false;
  bool use_residual = true;
  bool do_track = false;
  bool do_store = true;
  bool include_pad = false;
  bool include_mask_zero = false;
  bool include_negative_loc = false;
  double target_gbps = 0.0;
};

template <typename Element_>
struct DecodeParams {
  using Element = Element_;

  Element const* __restrict__ qkvr;
  Element* __restrict__ k_cache;
  Element* __restrict__ v_cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ cache_mask;
  Element const* __restrict__ k_weight;
  Element const* __restrict__ v_weight;
  uint8_t const* __restrict__ track_mask;
  int64_t const* __restrict__ track_indices;
  Element const* __restrict__ q_gamma;
  Element const* __restrict__ k_gamma;
  Element* __restrict__ q_out;
  Element* __restrict__ k_out;
  Element* __restrict__ v_out;
  int64_t const* __restrict__ loc;
  Element* __restrict__ k_buf;
  Element* __restrict__ v_buf;
  float eps;
  int T;
  int dq;
  int dkv;
  int qkvr_stride_t;
  int q_off;
  int k_off;
  int v_off;
  int cache_stride_slot;
  int cache_stride_w;
  int weight_stride_d;
  int track_idx_stride;
  int kv_buf_stride;
};

template <typename Element_>
struct HostTensors {
  using Element = Element_;

  std::vector<Element> qkvr;
  std::vector<Element> k_cache;
  std::vector<Element> v_cache;
  std::vector<int32_t> cache_indices;
  std::vector<uint8_t> cache_mask;
  std::vector<Element> k_weight;
  std::vector<Element> v_weight;
  std::vector<uint8_t> track_mask;
  std::vector<int64_t> track_indices;
  std::vector<Element> q_gamma;
  std::vector<Element> k_gamma;
  std::vector<Element> q_out;
  std::vector<Element> k_out;
  std::vector<Element> v_out;
  std::vector<int64_t> loc;
  std::vector<Element> k_buf;
  std::vector<Element> v_buf;
  std::vector<Element> ref_q_out;
  std::vector<Element> ref_k_out;
  std::vector<Element> ref_v_out;
  std::vector<Element> ref_k_cache;
  std::vector<Element> ref_v_cache;
  std::vector<Element> ref_k_buf;
  std::vector<Element> ref_v_buf;
  int T = 0;
  int slots = 0;
  int kv_slots = 0;
  int qkvr_stride_t = 0;
  int q_off = 0;
  int k_off = 0;
  int v_off = 0;
  int cache_stride_w = 0;
  int cache_stride_slot = 0;
  int weight_stride_d = 0;
  int track_idx_stride = 1;
  int kv_buf_stride = 0;
};

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

template <typename Element>
CUTLASS_HOST_DEVICE
uint16_t raw_bits(Element x) {
  return x.raw();
}

template <typename Element>
CUTLASS_DEVICE
Element element_from_raw(uint64_t raw, int lane) {
  return Element::bitcast(static_cast<uint16_t>(raw >> (16 * lane)));
}

template <typename Element>
CUTLASS_DEVICE
void load_vec8(Element const* ptr, float (&values)[kVecElems]) {
  uint64_t raw0 = *reinterpret_cast<uint64_t const*>(ptr);
  uint64_t raw1 = *reinterpret_cast<uint64_t const*>(ptr + 4);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = to_float(element_from_raw<Element>(raw0, i));
    values[i + 4] = to_float(element_from_raw<Element>(raw1, i));
  }
}

template <typename Element>
CUTLASS_DEVICE
void store_vec8(Element* ptr, float const (&values)[kVecElems]) {
  uint64_t raw0 = 0;
  uint64_t raw1 = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    raw0 |= static_cast<uint64_t>(Element(values[i]).raw()) << (16 * i);
    raw1 |= static_cast<uint64_t>(Element(values[i + 4]).raw()) << (16 * i);
  }
  *reinterpret_cast<uint64_t*>(ptr) = raw0;
  *reinterpret_cast<uint64_t*>(ptr + 4) = raw1;
}

template <typename Element>
CUTLASS_DEVICE
void copy_vec8_raw(Element const* src, Element* dst) {
  uint64_t raw0 = *reinterpret_cast<uint64_t const*>(src);
  uint64_t raw1 = *reinterpret_cast<uint64_t const*>(src + 4);
  *reinterpret_cast<uint64_t*>(dst) = raw0;
  *reinterpret_cast<uint64_t*>(dst + 4) = raw1;
}

struct Vec8Raw {
  uint64_t lo = 0;
  uint64_t hi = 0;
};

template <typename Element>
CUTLASS_DEVICE
Vec8Raw load_vec8_raw(Element const* ptr) {
  return {
      *reinterpret_cast<uint64_t const*>(ptr),
      *reinterpret_cast<uint64_t const*>(ptr + 4)};
}

template <typename Element>
CUTLASS_DEVICE
void raw_to_float(Vec8Raw raw, float (&values)[kVecElems]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = to_float(element_from_raw<Element>(raw.lo, i));
    values[i + 4] = to_float(element_from_raw<Element>(raw.hi, i));
  }
}

template <typename Element>
CUTLASS_DEVICE
void store_vec8_raw(Element* ptr, Vec8Raw raw) {
  *reinterpret_cast<uint64_t*>(ptr) = raw.lo;
  *reinterpret_cast<uint64_t*>(ptr + 4) = raw.hi;
}

std::string bool_text(bool value) {
  return value ? "true" : "false";
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

int round_up(int value, int align) {
  return ((value + align - 1) / align) * align;
}

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

template <typename Element, int W, bool UseSilu, bool UseResidual>
CUTLASS_DEVICE
void compute_kv_decode_conv_vec(
    DecodeParams<Element> const& p,
    int t,
    int slot,
    float cache_gate,
    bool is_k,
    int ch,
    float (&out)[kVecElems],
    Vec8Raw (&history)[W - 1],
    Vec8Raw& xraw) {
  constexpr int W1 = W - 1;
  int x_off = is_k ? p.k_off : p.v_off;
  Element const* x_base = p.qkvr + static_cast<int64_t>(t) * p.qkvr_stride_t + x_off + ch;
  Element* cache_base = (is_k ? p.k_cache : p.v_cache)
      + static_cast<int64_t>(slot) * p.cache_stride_slot + ch;
  Element const* weight = is_k ? p.k_weight : p.v_weight;

  float xcur[kVecElems];
  xraw = load_vec8_raw(x_base);
  raw_to_float<Element>(xraw, xcur);

#pragma unroll
  for (int iw = 0; iw < W1; ++iw) {
    history[iw] = load_vec8_raw(cache_base + static_cast<int64_t>(iw) * p.cache_stride_w);
  }

#pragma unroll
  for (int j = 0; j < kVecElems; ++j) {
    out[j] = 0.0f;
  }

#pragma unroll
  for (int iw = 0; iw < W; ++iw) {
    float tap[kVecElems];
    if (iw == W1) {
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        tap[j] = xcur[j];
      }
    } else {
      raw_to_float<Element>(history[iw], tap);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        tap[j] *= cache_gate;
      }
    }
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      float w = to_float(weight[static_cast<int64_t>(ch + j) * p.weight_stride_d + iw]);
      out[j] += tap[j] * w;
    }
  }

  if constexpr (UseSilu) {
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      out[j] = out[j] / (1.0f + sycl::exp(-out[j]));
    }
  }
  if constexpr (UseResidual) {
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      out[j] += xcur[j];
    }
  }
}

template <typename Element, int W>
CUTLASS_DEVICE
void update_decode_cache_vec(
    DecodeParams<Element> const& p,
    int slot,
    bool do_track,
    int64_t track_slot,
    float cache_gate,
    bool is_k,
    int ch,
    Vec8Raw const (&history)[W - 1],
    Vec8Raw xraw) {
  constexpr int W1 = W - 1;
  Element* cache = is_k ? p.k_cache : p.v_cache;
  Element* cache_base = cache
      + static_cast<int64_t>(slot) * p.cache_stride_slot
      + ch;
  Element* track_base = cache
      + static_cast<int64_t>(track_slot) * p.cache_stride_slot
      + ch;
  Vec8Raw zero{};

#pragma unroll
  for (int w = 0; w < W1; ++w) {
    Vec8Raw next = (w < W1 - 1) ? ((cache_gate != 0.0f) ? history[w + 1] : zero) : xraw;
    store_vec8_raw(cache_base + static_cast<int64_t>(w) * p.cache_stride_w, next);
    if (do_track) {
      store_vec8_raw(track_base + static_cast<int64_t>(w) * p.cache_stride_w, next);
    }
  }
}

template <
    typename Element,
    int W,
    bool UseSilu,
    bool UseResidual,
    bool DoTrack,
    bool DoStore>
class AttnPrologueDecodeKernel {
 public:
  DecodeParams<Element> params;

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::properties{syclex::sub_group_size<16>};
  }

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int t = static_cast<int>(item.get_group(0));
    int lane = static_cast<int>(item.get_local_id(0));
    int nq = params.dq / kVecElems;
    int nkv = params.dkv / kVecElems;
    int total_lanes = nq + 2 * nkv;

    float values[kVecElems];
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      values[j] = 0.0f;
    }

    if (lane >= total_lanes) {
      return;
    }

    int ci = params.cache_indices[t];
    bool valid = ci != kPadSlot;
    int slot = valid ? ci : 0;
    float cache_gate = (valid && params.cache_mask[t] != 0) ? 1.0f : 0.0f;

    if (lane < nq) {
      int ch = lane * kVecElems;
      load_vec8(
          params.qkvr + static_cast<int64_t>(t) * params.qkvr_stride_t + params.q_off + ch,
          values);
      float ss = 0.0f;
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        ss += values[j] * values[j];
      }
      float head_ss = sycl::reduce_over_group(item.get_sub_group(), ss, sycl::plus<float>());
      float inv = sycl::rsqrt(head_ss / static_cast<float>(kHeadDim) + params.eps);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        int c = ch + j;
        float gamma = to_float(params.q_gamma[c % kHeadDim]);
        values[j] *= inv * gamma;
      }
      store_vec8(params.q_out + static_cast<int64_t>(t) * params.dq + ch, values);
      return;
    }

    if (lane < nq + nkv) {
      int ch = (lane - nq) * kVecElems;
      Vec8Raw history[W - 1];
      Vec8Raw xraw;
      compute_kv_decode_conv_vec<Element, W, UseSilu, UseResidual>(
          params, t, slot, cache_gate, true, ch, values, history, xraw);
      if (valid) {
        bool do_track = false;
        int64_t track_slot = 0;
        if constexpr (DoTrack) {
          do_track = params.track_mask[t] != 0;
          track_slot = params.track_indices[static_cast<int64_t>(t) * params.track_idx_stride];
        }
        update_decode_cache_vec<Element, W>(
            params, slot, do_track, track_slot, cache_gate, true, ch, history, xraw);
      }
      float ss = 0.0f;
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        values[j] = to_float(Element(values[j]));
        ss += values[j] * values[j];
      }
      float head_ss = sycl::reduce_over_group(item.get_sub_group(), ss, sycl::plus<float>());
      float inv = sycl::rsqrt(head_ss / static_cast<float>(kHeadDim) + params.eps);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        int c = ch + j;
        float gamma = to_float(params.k_gamma[c % kHeadDim]);
        values[j] *= inv * gamma;
      }
      store_vec8(params.k_out + static_cast<int64_t>(t) * params.dkv + ch, values);
      if constexpr (DoStore) {
        int64_t kv_slot = params.loc[t];
        if (valid && kv_slot >= 0) {
          store_vec8(params.k_buf + kv_slot * params.kv_buf_stride + ch, values);
        }
      }
      return;
    }

    int ch = (lane - nq - nkv) * kVecElems;
    Vec8Raw history[W - 1];
    Vec8Raw xraw;
    compute_kv_decode_conv_vec<Element, W, UseSilu, UseResidual>(
        params, t, slot, cache_gate, false, ch, values, history, xraw);
    if (valid) {
      bool do_track = false;
      int64_t track_slot = 0;
      if constexpr (DoTrack) {
        do_track = params.track_mask[t] != 0;
        track_slot = params.track_indices[static_cast<int64_t>(t) * params.track_idx_stride];
      }
      update_decode_cache_vec<Element, W>(
          params, slot, do_track, track_slot, cache_gate, false, ch, history, xraw);
    }
    store_vec8(params.v_out + static_cast<int64_t>(t) * params.dkv + ch, values);
    if constexpr (DoStore) {
      int64_t kv_slot = params.loc[t];
      if (valid && kv_slot >= 0) {
        store_vec8(params.v_buf + kv_slot * params.kv_buf_stride + ch, values);
      }
    }
  }
};

template <
    typename Element,
    int W,
    bool UseSilu,
    bool UseResidual,
    bool DoTrack,
    bool DoStore>
sycl::event launch_decode_static(sycl::queue& q, DecodeParams<Element> const& params, int local_size) {
  if (params.T == 0) {
    return sycl::event{};
  }
  int global = params.T * local_size;
  return q.submit([&](sycl::handler& cgh) {
    AttnPrologueDecodeKernel<Element, W, UseSilu, UseResidual, DoTrack, DoStore> kernel{params};
    cgh.parallel_for<AttnPrologueDecodeKernel<Element, W, UseSilu, UseResidual, DoTrack, DoStore>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local_size))),
        kernel);
  });
}

template <typename Element, int W, bool UseSilu, bool UseResidual>
sycl::event launch_store_selected(
    sycl::queue& q,
    DecodeParams<Element> const& params,
    int local_size,
    bool do_track,
    bool do_store) {
  if (do_track) {
    if (do_store) {
      return launch_decode_static<Element, W, UseSilu, UseResidual, true, true>(q, params, local_size);
    }
    return launch_decode_static<Element, W, UseSilu, UseResidual, true, false>(q, params, local_size);
  }
  if (do_store) {
    return launch_decode_static<Element, W, UseSilu, UseResidual, false, true>(q, params, local_size);
  }
  return launch_decode_static<Element, W, UseSilu, UseResidual, false, false>(q, params, local_size);
}

template <typename Element, int W, bool UseSilu>
sycl::event launch_residual_selected(
    sycl::queue& q,
    DecodeParams<Element> const& params,
    int local_size,
    bool use_residual,
    bool do_track,
    bool do_store) {
  if (use_residual) {
    return launch_store_selected<Element, W, UseSilu, true>(q, params, local_size, do_track, do_store);
  }
  return launch_store_selected<Element, W, UseSilu, false>(q, params, local_size, do_track, do_store);
}

template <typename Element, int W>
sycl::event launch_silu_selected(
    sycl::queue& q,
    DecodeParams<Element> const& params,
    int local_size,
    bool use_silu,
    bool use_residual,
    bool do_track,
    bool do_store) {
  if (use_silu) {
    return launch_residual_selected<Element, W, true>(q, params, local_size, use_residual, do_track, do_store);
  }
  return launch_residual_selected<Element, W, false>(q, params, local_size, use_residual, do_track, do_store);
}

template <typename Element>
sycl::event launch_decode(
    sycl::queue& q,
    DecodeParams<Element> const& params,
    int W,
    bool use_silu,
    bool use_residual,
    bool do_track,
    bool do_store) {
  int lanes = params.dq / kVecElems + 2 * (params.dkv / kVecElems);
  if (params.dq % kHeadDim != 0 || params.dkv % kHeadDim != 0) {
    throw std::invalid_argument("dq and dkv must be multiples of 128");
  }
  if (params.qkvr_stride_t % kVecElems != 0 ||
      params.q_off % kVecElems != 0 ||
      params.k_off % kVecElems != 0 ||
      params.v_off % kVecElems != 0 ||
      params.cache_stride_w % kVecElems != 0 ||
      params.kv_buf_stride % kVecElems != 0) {
    throw std::invalid_argument("all vectorized strides and offsets must be 8-element aligned");
  }
  int local_size = std::max(kMinLocalSize, round_up(lanes, kMinLocalSize));
  if (local_size > kMaxLanes) {
    throw std::invalid_argument("decode prologue lanes exceed 1024 work-items");
  }
  if (W == 3) {
    return launch_silu_selected<Element, 3>(q, params, local_size, use_silu, use_residual, do_track, do_store);
  }
  if (W == 4) {
    return launch_silu_selected<Element, 4>(q, params, local_size, use_silu, use_residual, do_track, do_store);
  }
  throw std::invalid_argument("only W=3 and W=4 are supported");
}

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  if (cfg.tokens <= 0 || cfg.dq <= 0 || cfg.dkv <= 0) {
    throw std::invalid_argument("invalid non-positive shape");
  }
  if (cfg.dq % kHeadDim != 0 || cfg.dkv % kHeadDim != 0) {
    throw std::invalid_argument("dq and dkv must be multiples of 128");
  }
  if (cfg.W != 3 && cfg.W != 4) {
    throw std::invalid_argument("only W=3 and W=4 are supported");
  }

  HostTensors<Element> h;
  h.T = cfg.tokens;
  h.slots = std::max(1, cfg.tokens * 2 + cfg.extra_slots);
  h.kv_slots = h.T + 16;
  h.q_off = 0;
  h.k_off = round_up(cfg.dq + cfg.slice_gap, kVecElems);
  h.v_off = round_up(h.k_off + cfg.dkv + cfg.slice_gap, kVecElems);
  h.qkvr_stride_t = round_up(h.v_off + cfg.dkv + cfg.row_padding, kVecElems);
  h.cache_stride_w = round_up(cfg.dkv + cfg.cache_padding, kVecElems);
  h.cache_stride_slot = (cfg.W - 1) * h.cache_stride_w;
  h.weight_stride_d = cfg.W;
  h.track_idx_stride = 1;
  h.kv_buf_stride = round_up(cfg.dkv + cfg.kv_padding, kVecElems);

  h.qkvr.resize(static_cast<std::size_t>(h.T) * h.qkvr_stride_t);
  h.k_cache.resize(static_cast<std::size_t>(h.slots) * h.cache_stride_slot);
  h.v_cache.resize(static_cast<std::size_t>(h.slots) * h.cache_stride_slot);
  h.cache_indices.resize(h.T);
  h.cache_mask.resize(h.T);
  h.k_weight.resize(static_cast<std::size_t>(cfg.dkv) * h.weight_stride_d);
  h.v_weight.resize(static_cast<std::size_t>(cfg.dkv) * h.weight_stride_d);
  h.track_mask.resize(h.T);
  h.track_indices.resize(h.T);
  h.q_gamma.resize(kHeadDim);
  h.k_gamma.resize(kHeadDim);
  h.q_out.resize(static_cast<std::size_t>(h.T) * cfg.dq);
  h.k_out.resize(static_cast<std::size_t>(h.T) * cfg.dkv);
  h.v_out.resize(static_cast<std::size_t>(h.T) * cfg.dkv);
  h.loc.resize(h.T);
  h.k_buf.resize(static_cast<std::size_t>(h.kv_slots) * h.kv_buf_stride);
  h.v_buf.resize(static_cast<std::size_t>(h.kv_slots) * h.kv_buf_stride);

  std::mt19937 gen(20260719u + static_cast<unsigned>(cfg.tokens * 17 + cfg.dq + cfg.dkv + cfg.W));
  std::uniform_real_distribution<float> x_dist(-0.60f, 0.60f);
  std::uniform_real_distribution<float> w_dist(-0.18f, 0.18f);
  std::uniform_real_distribution<float> c_dist(-0.12f, 0.12f);
  std::uniform_real_distribution<float> g_dist(0.80f, 1.20f);
  std::uniform_real_distribution<float> init_dist(-0.03f, 0.03f);

  for (auto& x : h.qkvr) {
    x = Element(x_dist(gen));
  }
  for (auto& x : h.k_cache) {
    x = Element(c_dist(gen));
  }
  for (auto& x : h.v_cache) {
    x = Element(c_dist(gen));
  }
  for (auto& x : h.k_weight) {
    x = Element(w_dist(gen));
  }
  for (auto& x : h.v_weight) {
    x = Element(w_dist(gen));
  }
  for (auto& x : h.q_gamma) {
    x = Element(g_dist(gen));
  }
  for (auto& x : h.k_gamma) {
    x = Element(g_dist(gen));
  }
  for (auto& x : h.q_out) {
    x = Element(0.0f);
  }
  for (auto& x : h.k_out) {
    x = Element(0.0f);
  }
  for (auto& x : h.v_out) {
    x = Element(0.0f);
  }
  for (auto& x : h.k_buf) {
    x = Element(init_dist(gen));
  }
  for (auto& x : h.v_buf) {
    x = Element(init_dist(gen));
  }

  for (int t = 0; t < h.T; ++t) {
    bool pad = cfg.include_pad && (t % 7 == 3);
    h.cache_indices[t] = pad ? kPadSlot : t % h.slots;
    h.cache_mask[t] = static_cast<uint8_t>(!(cfg.include_mask_zero && (t % 5 == 2)));
    h.track_mask[t] = static_cast<uint8_t>(cfg.do_track && !pad && (t % 3 != 1));
    h.track_indices[t] = static_cast<int64_t>((h.T + t) % h.slots);
  }
  for (int t = 0; t < h.T; ++t) {
    h.loc[t] = (cfg.include_negative_loc && (t % 11 == 5)) ? -1 : static_cast<int64_t>(t + 4);
  }

  h.ref_q_out = h.q_out;
  h.ref_k_out = h.k_out;
  h.ref_v_out = h.v_out;
  h.ref_k_cache = h.k_cache;
  h.ref_v_cache = h.v_cache;
  h.ref_k_buf = h.k_buf;
  h.ref_v_buf = h.v_buf;
  return h;
}

template <typename Element>
float host_load(HostTensors<Element> const& h, std::vector<Element> const& storage, std::size_t idx) {
  return to_float(storage[idx]);
}

template <typename Element, int W>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  constexpr int W1 = W - 1;
  float eps = 1.0e-5f;
  std::vector<float> k_work(static_cast<std::size_t>(h.T) * cfg.dkv, 0.0f);
  std::vector<float> v_work(static_cast<std::size_t>(h.T) * cfg.dkv, 0.0f);

  for (int t = 0; t < h.T; ++t) {
    for (int head = 0; head < cfg.dq / kHeadDim; ++head) {
      float partial[kHeadLanes];
#pragma unroll
      for (int lane = 0; lane < kHeadLanes; ++lane) {
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < kVecElems; ++j) {
          int c = head * kHeadDim + lane * kVecElems + j;
          float x = host_load(h, h.qkvr, static_cast<std::size_t>(t) * h.qkvr_stride_t + h.q_off + c);
          ss += x * x;
        }
        partial[lane] = ss;
      }
      float ss = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kHeadLanes; ++lane) {
        ss += partial[lane];
      }
      float inv = 1.0f / std::sqrt(ss / static_cast<float>(kHeadDim) + eps);
      for (int c = 0; c < kHeadDim; ++c) {
        int d = head * kHeadDim + c;
        float x = host_load(h, h.qkvr, static_cast<std::size_t>(t) * h.qkvr_stride_t + h.q_off + d);
        float gamma = to_float(h.q_gamma[c]);
        h.ref_q_out[static_cast<std::size_t>(t) * cfg.dq + d] = Element(x * inv * gamma);
      }
    }
  }

  for (int t = 0; t < h.T; ++t) {
    int ci = h.cache_indices[t];
    bool valid = ci != kPadSlot;
    int slot = valid ? ci : 0;
    bool use_cache = valid && h.cache_mask[t] != 0;
    bool do_track = cfg.do_track && valid && h.track_mask[t] != 0;
    int64_t track_slot = do_track ? h.track_indices[static_cast<std::size_t>(t) * h.track_idx_stride] : 0;

    for (int d = 0; d < cfg.dkv; ++d) {
      for (int role = 0; role < 2; ++role) {
        bool is_k = role == 0;
        auto const& cache = is_k ? h.k_cache : h.v_cache;
        auto const& weight = is_k ? h.k_weight : h.v_weight;
        int x_off = is_k ? h.k_off : h.v_off;
        float xcur = host_load(h, h.qkvr, static_cast<std::size_t>(t) * h.qkvr_stride_t + x_off + d);
        float acc = 0.0f;
        for (int iw = 0; iw < W1; ++iw) {
          float tap = use_cache
              ? host_load(
                    h,
                    cache,
                    static_cast<std::size_t>(slot) * h.cache_stride_slot
                        + static_cast<std::size_t>(iw) * h.cache_stride_w + d)
              : 0.0f;
          acc += tap * to_float(weight[static_cast<std::size_t>(d) * h.weight_stride_d + iw]);
        }
        acc += xcur * to_float(weight[static_cast<std::size_t>(d) * h.weight_stride_d + W1]);
        if (cfg.use_silu) {
          acc = silu(acc);
        }
        if (cfg.use_residual) {
          acc += xcur;
        }
        if (is_k) {
          k_work[static_cast<std::size_t>(t) * cfg.dkv + d] = to_float(Element(acc));
        } else {
          v_work[static_cast<std::size_t>(t) * cfg.dkv + d] = acc;
        }
      }
    }

    if (!valid) {
      continue;
    }
    for (int d = 0; d < cfg.dkv; ++d) {
      for (int role = 0; role < 2; ++role) {
        bool is_k = role == 0;
        auto const& src_cache = is_k ? h.k_cache : h.v_cache;
        auto& dst_cache = is_k ? h.ref_k_cache : h.ref_v_cache;
        int x_off = is_k ? h.k_off : h.v_off;
        for (int w = 0; w < W1; ++w) {
          Element next = Element(0.0f);
          if (w < W1 - 1) {
            if (use_cache) {
              next = src_cache[
                  static_cast<std::size_t>(slot) * h.cache_stride_slot
                  + static_cast<std::size_t>(w + 1) * h.cache_stride_w + d];
            }
          } else {
            next = h.qkvr[static_cast<std::size_t>(t) * h.qkvr_stride_t + x_off + d];
          }
          std::size_t main_dst = static_cast<std::size_t>(slot) * h.cache_stride_slot
              + static_cast<std::size_t>(w) * h.cache_stride_w + d;
          dst_cache[main_dst] = next;
          if (do_track) {
            std::size_t track_dst = static_cast<std::size_t>(track_slot) * h.cache_stride_slot
                + static_cast<std::size_t>(w) * h.cache_stride_w + d;
            dst_cache[track_dst] = next;
          }
        }
      }
    }
  }

  for (int t = 0; t < h.T; ++t) {
    for (int head = 0; head < cfg.dkv / kHeadDim; ++head) {
      float partial[kHeadLanes];
#pragma unroll
      for (int lane = 0; lane < kHeadLanes; ++lane) {
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < kVecElems; ++j) {
          int c = head * kHeadDim + lane * kVecElems + j;
          float x = k_work[static_cast<std::size_t>(t) * cfg.dkv + c];
          ss += x * x;
        }
        partial[lane] = ss;
      }
      float ss = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kHeadLanes; ++lane) {
        ss += partial[lane];
      }
      float inv = 1.0f / std::sqrt(ss / static_cast<float>(kHeadDim) + eps);
      for (int c = 0; c < kHeadDim; ++c) {
        int d = head * kHeadDim + c;
        float gamma = to_float(h.k_gamma[c]);
        h.ref_k_out[static_cast<std::size_t>(t) * cfg.dkv + d] =
            Element(k_work[static_cast<std::size_t>(t) * cfg.dkv + d] * inv * gamma);
      }
    }
    for (int d = 0; d < cfg.dkv; ++d) {
      h.ref_v_out[static_cast<std::size_t>(t) * cfg.dkv + d] =
          Element(v_work[static_cast<std::size_t>(t) * cfg.dkv + d]);
    }
  }

  if (cfg.do_store) {
    for (int t = 0; t < h.T; ++t) {
      if (h.cache_indices[t] == kPadSlot) {
        continue;
      }
      int64_t slot = h.loc[t];
      if (slot < 0) {
        continue;
      }
      for (int d = 0; d < cfg.dkv; ++d) {
        h.ref_k_buf[static_cast<std::size_t>(slot) * h.kv_buf_stride + d] =
            h.ref_k_out[static_cast<std::size_t>(t) * cfg.dkv + d];
        h.ref_v_buf[static_cast<std::size_t>(slot) * h.kv_buf_stride + d] =
            h.ref_v_out[static_cast<std::size_t>(t) * cfg.dkv + d];
      }
    }
  }
}

template <typename Element>
DecodeParams<Element> make_params(HostTensors<Element> const& h, CaseConfig const& cfg) {
  DecodeParams<Element> params;
  params.qkvr = nullptr;
  params.k_cache = nullptr;
  params.v_cache = nullptr;
  params.cache_indices = nullptr;
  params.cache_mask = nullptr;
  params.k_weight = nullptr;
  params.v_weight = nullptr;
  params.track_mask = nullptr;
  params.track_indices = nullptr;
  params.q_gamma = nullptr;
  params.k_gamma = nullptr;
  params.q_out = nullptr;
  params.k_out = nullptr;
  params.v_out = nullptr;
  params.loc = nullptr;
  params.k_buf = nullptr;
  params.v_buf = nullptr;
  params.eps = 1.0e-5f;
  params.T = h.T;
  params.dq = cfg.dq;
  params.dkv = cfg.dkv;
  params.qkvr_stride_t = h.qkvr_stride_t;
  params.q_off = h.q_off;
  params.k_off = h.k_off;
  params.v_off = h.v_off;
  params.cache_stride_slot = h.cache_stride_slot;
  params.cache_stride_w = h.cache_stride_w;
  params.weight_stride_d = h.weight_stride_d;
  params.track_idx_stride = h.track_idx_stride;
  params.kv_buf_stride = h.kv_buf_stride;
  return params;
}

struct VerifyResult {
  bool passed = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  int bad_index = -1;
};

template <typename Element>
VerifyResult compare_close(
    std::vector<Element> const& got,
    std::vector<Element> const& ref,
    double atol,
    double rtol) {
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double g = static_cast<double>(to_float(got[i]));
    double r = static_cast<double>(to_float(ref[i]));
    double abs_err = std::abs(g - r);
    double rel_err = abs_err / std::max(1.0e-12, std::abs(r));
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);
    if (abs_err > atol + rtol * std::abs(r) && result.passed) {
      result.passed = false;
      result.bad_index = static_cast<int>(i);
    }
  }
  return result;
}

template <typename Element>
VerifyResult compare_exact(std::vector<Element> const& got, std::vector<Element> const& ref) {
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double g = static_cast<double>(to_float(got[i]));
    double r = static_cast<double>(to_float(ref[i]));
    double abs_err = std::abs(g - r);
    result.max_abs = std::max(result.max_abs, abs_err);
    if (raw_bits(got[i]) != raw_bits(ref[i]) && result.passed) {
      result.passed = false;
      result.bad_index = static_cast<int>(i);
    }
  }
  return result;
}

void print_verify_result(std::string const& label, VerifyResult const& result) {
  std::cout << "    " << std::setw(8) << label
            << ": " << (result.passed ? "pass" : "FAIL")
            << " max_abs=" << result.max_abs
            << " max_rel=" << result.max_rel;
  if (!result.passed) {
    std::cout << " bad_index=" << result.bad_index;
  }
  std::cout << "\n";
}

double estimate_bytes(CaseConfig const& cfg) {
  double T = static_cast<double>(cfg.tokens);
  double elem = 2.0;
  double q_bytes = T * cfg.dq * elem * 3.0;
  double kv_conv_reads = T * 2.0 * cfg.dkv * elem * static_cast<double>((cfg.W - 1) + cfg.W);
  double kv_writes = T * 2.0 * cfg.dkv * elem;
  double gamma_reads = T * (cfg.dq + cfg.dkv) * elem;
  double cache_update = T * 2.0 * cfg.dkv * elem * static_cast<double>(cfg.W - 1);
  double track_update = cfg.do_track ? cache_update : 0.0;
  double store = cfg.do_store ? T * 2.0 * cfg.dkv * elem : 0.0;
  return q_bytes + kv_conv_reads + kv_writes + gamma_reads + cache_update + track_update + store;
}

double estimate_flops(CaseConfig const& cfg) {
  double T = static_cast<double>(cfg.tokens);
  double q_norm = T * cfg.dq * 4.0;
  double kv_conv = T * 2.0 * cfg.dkv * static_cast<double>(2 * cfg.W + (cfg.use_silu ? 4 : 0) + (cfg.use_residual ? 1 : 0));
  double k_norm = T * cfg.dkv * 4.0;
  return q_norm + kv_conv + k_norm;
}

template <typename Element>
bool run_case(
    sycl::queue& q,
    CaseConfig const& cfg,
    int iterations,
    bool verify,
    double target_gbps) {
  HostTensors<Element> h = initialize_case<Element>(cfg);
  if (verify) {
    if (cfg.W == 3) {
      reference_case<Element, 3>(cfg, h);
    } else if (cfg.W == 4) {
      reference_case<Element, 4>(cfg, h);
    } else {
      throw std::invalid_argument("unsupported W");
    }
  }

  DeviceBuffer<Element> d_qkvr(q, h.qkvr.size());
  DeviceBuffer<Element> d_k_cache(q, h.k_cache.size());
  DeviceBuffer<Element> d_v_cache(q, h.v_cache.size());
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<uint8_t> d_cache_mask(q, h.cache_mask.size());
  DeviceBuffer<Element> d_k_weight(q, h.k_weight.size());
  DeviceBuffer<Element> d_v_weight(q, h.v_weight.size());
  DeviceBuffer<uint8_t> d_track_mask(q, h.track_mask.size());
  DeviceBuffer<int64_t> d_track_indices(q, h.track_indices.size());
  DeviceBuffer<Element> d_q_gamma(q, h.q_gamma.size());
  DeviceBuffer<Element> d_k_gamma(q, h.k_gamma.size());
  DeviceBuffer<Element> d_q_out(q, h.q_out.size());
  DeviceBuffer<Element> d_k_out(q, h.k_out.size());
  DeviceBuffer<Element> d_v_out(q, h.v_out.size());
  DeviceBuffer<int64_t> d_loc(q, h.loc.size());
  DeviceBuffer<Element> d_k_buf(q, h.k_buf.size());
  DeviceBuffer<Element> d_v_buf(q, h.v_buf.size());

  d_qkvr.copy_from(h.qkvr);
  d_k_cache.copy_from(h.k_cache);
  d_v_cache.copy_from(h.v_cache);
  d_cache_indices.copy_from(h.cache_indices);
  d_cache_mask.copy_from(h.cache_mask);
  d_k_weight.copy_from(h.k_weight);
  d_v_weight.copy_from(h.v_weight);
  d_track_mask.copy_from(h.track_mask);
  d_track_indices.copy_from(h.track_indices);
  d_q_gamma.copy_from(h.q_gamma);
  d_k_gamma.copy_from(h.k_gamma);
  d_q_out.copy_from(h.q_out);
  d_k_out.copy_from(h.k_out);
  d_v_out.copy_from(h.v_out);
  d_loc.copy_from(h.loc);
  d_k_buf.copy_from(h.k_buf);
  d_v_buf.copy_from(h.v_buf);

  DecodeParams<Element> params = make_params(h, cfg);
  params.qkvr = d_qkvr.get();
  params.k_cache = d_k_cache.get();
  params.v_cache = d_v_cache.get();
  params.cache_indices = d_cache_indices.get();
  params.cache_mask = d_cache_mask.get();
  params.k_weight = d_k_weight.get();
  params.v_weight = d_v_weight.get();
  params.track_mask = d_track_mask.get();
  params.track_indices = d_track_indices.get();
  params.q_gamma = d_q_gamma.get();
  params.k_gamma = d_k_gamma.get();
  params.q_out = d_q_out.get();
  params.k_out = d_k_out.get();
  params.v_out = d_v_out.get();
  params.loc = d_loc.get();
  params.k_buf = d_k_buf.get();
  params.v_buf = d_v_buf.get();

  auto launch = [&]() {
    return launch_decode<Element>(
        q, params, cfg.W, cfg.use_silu, cfg.use_residual, cfg.do_track, cfg.do_store);
  };

  launch().wait_and_throw();

  bool passed = true;
  if (verify) {
    d_q_out.copy_to(h.q_out);
    d_k_out.copy_to(h.k_out);
    d_v_out.copy_to(h.v_out);
    d_k_cache.copy_to(h.k_cache);
    d_v_cache.copy_to(h.v_cache);
    d_k_buf.copy_to(h.k_buf);
    d_v_buf.copy_to(h.v_buf);

    double atol = std::is_same_v<Element, cutlass::bfloat16_t> ? 3.5e-2 : 4.0e-3;
    double rtol = std::is_same_v<Element, cutlass::bfloat16_t> ? 3.5e-2 : 4.0e-3;
    VerifyResult q_result = compare_close(h.q_out, h.ref_q_out, atol, rtol);
    VerifyResult k_result = compare_close(h.k_out, h.ref_k_out, atol, rtol);
    VerifyResult v_result = compare_close(h.v_out, h.ref_v_out, atol, rtol);
    VerifyResult kc_result = compare_exact(h.k_cache, h.ref_k_cache);
    VerifyResult vc_result = compare_exact(h.v_cache, h.ref_v_cache);
    VerifyResult kb_result = compare_close(h.k_buf, h.ref_k_buf, atol, rtol);
    VerifyResult vb_result = compare_close(h.v_buf, h.ref_v_buf, atol, rtol);
    passed = q_result.passed && k_result.passed && v_result.passed &&
        kc_result.passed && vc_result.passed && kb_result.passed && vb_result.passed;
    if (!passed) {
      print_verify_result("q", q_result);
      print_verify_result("k", k_result);
      print_verify_result("v", v_result);
      print_verify_result("k_cache", kc_result);
      print_verify_result("v_cache", vc_result);
      print_verify_result("k_buf", kb_result);
      print_verify_result("v_buf", vb_result);
    }
  }

  int warmup_iterations = std::min(10, std::max(2, iterations));
  for (int i = 0; i < warmup_iterations; ++i) {
    launch().wait_and_throw();
  }
  std::vector<sycl::event> events;
  int timing_iterations = std::max(1, iterations);
  events.reserve(timing_iterations);
  for (int i = 0; i < timing_iterations; ++i) {
    events.push_back(launch());
  }
  double total_ns = 0.0;
  for (auto& event : events) {
    event.wait_and_throw();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_ns += static_cast<double>(end - start);
  }
  double avg_s = total_ns * 1.0e-9 / static_cast<double>(timing_iterations);
  double bytes = estimate_bytes(cfg);
  double flops = estimate_flops(cfg);
  double gbps = bytes / avg_s / 1.0e9;
  double tops = flops / avg_s / 1.0e12;
  std::ostringstream target_suffix;
  if (target_gbps > 0.0 && bytes >= kMinSustainedTargetBytes) {
    target_suffix << " target=" << std::fixed << std::setprecision(2) << target_gbps << " GB/s";
  }
  bool perf_passed = target_gbps <= 0.0 || bytes < kMinSustainedTargetBytes || gbps >= target_gbps;
  passed = passed && perf_passed;

  std::cout << "  [" << element_dtype_text<Element>() << "] "
            << std::left << std::setw(28) << cfg.name << std::right
            << " T=" << cfg.tokens
            << " dq=" << cfg.dq
            << " dkv=" << cfg.dkv
            << " W=" << cfg.W
            << " silu=" << bool_text(cfg.use_silu)
            << " residual=" << bool_text(cfg.use_residual)
            << " track=" << bool_text(cfg.do_track)
            << " store=" << bool_text(cfg.do_store)
            << "  " << std::fixed << std::setprecision(3)
            << (avg_s * 1.0e6) << " us"
            << "  " << std::setprecision(2) << gbps << " GB/s"
            << target_suffix.str()
            << "  " << std::setprecision(3) << tops << " TOPS"
            << "  " << (verify ? (passed ? "passed" : "FAILED") : (perf_passed ? "verification skipped" : "FAILED"))
            << "\n";
  if (!perf_passed) {
    std::cerr << "    performance target failed: " << gbps
              << " GB/s < " << target_gbps << " GB/s\n";
  }
  return passed;
}

std::vector<CaseConfig> quick_suite() {
  return {
      {"tiny_w3_no_residual", 2, 128, 128, 3, 0, 0, 0, 0, 8, false, false, false, true, false, false, false},
      {"padded_track_mask_loc", 17, 256, 128, 4, 8, 16, 8, 8, 40, false, true, true, true, true, true, true},
      {"silu_track_no_store", 20, 128, 256, 4, 0, 0, 0, 0, 48, true, true, true, false, false, true, false},
      {"prod_t128_dq1024_dkv256", 128, 1024, 256, 4, 0, 0, 0, 0, 32, false, true, false, true, false, false, false},
  };
}

// Inkling decode: one token per active sequence (`tokens` = decode batch).
// W = sconv_kernel_size = 4, head_dim = 128.
//   dq  = head_dim * (num_attention_heads / P)
//   dkv = head_dim * max(1, num_key_value_heads / P)
// The max(1, ...) is the real replication rule: once P > num_kv_heads the K/V
// heads are REPLICATED across ranks, not split, so dkv floors at head_dim
// (InklingAttention.num_tp_kv_heads and InklingConfig.tp_local_kv_conv_dim in
// sglang). SWA (local) layers use swa_num_key_value_heads instead.
// Geometries:
//   config defaults  hidden_size=1536, Nq=12, Nkv=4          (TP∈{1,2,4})
//   production       hidden_size=6144, Nq=48, Nkv=4          (TP∈{1,2,4,8})
//   shipped ckpt     hidden_size=768,  Nq=8,  Nkv=2, swa 4   (TP∈{1,2,4,8})
// The checkpoint geometry is thinkingmachines/Inkling config.json text_config:
// hidden_size=768, num_attention_heads=8, head_dim=128, num_key_value_heads=2,
// swa_num_key_value_heads=4. head_dim is set EXPLICITLY there, so it is not
// hidden_size/Nq (InklingConfig only falls back to that when head_dim is None):
// dq = 8 * 128 = 1024 at TP=1, wider than hidden_size.
// Batches sized to expose all decode-cache traffic.
std::vector<CaseConfig> inkling_suite() {
  return {
      // hidden_size=1536 (config defaults)
      {"decode_h1536_tp1_dq1536_dkv512",  256, 1536, 512, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_h1536_tp2_dq768_dkv256",   256,  768, 256, 4, 0, 0, 0, 0, 32, false, true, true,  true,  true,  true,  true },
      {"decode_h1536_tp4_dq384_dkv128",   512,  384, 128, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      // hidden_size=6144 (production checkpoint)
      {"decode_h6144_tp1_dq6144_dkv512",  128, 6144, 512, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_h6144_tp2_dq3072_dkv256",  256, 3072, 256, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_h6144_tp4_dq1536_dkv128",  512, 1536, 128, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_h6144_tp8_dq768_dkv128",  1024,  768, 128, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      // shipped checkpoint (hidden_size=768, Nq=8, head_dim=128): full layers
      // Nkv=2, SWA layers swa_num_key_value_heads=4. At P>=4 the KV heads are
      // replicated, so full and SWA collapse onto the same dkv=128.
      {"decode_ckpt_tp1_dq1024_dkv256",   256, 1024, 256, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_ckpt_tp1_swa_dq1024_dkv512", 256, 1024, 512, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_ckpt_tp2_dq512_dkv128",    512,  512, 128, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_ckpt_tp2_swa_dq512_dkv256", 512,  512, 256, 4, 0, 0, 0, 0, 32, false, true, true,  true,  true,  true,  true },
      {"decode_ckpt_tp4_dq256_dkv128",    512,  256, 128, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      {"decode_ckpt_tp8_dq128_dkv128",   1024,  128, 128, 4, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
      // Behavior variants at a real Inkling shape (flagship h=1536 TP=1)
      {"decode_h1536_track",              128, 1536, 512, 4, 0, 0, 0, 0, 32, false, true, true,  true,  false, false, false},
      {"decode_h1536_no_store_swa",       128, 1536, 512, 4, 0, 0, 0, 0, 32, false, true, false, false, false, false, false},
      {"decode_h1536_silu_no_residual",   128, 1536, 512, 4, 0, 0, 0, 0, 32, true,  false, false, true, false, false, false},
      {"decode_h1536_W3",                 128, 1536, 512, 3, 0, 0, 0, 0, 32, false, true, false, true,  false, false, false},
  };
}

// Perf-only sweep. Each case pushes the working set well beyond the sustained
// throughput gate (kMinSustainedTargetBytes = 32 MB) at real TP-driven dq/dkv;
// scaling `tokens` avoids cache-hit microbenchmark behavior.
std::vector<CaseConfig> perf_suite() {
  return {
      {"perf_h1536_tp1_dq1536_dkv512_t2048", 2048, 1536, 512, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 445.0},
      {"perf_h1536_tp2_dq768_dkv256_t4096",  4096,  768, 256, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 455.0},
      {"perf_h1536_tp4_dq384_dkv128_t8192",  8192,  384, 128, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 450.0},
      {"perf_h6144_tp1_dq6144_dkv512_t1024", 1024, 6144, 512, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 340.0},
      {"perf_h6144_tp2_dq3072_dkv256_t2048", 2048, 3072, 256, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 390.0},
      {"perf_h6144_tp4_dq1536_dkv128_t4096", 4096, 1536, 128, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 435.0},
      {"perf_h6144_tp8_dq768_dkv128_t8192",  8192,  768, 128, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 445.0},
      // Shipped-checkpoint geometry (hidden_size=768, Nq=8, Nkv=2, swa Nkv=4).
      // target_gbps=0.0: report-only, these gates are not calibrated yet (the
      // measuring GPU was shared, so a number here would flake CI).
      {"perf_ckpt_tp1_dq1024_dkv256_t4096",   4096, 1024, 256, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 0.0},
      {"perf_ckpt_tp1_swa_dq1024_dkv512_t4096", 4096, 1024, 512, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 0.0},
      {"perf_ckpt_tp2_dq512_dkv128_t8192",    8192,  512, 128, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 0.0},
      {"perf_ckpt_tp2_swa_dq512_dkv256_t8192", 8192,  512, 256, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 0.0},
      {"perf_ckpt_tp4_dq256_dkv128_t8192",    8192,  256, 128, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 0.0},
      {"perf_ckpt_tp8_dq128_dkv128_t8192",    8192,  128, 128, 4, 0, 0, 0, 0, 64, false, true, false, true, false, false, false, 0.0},
  };
}

bool parse_bool_value(std::string const& text, bool& value) {
  if (text == "1" || text == "true" || text == "True") {
    value = true;
    return true;
  }
  if (text == "0" || text == "false" || text == "False") {
    value = false;
    return true;
  }
  return false;
}

bool parse_single_shape(std::string const& text, CaseConfig& cfg) {
  cfg = CaseConfig{};
  cfg.name = "custom";
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto pos = item.find('=');
    if (pos == std::string::npos) {
      return false;
    }
    std::string key = item.substr(0, pos);
    std::string value = item.substr(pos + 1);
    try {
      if (key == "T" || key == "tokens" || key == "B") {
        cfg.tokens = std::stoi(value);
      } else if (key == "dq") {
        cfg.dq = std::stoi(value);
      } else if (key == "dkv") {
        cfg.dkv = std::stoi(value);
      } else if (key == "W") {
        cfg.W = std::stoi(value);
      } else if (key == "gap") {
        cfg.slice_gap = std::stoi(value);
      } else if (key == "rowpad") {
        cfg.row_padding = std::stoi(value);
      } else if (key == "cachepad") {
        cfg.cache_padding = std::stoi(value);
      } else if (key == "kvpad") {
        cfg.kv_padding = std::stoi(value);
      } else if (key == "silu") {
        if (!parse_bool_value(value, cfg.use_silu)) {
          return false;
        }
      } else if (key == "residual") {
        if (!parse_bool_value(value, cfg.use_residual)) {
          return false;
        }
      } else if (key == "track") {
        if (!parse_bool_value(value, cfg.do_track)) {
          return false;
        }
      } else if (key == "store") {
        if (!parse_bool_value(value, cfg.do_store)) {
          return false;
        }
      } else if (key == "pad") {
        if (!parse_bool_value(value, cfg.include_pad)) {
          return false;
        }
      } else if (key == "maskzero") {
        if (!parse_bool_value(value, cfg.include_mask_zero)) {
          return false;
        }
      } else if (key == "negloc") {
        if (!parse_bool_value(value, cfg.include_negative_loc)) {
          return false;
        }
      } else if (key == "target" || key == "target_gbps" || key == "target-gbps") {
        cfg.target_gbps = std::stod(value);
      } else {
        return false;
      }
    } catch (...) {
      return false;
    }
  }
  return true;
}

struct Options {
  std::string suite = "quick";
  std::string shape;
  DType dtype = DType::kAll;
  bool verify = true;
  int iterations = 20;
  double target_gbps = 0.0;
  bool target_gbps_set = false;
};

}  // namespace cutlass::examples::attn_prologue_decode

int main(int argc, char const** argv) {
  using namespace cutlass::examples::attn_prologue_decode;

  Options options;
  try {
    cutlass::CommandLine cmd(argc, argv);
    cmd.get_cmd_line_argument("suite", options.suite, std::string("quick"));
    cmd.get_cmd_line_argument("shape", options.shape, std::string(""));
    std::string dtype_text_arg = "all";
    cmd.get_cmd_line_argument("dtype", dtype_text_arg, std::string("all"));
    if (!parse_dtype(dtype_text_arg, options.dtype)) {
      std::cerr << "Invalid --dtype value: " << dtype_text_arg << "\n";
      return -1;
    }
    int verify_int = 1;
    cmd.get_cmd_line_argument("verify", verify_int, 1);
    options.verify = verify_int != 0;
    cmd.get_cmd_line_argument("iterations", options.iterations, 20);
    options.target_gbps_set = cmd.check_cmd_line_flag("target-gbps");
    cmd.get_cmd_line_argument("target-gbps", options.target_gbps, 0.0);

    if (cmd.check_cmd_line_flag("help")) {
      std::cout
          << "Inkling decode attention prologue example\n\n"
          << "Options:\n"
          << "  --suite=<quick|inkling|perf>    Built-in shape suite (default: quick)\n"
          << "  --shape=T=...,dq=...,dkv=...,W=...,silu=0,residual=1,track=0,store=1\n"
          << "                                  Single custom shape; overrides suite\n"
          << "  --dtype=<all|bf16|fp16>         Element dtype (default: all)\n"
          << "  --iterations=<int>              Timed kernel iterations\n"
          << "  --verify=<0|1>                  Run CPU reference comparison\n"
          << "  --target-gbps=<float>           Override sustained effective GB/s gate; 0 disables\n\n"
          << "Examples:\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_decode --suite=quick\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_decode --suite=inkling --dtype=bf16\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_decode --suite=perf --verify=0 --iterations=100\n";
      return 0;
    }
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return -1;
  }

  std::vector<CaseConfig> cases;
  if (!options.shape.empty()) {
    CaseConfig cfg;
    if (!parse_single_shape(options.shape, cfg)) {
      std::cerr << "Invalid --shape string: " << options.shape << "\n";
      return -1;
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
    return -1;
  }

  try {
    sycl::queue q(
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Suite=" << options.suite
              << " dtype=" << dtype_text(options.dtype)
              << " iterations=" << options.iterations
              << " verify=" << bool_text(options.verify) << "\n";

    bool all_passed = true;
    for (auto const& cfg : cases) {
      double target_gbps = options.target_gbps_set ? options.target_gbps : cfg.target_gbps;
      if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
        all_passed &= run_case<cutlass::bfloat16_t>(
            q, cfg, options.iterations, options.verify, target_gbps);
      }
      if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
        all_passed &= run_case<cutlass::half_t>(
            q, cfg, options.iterations, options.verify, target_gbps);
      }
    }
    return all_passed ? 0 : -1;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return -1;
  }
}
