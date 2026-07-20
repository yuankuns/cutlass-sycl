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
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Inkling extend/prefill attention prologue for CUTLASS SYCL on BMG.

    This standalone example mirrors the extend branch of Inkling's fused
    attention prologue:

      q_out = per-head RMSNorm(q)
      k_work/v_work = varlen causal depthwise short-conv using old cache prefixes
      k_out = per-head RMSNorm(round_to_dtype(k_work))
      v_out = round_to_dtype(v_work)
      k_buf/v_buf optionally receive the final K/V rows at loc[t]
      k_cache/v_cache optionally receive the trailing sequence-end conv windows

    The main kernel uses one work-group per token and one 16-byte vec8 lane per
    work-item. A second tiny kernel updates the convolution cache after the main
    kernel completes; doing it in the main kernel would race with early-token
    blocks that still need to read the old prefix cache.

    Roofline: for the common W=4 bf16/fp16 extend path, each K/V channel does
    about 8 useful convolution FLOPs plus modest norm/residual work while
    streaming qkvr, cache prefix, weights, outputs, optional KV stores, and
    cache-update rows. Arithmetic intensity is well below 1 FLOP/B for large
    prefill shapes, so this is memory-bound. Performance reporting emphasizes
    effective GB/s and the perf suite uses large working sets to avoid cache-hit
    microbenchmarks.
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

namespace cutlass::examples::attn_prologue_extend {

constexpr int kPadSlot = -1;
constexpr int kVecElems = 8;
constexpr int kHeadDim = 128;
constexpr int kHeadLanes = kHeadDim / kVecElems;
constexpr int kMaxLanes = 1024;
constexpr int kMinLocalSize = 32;
constexpr int kUpdateThreads = 256;
constexpr double kMinSustainedTargetBytes = 64.0 * 1024.0 * 1024.0;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

struct CaseConfig {
  std::string name;
  int batch = 1;
  int max_q = 1;
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
  bool do_store = true;
  bool do_cache_update = true;
  bool do_track = false;
  bool include_pad = false;
  bool include_mask_zero = false;
  bool include_negative_loc = false;
  bool include_zero_length = false;
  double target_gbps = 0.0;
};

template <typename Element_>
struct ExtendParams {
  using Element = Element_;

  Element const* __restrict__ qkvr;
  Element const* __restrict__ k_cache;
  Element const* __restrict__ v_cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ cache_mask;
  int64_t const* __restrict__ cu;
  int32_t const* __restrict__ si;
  Element const* __restrict__ k_weight;
  Element const* __restrict__ v_weight;
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
  int batch;
  int dq;
  int dkv;
  int qkvr_stride_t;
  int q_off;
  int k_off;
  int v_off;
  int cache_stride_slot;
  int cache_stride_w;
  int weight_stride_d;
  int kv_buf_stride;
};

template <typename Element_>
struct CacheUpdateParams {
  using Element = Element_;

  Element const* __restrict__ qkvr;
  Element* __restrict__ k_cache;
  Element* __restrict__ v_cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ has_init;
  int64_t const* __restrict__ cu;
  int64_t const* __restrict__ track_rows;
  uint8_t const* __restrict__ track_mask;
  int64_t const* __restrict__ track_dst;
  int qkvr_stride_t;
  int k_off;
  int v_off;
  int cache_stride_slot;
  int cache_stride_w;
  int track_dst_stride;
  int batch;
  int dkv;
};

template <typename Element_>
struct HostTensors {
  using Element = Element_;

  std::vector<Element> qkvr;
  std::vector<Element> k_cache;
  std::vector<Element> v_cache;
  std::vector<int32_t> cache_indices;
  std::vector<uint8_t> cache_mask;
  std::vector<uint8_t> has_init;
  std::vector<int64_t> cu;
  std::vector<int32_t> si;
  std::vector<Element> k_weight;
  std::vector<Element> v_weight;
  std::vector<int64_t> track_rows;
  std::vector<uint8_t> track_mask;
  std::vector<int64_t> track_dst;
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

template <typename Element>
CUTLASS_DEVICE
void store_zero_vec8(Element* dst) {
  *reinterpret_cast<uint64_t*>(dst) = 0;
  *reinterpret_cast<uint64_t*>(dst + 4) = 0;
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
void compute_kv_short_conv_vec(
    ExtendParams<Element> const& p,
    int t,
    int bos,
    int slot,
    float cache_gate,
    bool is_k,
    int ch,
    float (&out)[kVecElems]) {
  constexpr int W1 = W - 1;
  int x_off = is_k ? p.k_off : p.v_off;
  Element const* x_base = p.qkvr + static_cast<int64_t>(t) * p.qkvr_stride_t + x_off + ch;
  Element const* cache_base = (is_k ? p.k_cache : p.v_cache)
      + static_cast<int64_t>(slot) * p.cache_stride_slot + ch;
  Element const* weight = is_k ? p.k_weight : p.v_weight;

  float xcur[kVecElems];
  load_vec8(x_base, xcur);

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
      int shifted = t - W1 + iw;
      if (shifted >= bos) {
        load_vec8(p.qkvr + static_cast<int64_t>(shifted) * p.qkvr_stride_t + x_off + ch, tap);
      } else {
        int prefix_pos = shifted - bos + W1;
        if (prefix_pos >= 0) {
          load_vec8(cache_base + static_cast<int64_t>(prefix_pos) * p.cache_stride_w, tap);
#pragma unroll
          for (int j = 0; j < kVecElems; ++j) {
            tap[j] *= cache_gate;
          }
        } else {
#pragma unroll
          for (int j = 0; j < kVecElems; ++j) {
            tap[j] = 0.0f;
          }
        }
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

template <
    typename Element,
    int W,
    bool UseSilu,
    bool UseResidual,
    bool DoStore>
class AttnPrologueExtendKernel {
 public:
  ExtendParams<Element> params;

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
    if (lane >= total_lanes) {
      return;
    }

    int seq = params.si[t];
    int bos = static_cast<int>(params.cu[seq]);
    int ci = params.cache_indices[seq];
    bool valid = ci != kPadSlot;
    int slot = valid ? ci : 0;
    float cache_gate = (valid && params.cache_mask[seq] != 0) ? 1.0f : 0.0f;

    float values[kVecElems];
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      values[j] = 0.0f;
    }

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
      compute_kv_short_conv_vec<Element, W, UseSilu, UseResidual>(
          params, t, bos, slot, cache_gate, true, ch, values);
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
        if (kv_slot >= 0) {
          store_vec8(params.k_buf + kv_slot * params.kv_buf_stride + ch, values);
        }
      }
      return;
    }

    int ch = (lane - nq - nkv) * kVecElems;
    compute_kv_short_conv_vec<Element, W, UseSilu, UseResidual>(
        params, t, bos, slot, cache_gate, false, ch, values);
    store_vec8(params.v_out + static_cast<int64_t>(t) * params.dkv + ch, values);
    if constexpr (DoStore) {
      int64_t kv_slot = params.loc[t];
      if (kv_slot >= 0) {
        store_vec8(params.v_buf + kv_slot * params.kv_buf_stride + ch, values);
      }
    }
  }
};

template <typename Element, int W, bool DoTrack>
class KvConvCacheUpdateKernel {
 public:
  CacheUpdateParams<Element> params;

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    constexpr int W1 = W - 1;
    int nkv = params.dkv / kVecElems;
    int items = params.batch * 2 * nkv;
    int idx = static_cast<int>(item.get_global_linear_id());
    if (idx >= items) {
      return;
    }

    int b = idx / (2 * nkv);
    int role_vec = idx - b * 2 * nkv;
    bool is_k = role_vec < nkv;
    int ch = (is_k ? role_vec : role_vec - nkv) * kVecElems;
    Element const* base = params.qkvr;
    int x_off = is_k ? params.k_off : params.v_off;
    Element* cache = is_k ? params.k_cache : params.v_cache;
    int slot = params.cache_indices[b];
    int64_t qlen = params.cu[b + 1] - params.cu[b];

    if (slot != kPadSlot && qlen > 0) {
      Element* cache_base = cache + static_cast<int64_t>(slot) * params.cache_stride_slot + ch;
      uint64_t old_raw0[W1];
      uint64_t old_raw1[W1];
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        Element* src = cache_base + static_cast<int64_t>(w) * params.cache_stride_w;
        old_raw0[w] = *reinterpret_cast<uint64_t const*>(src);
        old_raw1[w] = *reinterpret_cast<uint64_t const*>(src + 4);
      }
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        Element* dst = cache_base + static_cast<int64_t>(w) * params.cache_stride_w;
        if (qlen >= W1 - w) {
          int64_t row = params.cu[b + 1] - W1 + w;
          copy_vec8_raw(base + row * params.qkvr_stride_t + x_off + ch, dst);
        } else if (params.has_init[b] != 0) {
          int src_w = w + static_cast<int>(qlen);
          *reinterpret_cast<uint64_t*>(dst) = old_raw0[src_w];
          *reinterpret_cast<uint64_t*>(dst + 4) = old_raw1[src_w];
        } else {
          store_zero_vec8(dst);
        }
      }
    }

    if constexpr (DoTrack) {
      if (params.track_mask[b] != 0) {
        int64_t dst_slot = params.track_dst[static_cast<int64_t>(b) * params.track_dst_stride];
        Element* dst_base = cache + dst_slot * params.cache_stride_slot + ch;
        for (int w = 0; w < W1; ++w) {
          int64_t row = params.track_rows[static_cast<int64_t>(b) * W1 + w];
          copy_vec8_raw(
              base + row * params.qkvr_stride_t + x_off + ch,
              dst_base + static_cast<int64_t>(w) * params.cache_stride_w);
        }
      }
    }
  }
};

template <
    typename Element,
    int W,
    bool UseSilu,
    bool UseResidual,
    bool DoStore>
sycl::event launch_extend_main_static(
    sycl::queue& q,
    ExtendParams<Element> const& params,
    int local_size) {
  int global = params.T * local_size;
  return q.submit([&](sycl::handler& cgh) {
    AttnPrologueExtendKernel<Element, W, UseSilu, UseResidual, DoStore> kernel{params};
    cgh.parallel_for<AttnPrologueExtendKernel<Element, W, UseSilu, UseResidual, DoStore>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local_size))),
        kernel);
  });
}

template <typename Element, int W, bool UseSilu, bool UseResidual>
sycl::event launch_store_selected(
    sycl::queue& q,
    ExtendParams<Element> const& params,
    int local_size,
    bool do_store) {
  if (do_store) {
    return launch_extend_main_static<Element, W, UseSilu, UseResidual, true>(q, params, local_size);
  }
  return launch_extend_main_static<Element, W, UseSilu, UseResidual, false>(q, params, local_size);
}

template <typename Element, int W, bool UseSilu>
sycl::event launch_residual_selected(
    sycl::queue& q,
    ExtendParams<Element> const& params,
    int local_size,
    bool use_residual,
    bool do_store) {
  if (use_residual) {
    return launch_store_selected<Element, W, UseSilu, true>(q, params, local_size, do_store);
  }
  return launch_store_selected<Element, W, UseSilu, false>(q, params, local_size, do_store);
}

template <typename Element, int W>
sycl::event launch_silu_selected(
    sycl::queue& q,
    ExtendParams<Element> const& params,
    int local_size,
    bool use_silu,
    bool use_residual,
    bool do_store) {
  if (use_silu) {
    return launch_residual_selected<Element, W, true>(q, params, local_size, use_residual, do_store);
  }
  return launch_residual_selected<Element, W, false>(q, params, local_size, use_residual, do_store);
}

template <typename Element>
sycl::event launch_extend_main(
    sycl::queue& q,
    ExtendParams<Element> const& params,
    int W,
    bool use_silu,
    bool use_residual,
    bool do_store) {
  int lanes = params.dq / kVecElems + 2 * (params.dkv / kVecElems);
  int local_size = std::max(kMinLocalSize, round_up(lanes, kMinLocalSize));
  if (W == 3) {
    return launch_silu_selected<Element, 3>(q, params, local_size, use_silu, use_residual, do_store);
  }
  if (W == 4) {
    return launch_silu_selected<Element, 4>(q, params, local_size, use_silu, use_residual, do_store);
  }
  throw std::invalid_argument("only W=3 and W=4 are supported");
}

template <typename Element, int W, bool DoTrack>
sycl::event launch_cache_update_static(
    sycl::queue& q,
    CacheUpdateParams<Element> const& params) {
  int nkv = params.dkv / kVecElems;
  int items = params.batch * 2 * nkv;
  int global = round_up(items, kUpdateThreads);
  return q.submit([&](sycl::handler& cgh) {
    KvConvCacheUpdateKernel<Element, W, DoTrack> kernel{params};
    cgh.parallel_for<KvConvCacheUpdateKernel<Element, W, DoTrack>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kUpdateThreads))),
        kernel);
  });
}

template <typename Element, int W>
sycl::event launch_track_selected(
    sycl::queue& q,
    CacheUpdateParams<Element> const& params,
    bool do_track) {
  if (do_track) {
    return launch_cache_update_static<Element, W, true>(q, params);
  }
  return launch_cache_update_static<Element, W, false>(q, params);
}

template <typename Element>
sycl::event launch_cache_update(
    sycl::queue& q,
    CacheUpdateParams<Element> const& params,
    int W,
    bool do_track) {
  if (W == 3) {
    return launch_track_selected<Element, 3>(q, params, do_track);
  }
  if (W == 4) {
    return launch_track_selected<Element, 4>(q, params, do_track);
  }
  throw std::invalid_argument("only W=3 and W=4 are supported");
}

struct LaunchEvents {
  sycl::event first;
  sycl::event last;
};

template <typename Element>
LaunchEvents launch_extend(
    sycl::queue& q,
    ExtendParams<Element> const& main_params,
    CacheUpdateParams<Element> const& update_params,
    int W,
    bool use_silu,
    bool use_residual,
    bool do_store,
    bool do_cache_update,
    bool do_track) {
  if (main_params.T == 0) {
    return {};
  }
  if (main_params.dq % kHeadDim != 0 || main_params.dkv % kHeadDim != 0) {
    throw std::invalid_argument("dq and dkv must be multiples of 128");
  }
  if (main_params.qkvr_stride_t % kVecElems != 0 ||
      main_params.q_off % kVecElems != 0 ||
      main_params.k_off % kVecElems != 0 ||
      main_params.v_off % kVecElems != 0 ||
      main_params.cache_stride_w % kVecElems != 0 ||
      main_params.kv_buf_stride % kVecElems != 0) {
    throw std::invalid_argument("all vectorized strides and offsets must be 8-element aligned");
  }
  int lanes = main_params.dq / kVecElems + 2 * (main_params.dkv / kVecElems);
  if (lanes > kMaxLanes) {
    throw std::invalid_argument("extend prologue lanes exceed 1024 work-items");
  }

  sycl::event main_event = launch_extend_main<Element>(
      q, main_params, W, use_silu, use_residual, do_store);
  if (!do_cache_update) {
    return {main_event, main_event};
  }
  sycl::event update_event = launch_cache_update<Element>(q, update_params, W, do_track);
  return {main_event, update_event};
}

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  if (cfg.batch <= 0 || cfg.max_q <= 0 || cfg.dq <= 0 || cfg.dkv <= 0) {
    throw std::invalid_argument("invalid non-positive shape");
  }
  if (cfg.dq % kHeadDim != 0 || cfg.dkv % kHeadDim != 0) {
    throw std::invalid_argument("dq and dkv must be multiples of 128");
  }
  if (cfg.W != 3 && cfg.W != 4) {
    throw std::invalid_argument("only W=3 and W=4 are supported");
  }

  HostTensors<Element> h;
  h.q_off = 0;
  h.k_off = round_up(cfg.dq + cfg.slice_gap, kVecElems);
  h.v_off = round_up(h.k_off + cfg.dkv + cfg.slice_gap, kVecElems);
  h.qkvr_stride_t = round_up(h.v_off + cfg.dkv + cfg.row_padding, kVecElems);
  h.cache_stride_w = round_up(cfg.dkv + cfg.cache_padding, kVecElems);
  h.cache_stride_slot = (cfg.W - 1) * h.cache_stride_w;
  h.weight_stride_d = cfg.W;
  h.kv_buf_stride = round_up(cfg.dkv + cfg.kv_padding, kVecElems);

  h.cu.resize(cfg.batch + 1);
  h.cu[0] = 0;
  for (int b = 0; b < cfg.batch; ++b) {
    int len = 1 + ((b * 7 + 3) % cfg.max_q);
    if (cfg.include_zero_length && (b % 9 == 4)) {
      len = 0;
    }
    h.cu[b + 1] = h.cu[b] + len;
    for (int i = 0; i < len; ++i) {
      h.si.push_back(b);
    }
  }
  h.T = static_cast<int>(h.si.size());
  if (h.T == 0) {
    throw std::invalid_argument("case generated zero tokens");
  }
  h.slots = std::max(1, cfg.batch + (cfg.do_track ? cfg.batch : 0) + cfg.extra_slots + 8);
  while (h.slots / std::gcd(h.slots, 5) < cfg.batch) {
    ++h.slots;
  }
  h.kv_slots = h.T + 16;

  h.qkvr.resize(static_cast<std::size_t>(h.T) * h.qkvr_stride_t);
  h.k_cache.resize(static_cast<std::size_t>(h.slots) * h.cache_stride_slot);
  h.v_cache.resize(static_cast<std::size_t>(h.slots) * h.cache_stride_slot);
  h.cache_indices.resize(cfg.batch);
  h.cache_mask.resize(cfg.batch);
  h.has_init.resize(cfg.batch);
  h.k_weight.resize(static_cast<std::size_t>(cfg.dkv) * h.weight_stride_d);
  h.v_weight.resize(static_cast<std::size_t>(cfg.dkv) * h.weight_stride_d);
  h.track_rows.resize(static_cast<std::size_t>(cfg.batch) * (cfg.W - 1));
  h.track_mask.resize(cfg.batch);
  h.track_dst.resize(cfg.batch);
  h.q_gamma.resize(kHeadDim);
  h.k_gamma.resize(kHeadDim);
  h.q_out.resize(static_cast<std::size_t>(h.T) * cfg.dq);
  h.k_out.resize(static_cast<std::size_t>(h.T) * cfg.dkv);
  h.v_out.resize(static_cast<std::size_t>(h.T) * cfg.dkv);
  h.loc.resize(h.T);
  h.k_buf.resize(static_cast<std::size_t>(h.kv_slots) * h.kv_buf_stride);
  h.v_buf.resize(static_cast<std::size_t>(h.kv_slots) * h.kv_buf_stride);

  std::mt19937 gen(20260719u + static_cast<unsigned>(cfg.batch * 37 + cfg.max_q * 11 + cfg.dq + cfg.dkv + cfg.W));
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

  for (int b = 0; b < cfg.batch; ++b) {
    bool pad = cfg.include_pad && (b % 7 == 3);
    bool init = !(cfg.include_mask_zero && (b % 5 == 2));
    h.cache_indices[b] = pad ? kPadSlot : ((b * 5 + 1) % h.slots);
    h.has_init[b] = static_cast<uint8_t>(init);
    h.cache_mask[b] = static_cast<uint8_t>((!pad && init) ? 1 : 0);
    h.track_mask[b] = 0;
  }

  std::vector<uint8_t> used_slots(h.slots, 0);
  for (int b = 0; b < cfg.batch; ++b) {
    if (h.cache_indices[b] != kPadSlot) {
      used_slots[h.cache_indices[b]] = 1;
    }
  }
  for (int b = 0; b < cfg.batch; ++b) {
    if (cfg.do_track) {
      int candidate = (h.slots - 1 - b) % h.slots;
      int probes = 0;
      while (used_slots[candidate] != 0 && probes < h.slots) {
        candidate = (candidate + h.slots - 1) % h.slots;
        ++probes;
      }
      if (probes == h.slots) {
        throw std::runtime_error("insufficient unique cache slots for track destinations");
      }
      h.track_dst[b] = candidate;
      used_slots[candidate] = 1;
    } else {
      h.track_dst[b] = 0;
    }

    int64_t qlen = h.cu[b + 1] - h.cu[b];
    bool valid = h.cache_indices[b] != kPadSlot;
    if (cfg.do_track && valid && qlen >= cfg.W - 1 && (b % 3 == 1)) {
      h.track_mask[b] = 1;
      for (int w = 0; w < cfg.W - 1; ++w) {
        h.track_rows[static_cast<std::size_t>(b) * (cfg.W - 1) + w] = h.cu[b + 1] - (cfg.W - 1) + w;
      }
    } else {
      for (int w = 0; w < cfg.W - 1; ++w) {
        h.track_rows[static_cast<std::size_t>(b) * (cfg.W - 1) + w] = std::max<int64_t>(0, h.cu[b]);
      }
    }
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
float host_load(std::vector<Element> const& storage, std::size_t idx) {
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
      float ss = 0.0f;
      for (int c = 0; c < kHeadDim; ++c) {
        int d = head * kHeadDim + c;
        float x = host_load(h.qkvr, static_cast<std::size_t>(t) * h.qkvr_stride_t + h.q_off + d);
        ss += x * x;
      }
      float inv = 1.0f / std::sqrt(ss / static_cast<float>(kHeadDim) + eps);
      for (int c = 0; c < kHeadDim; ++c) {
        int d = head * kHeadDim + c;
        float x = host_load(h.qkvr, static_cast<std::size_t>(t) * h.qkvr_stride_t + h.q_off + d);
        float gamma = to_float(h.q_gamma[c]);
        h.ref_q_out[static_cast<std::size_t>(t) * cfg.dq + d] = Element(x * inv * gamma);
      }
    }
  }

  for (int b = 0; b < cfg.batch; ++b) {
    int ci = h.cache_indices[b];
    bool valid = ci != kPadSlot;
    int slot = valid ? ci : 0;
    bool use_cache = valid && h.cache_mask[b] != 0;
    int64_t bos = h.cu[b];
    int64_t eos = h.cu[b + 1];
    for (int64_t t64 = bos; t64 < eos; ++t64) {
      int t = static_cast<int>(t64);
      for (int d = 0; d < cfg.dkv; ++d) {
        for (int role = 0; role < 2; ++role) {
          bool is_k = role == 0;
          auto const& cache = is_k ? h.k_cache : h.v_cache;
          auto const& weight = is_k ? h.k_weight : h.v_weight;
          int x_off = is_k ? h.k_off : h.v_off;
          float xcur = host_load(h.qkvr, static_cast<std::size_t>(t) * h.qkvr_stride_t + x_off + d);
          float acc = 0.0f;
          for (int iw = 0; iw < W; ++iw) {
            float tap = 0.0f;
            if (iw == W1) {
              tap = xcur;
            } else {
              int64_t shifted = t64 - W1 + iw;
              if (shifted >= bos) {
                tap = host_load(h.qkvr, static_cast<std::size_t>(shifted) * h.qkvr_stride_t + x_off + d);
              } else {
                int64_t prefix_pos = shifted - bos + W1;
                if (prefix_pos >= 0 && use_cache) {
                  tap = host_load(
                      cache,
                      static_cast<std::size_t>(slot) * h.cache_stride_slot
                          + static_cast<std::size_t>(prefix_pos) * h.cache_stride_w + d);
                }
              }
            }
            acc += tap * to_float(weight[static_cast<std::size_t>(d) * h.weight_stride_d + iw]);
          }
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
    }
  }

  for (int t = 0; t < h.T; ++t) {
    for (int head = 0; head < cfg.dkv / kHeadDim; ++head) {
      float ss = 0.0f;
      for (int c = 0; c < kHeadDim; ++c) {
        int d = head * kHeadDim + c;
        float x = k_work[static_cast<std::size_t>(t) * cfg.dkv + d];
        ss += x * x;
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

  if (!cfg.do_cache_update) {
    return;
  }

  auto update_one_cache = [&](std::vector<Element> const& old_cache, std::vector<Element>& ref_cache, bool is_k) {
    int x_off = is_k ? h.k_off : h.v_off;
    for (int b = 0; b < cfg.batch; ++b) {
      int slot = h.cache_indices[b];
      int64_t qlen = h.cu[b + 1] - h.cu[b];
      if (slot != kPadSlot && qlen > 0) {
        for (int w = 0; w < W1; ++w) {
          for (int d = 0; d < cfg.dkv; ++d) {
            std::size_t dst = static_cast<std::size_t>(slot) * h.cache_stride_slot
                + static_cast<std::size_t>(w) * h.cache_stride_w + d;
            if (qlen >= W1 - w) {
              int64_t row = h.cu[b + 1] - W1 + w;
              ref_cache[dst] = h.qkvr[static_cast<std::size_t>(row) * h.qkvr_stride_t + x_off + d];
            } else if (h.has_init[b] != 0) {
              int src_w = w + static_cast<int>(qlen);
              std::size_t src = static_cast<std::size_t>(slot) * h.cache_stride_slot
                  + static_cast<std::size_t>(src_w) * h.cache_stride_w + d;
              ref_cache[dst] = old_cache[src];
            } else {
              ref_cache[dst] = Element(0.0f);
            }
          }
        }
      }
      if (cfg.do_track && h.track_mask[b] != 0) {
        int64_t dst_slot = h.track_dst[b];
        for (int w = 0; w < W1; ++w) {
          int64_t row = h.track_rows[static_cast<std::size_t>(b) * W1 + w];
          for (int d = 0; d < cfg.dkv; ++d) {
            std::size_t dst = static_cast<std::size_t>(dst_slot) * h.cache_stride_slot
                + static_cast<std::size_t>(w) * h.cache_stride_w + d;
            ref_cache[dst] = h.qkvr[static_cast<std::size_t>(row) * h.qkvr_stride_t + x_off + d];
          }
        }
      }
    }
  };

  std::vector<Element> old_k_cache = h.k_cache;
  std::vector<Element> old_v_cache = h.v_cache;
  update_one_cache(old_k_cache, h.ref_k_cache, true);
  update_one_cache(old_v_cache, h.ref_v_cache, false);
}

template <typename Element>
ExtendParams<Element> make_main_params(HostTensors<Element> const& h, CaseConfig const& cfg) {
  ExtendParams<Element> params;
  params.qkvr = nullptr;
  params.k_cache = nullptr;
  params.v_cache = nullptr;
  params.cache_indices = nullptr;
  params.cache_mask = nullptr;
  params.cu = nullptr;
  params.si = nullptr;
  params.k_weight = nullptr;
  params.v_weight = nullptr;
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
  params.batch = cfg.batch;
  params.dq = cfg.dq;
  params.dkv = cfg.dkv;
  params.qkvr_stride_t = h.qkvr_stride_t;
  params.q_off = h.q_off;
  params.k_off = h.k_off;
  params.v_off = h.v_off;
  params.cache_stride_slot = h.cache_stride_slot;
  params.cache_stride_w = h.cache_stride_w;
  params.weight_stride_d = h.weight_stride_d;
  params.kv_buf_stride = h.kv_buf_stride;
  return params;
}

template <typename Element>
CacheUpdateParams<Element> make_update_params(HostTensors<Element> const& h, CaseConfig const& cfg) {
  CacheUpdateParams<Element> params;
  params.qkvr = nullptr;
  params.k_cache = nullptr;
  params.v_cache = nullptr;
  params.cache_indices = nullptr;
  params.has_init = nullptr;
  params.cu = nullptr;
  params.track_rows = nullptr;
  params.track_mask = nullptr;
  params.track_dst = nullptr;
  params.qkvr_stride_t = h.qkvr_stride_t;
  params.k_off = h.k_off;
  params.v_off = h.v_off;
  params.cache_stride_slot = h.cache_stride_slot;
  params.cache_stride_w = h.cache_stride_w;
  params.track_dst_stride = 1;
  params.batch = cfg.batch;
  params.dkv = cfg.dkv;
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

double estimate_bytes(CaseConfig const& cfg, int T) {
  double elem = 2.0;
  double q_norm = static_cast<double>(T) * cfg.dq * elem * 3.0;
  double kv_conv = static_cast<double>(T) * 2.0 * cfg.dkv * elem * static_cast<double>(cfg.W * 2);
  double kv_out = static_cast<double>(T) * 2.0 * cfg.dkv * elem;
  double gamma = static_cast<double>(T) * (cfg.dq + cfg.dkv) * elem;
  double store = cfg.do_store ? static_cast<double>(T) * 2.0 * cfg.dkv * elem : 0.0;
  double update = cfg.do_cache_update
      ? static_cast<double>(cfg.batch) * 2.0 * cfg.dkv * elem * static_cast<double>(cfg.W - 1) * 2.0
      : 0.0;
  return q_norm + kv_conv + kv_out + gamma + store + update;
}

double estimate_flops(CaseConfig const& cfg, int T) {
  double q_norm = static_cast<double>(T) * cfg.dq * 4.0;
  double kv_conv = static_cast<double>(T) * 2.0 * cfg.dkv
      * static_cast<double>(2 * cfg.W + (cfg.use_silu ? 4 : 0) + (cfg.use_residual ? 1 : 0));
  double k_norm = static_cast<double>(T) * cfg.dkv * 4.0;
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
  DeviceBuffer<uint8_t> d_has_init(q, h.has_init.size());
  DeviceBuffer<int64_t> d_cu(q, h.cu.size());
  DeviceBuffer<int32_t> d_si(q, h.si.size());
  DeviceBuffer<Element> d_k_weight(q, h.k_weight.size());
  DeviceBuffer<Element> d_v_weight(q, h.v_weight.size());
  DeviceBuffer<int64_t> d_track_rows(q, h.track_rows.size());
  DeviceBuffer<uint8_t> d_track_mask(q, h.track_mask.size());
  DeviceBuffer<int64_t> d_track_dst(q, h.track_dst.size());
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
  d_has_init.copy_from(h.has_init);
  d_cu.copy_from(h.cu);
  d_si.copy_from(h.si);
  d_k_weight.copy_from(h.k_weight);
  d_v_weight.copy_from(h.v_weight);
  d_track_rows.copy_from(h.track_rows);
  d_track_mask.copy_from(h.track_mask);
  d_track_dst.copy_from(h.track_dst);
  d_q_gamma.copy_from(h.q_gamma);
  d_k_gamma.copy_from(h.k_gamma);
  d_q_out.copy_from(h.q_out);
  d_k_out.copy_from(h.k_out);
  d_v_out.copy_from(h.v_out);
  d_loc.copy_from(h.loc);
  d_k_buf.copy_from(h.k_buf);
  d_v_buf.copy_from(h.v_buf);

  ExtendParams<Element> main_params = make_main_params(h, cfg);
  main_params.qkvr = d_qkvr.get();
  main_params.k_cache = d_k_cache.get();
  main_params.v_cache = d_v_cache.get();
  main_params.cache_indices = d_cache_indices.get();
  main_params.cache_mask = d_cache_mask.get();
  main_params.cu = d_cu.get();
  main_params.si = d_si.get();
  main_params.k_weight = d_k_weight.get();
  main_params.v_weight = d_v_weight.get();
  main_params.q_gamma = d_q_gamma.get();
  main_params.k_gamma = d_k_gamma.get();
  main_params.q_out = d_q_out.get();
  main_params.k_out = d_k_out.get();
  main_params.v_out = d_v_out.get();
  main_params.loc = d_loc.get();
  main_params.k_buf = d_k_buf.get();
  main_params.v_buf = d_v_buf.get();

  CacheUpdateParams<Element> update_params = make_update_params(h, cfg);
  update_params.qkvr = d_qkvr.get();
  update_params.k_cache = d_k_cache.get();
  update_params.v_cache = d_v_cache.get();
  update_params.cache_indices = d_cache_indices.get();
  update_params.has_init = d_has_init.get();
  update_params.cu = d_cu.get();
  update_params.track_rows = d_track_rows.get();
  update_params.track_mask = d_track_mask.get();
  update_params.track_dst = d_track_dst.get();

  auto launch = [&]() {
    return launch_extend<Element>(
        q,
        main_params,
        update_params,
        cfg.W,
        cfg.use_silu,
        cfg.use_residual,
        cfg.do_store,
        cfg.do_cache_update,
        cfg.do_track);
  };

  launch().last.wait_and_throw();

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
    launch().last.wait_and_throw();
  }
  std::vector<LaunchEvents> events;
  int timing_iterations = std::max(1, iterations);
  events.reserve(timing_iterations);
  for (int i = 0; i < timing_iterations; ++i) {
    events.push_back(launch());
  }
  double total_ns = 0.0;
  for (auto& event_pair : events) {
    event_pair.last.wait_and_throw();
    auto start = event_pair.first.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event_pair.last.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_ns += static_cast<double>(end - start);
  }
  double avg_s = total_ns * 1.0e-9 / static_cast<double>(timing_iterations);
  double bytes = estimate_bytes(cfg, h.T);
  double flops = estimate_flops(cfg, h.T);
  double gbps = bytes / avg_s / 1.0e9;
  double tops = flops / avg_s / 1.0e12;
  std::ostringstream target_suffix;
  if (target_gbps > 0.0 && bytes >= kMinSustainedTargetBytes) {
    target_suffix << " target=" << std::fixed << std::setprecision(2) << target_gbps << " GB/s";
  }
  bool perf_passed = target_gbps <= 0.0 || bytes < kMinSustainedTargetBytes || gbps >= target_gbps;
  passed = passed && perf_passed;

  std::cout << "  [" << element_dtype_text<Element>() << "] "
            << std::left << std::setw(30) << cfg.name << std::right
            << " B=" << cfg.batch
            << " T=" << h.T
            << " maxq=" << cfg.max_q
            << " dq=" << cfg.dq
            << " dkv=" << cfg.dkv
            << " W=" << cfg.W
            << " silu=" << bool_text(cfg.use_silu)
            << " residual=" << bool_text(cfg.use_residual)
            << " store=" << bool_text(cfg.do_store)
            << " update=" << bool_text(cfg.do_cache_update)
            << " track=" << bool_text(cfg.do_track)
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
      {"tiny_varlen_w3_no_residual", 3, 5, 128, 128, 3, 0, 0, 0, 0, 4, false, false, true, true, false, false, false, false, false},
      {"padded_mask_track_loc", 8, 6, 256, 128, 4, 8, 16, 8, 8, 5, false, true, true, true, true, true, true, true, true},
      {"silu_no_store_no_update", 5, 4, 128, 256, 4, 0, 0, 0, 0, 4, true, true, false, false, false, false, true, false, false},
      {"prod_like_varlen_small", 16, 17, 1024, 256, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false},
  };
}

// Inkling extend/prefill: W=4, head_dim=128, num_kv_heads=4. dq = 128 * num_heads/tp,
// dkv = 128 * max(1, 4/tp). Chunked prefill caps total tokens at max_prefill_tokens=16384;
// per-seq q ranges 1..max_q from the initializer's `1 + (b*7+3) % max_q` schedule.
// hidden_size=1536 → num_heads=12 (TP∈{1,2,4}); hidden_size=6144 → num_heads=48 (TP∈{1,2,4,8}).
std::vector<CaseConfig> inkling_suite() {
  return {
      // hidden_size=1536 (config defaults)
      {"extend_h1536_tp1_dq1536_dkv512",   16,  256, 1536, 512, 4, 0, 0, 0, 0, 8, false, true, true, true,  false, false, false, false, false},
      {"extend_h1536_tp2_dq768_dkv256",    32,  128,  768, 256, 4, 0, 0, 0, 0, 8, false, true, true, true,  true,  false, true,  false, false},
      {"extend_h1536_tp4_dq384_dkv128",    64,   96,  384, 128, 4, 0, 0, 0, 0, 8, false, true, true, true,  false, false, false, false, true },
      // hidden_size=6144 (production checkpoint)
      {"extend_h6144_tp1_dq6144_dkv512",    8,  256, 6144, 512, 4, 0, 0, 0, 0, 8, false, true, true, true,  false, false, false, false, false},
      {"extend_h6144_tp2_dq3072_dkv256",   16,  256, 3072, 256, 4, 0, 0, 0, 0, 8, false, true, true, true,  false, false, false, false, false},
      {"extend_h6144_tp4_dq1536_dkv128",   32,  256, 1536, 128, 4, 0, 0, 0, 0, 8, false, true, true, true,  false, false, false, false, false},
      {"extend_h6144_tp8_dq768_dkv128",    64,  128,  768, 128, 4, 0, 0, 0, 0, 8, false, true, true, true,  false, false, false, false, false},
      // Behavior variants at a real Inkling shape
      {"extend_h1536_no_update_v2",        16,  256, 1536, 512, 4, 0, 0, 0, 0, 8, false, true, true, false, false, false, false, false, false},
      {"extend_h1536_track",               24,  128, 1536, 512, 4, 0, 0, 0, 0, 8, false, true, true, true,  true,  true,  true,  true,  true },
      {"extend_h1536_silu_no_residual",    16,  128, 1536, 512, 4, 0, 0, 0, 0, 8, true,  false, true, true, false, false, false, false, false},
      {"extend_h1536_no_store_swa",        16,  128, 1536, 512, 4, 0, 0, 0, 0, 8, false, true, false, true, false, false, false, false, false},
      {"extend_h1536_W3",                  16,  128, 1536, 512, 3, 0, 0, 0, 0, 8, false, true, true, true,  false, false, false, false, false},
  };
}

// Perf-only sweep: batch*max_q kept near max_prefill_tokens=16384 to cover the real
// chunked prefill working set. Each case comfortably exceeds the sustained-throughput
// gate (kMinSustainedTargetBytes = 64 MB).
std::vector<CaseConfig> perf_suite() {
  return {
      {"perf_h1536_tp1_dq1536_dkv512_B64x256",     64,  256, 1536, 512, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false, 420.0},
      {"perf_h1536_tp2_dq768_dkv256_B128x128",    128,  128,  768, 256, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false, 500.0},
      {"perf_h1536_tp4_dq384_dkv128_B256x64",     256,   64,  384, 128, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false, 500.0},
      {"perf_h6144_tp1_dq6144_dkv512_B32x512",     32,  512, 6144, 512, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false, 360.0},
      {"perf_h6144_tp2_dq3072_dkv256_B64x256",     64,  256, 3072, 256, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false, 350.0},
      {"perf_h6144_tp4_dq1536_dkv128_B128x128",   128,  128, 1536, 128, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false, 420.0},
      {"perf_h6144_tp8_dq768_dkv128_B256x64",     256,   64,  768, 128, 4, 0, 0, 0, 0, 8, false, true, true, true, false, false, false, false, false, 430.0},
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
      if (key == "B") {
        cfg.batch = std::stoi(value);
      } else if (key == "maxq") {
        cfg.max_q = std::stoi(value);
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
      } else if (key == "store") {
        if (!parse_bool_value(value, cfg.do_store)) {
          return false;
        }
      } else if (key == "update") {
        if (!parse_bool_value(value, cfg.do_cache_update)) {
          return false;
        }
      } else if (key == "track") {
        if (!parse_bool_value(value, cfg.do_track)) {
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
      } else if (key == "zerolen") {
        if (!parse_bool_value(value, cfg.include_zero_length)) {
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

}  // namespace cutlass::examples::attn_prologue_extend

int main(int argc, char const** argv) {
  using namespace cutlass::examples::attn_prologue_extend;

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
          << "Inkling extend attention prologue example\n\n"
          << "Options:\n"
          << "  --suite=<quick|inkling|perf>    Built-in shape suite (default: quick)\n"
          << "  --shape=B=...,maxq=...,dq=...,dkv=...,W=...,silu=0,residual=1,store=1,update=1\n"
          << "                                  Single custom shape; overrides suite\n"
          << "  --dtype=<all|bf16|fp16>         Element dtype (default: all)\n"
          << "  --iterations=<int>              Timed kernel iterations\n"
          << "  --verify=<0|1>                  Run CPU reference comparison\n"
          << "  --target-gbps=<float>           Override sustained effective GB/s gate; 0 disables\n\n"
          << "Examples:\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_extend --suite=quick\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_extend --suite=inkling --dtype=bf16\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_extend --suite=perf --verify=0 --iterations=100\n";
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
