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
    \brief Inkling MXFP8 KV-store and log-tau prologue for CUTLASS SYCL on BMG.

    This standalone example implements the 02.4 attention-prologue semantics:

      q_norm = per-head RMSNorm(q)
      q_norm is rounded to the input dtype, optionally multiplied by per-token tau,
      rounded again, then quantized to MXFP8 E4M3 with one E8M0 scale per 32 channels
      k_out = per-head RMSNorm(k), rounded to the input dtype
      v_out = rounded v
      k_buf/v_buf optionally receive MXFP8 K/V rows at loc[t] with the FA4
      interleaved scale layout [pages, Hkv, 32, page_size / 32, 4]

    BMG does not provide native FP8 arithmetic here. The kernel stores software
    E4M3 bytes and E8M0 scale bytes, while all arithmetic is fp32 with bf16/fp16
    round points matching the fused CUDA prologue contract. This is an FP8 store
    kernel, not an FP8 attention matmul.

    Roofline: per 32-channel block, Q/K RMSNorm performs a sum-of-squares pass,
    a scale/gamma pass, optional tau, one block amax, software E4M3 packing, and
    contiguous stores. Even with the software conversion counted as arithmetic,
    production shapes such as T=4096, dq=3072, dkv=512 stay far below the BMG
    compute roofline balance (~50 TOPS / 350 GB/s ~= 143 op/B). The kernel is
    bandwidth-oriented; performance is reported as effective GB/s with large
    working-set perf cases to avoid cache-hit-only measurements.
*/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/util/command_line.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

namespace cutlass::examples::attn_prologue_mxfp8 {

constexpr int kVecElems = 8;
constexpr int kMXFP8Block = 32;
constexpr int kHeadDim = 128;
constexpr int kBlocksPerHead = kHeadDim / kMXFP8Block;
constexpr int kMaxLanes = 1024;
constexpr int kMinLocalSize = 32;
constexpr float kE4M3Max = 448.0f;
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
  int page_size = 128;
  int row_padding = 0;
  int kv_padding = 0;
  int extra_slots = 16;
  bool use_tau = true;
  bool do_store = true;
  bool include_negative_loc = false;
  bool stress_values = false;
  double target_gbps = 0.0;
};

template <typename Element_>
struct Mxfp8StoreTauParams {
  using Element = Element_;

  Element const* __restrict__ q;
  Element const* __restrict__ k;
  Element const* __restrict__ v;
  Element const* __restrict__ q_gamma;
  Element const* __restrict__ k_gamma;
  float const* __restrict__ tau;
  int64_t const* __restrict__ loc;
  Element* __restrict__ k_out;
  Element* __restrict__ v_out;
  uint8_t* __restrict__ q_mxfp8;
  uint8_t* __restrict__ k_buf_mxfp8;
  uint8_t* __restrict__ v_buf_mxfp8;
  uint8_t* __restrict__ sfq;
  uint8_t* __restrict__ sfk;
  uint8_t* __restrict__ sfv;
  float eps;
  int T;
  int dq;
  int dkv;
  int q_stride;
  int k_stride;
  int v_stride;
  int k_out_stride;
  int v_out_stride;
  int q_mxfp8_stride;
  int kv_buf_stride;
  int page_size;
};

template <typename Element_>
struct HostTensors {
  using Element = Element_;

  std::vector<Element> q;
  std::vector<Element> k;
  std::vector<Element> v;
  std::vector<Element> q_gamma;
  std::vector<Element> k_gamma;
  std::vector<float> tau;
  std::vector<int64_t> loc;
  std::vector<Element> k_out;
  std::vector<Element> v_out;
  std::vector<uint8_t> q_mxfp8;
  std::vector<uint8_t> k_buf_mxfp8;
  std::vector<uint8_t> v_buf_mxfp8;
  std::vector<uint8_t> sfq;
  std::vector<uint8_t> sfk;
  std::vector<uint8_t> sfv;
  std::vector<Element> ref_k_out;
  std::vector<Element> ref_v_out;
  std::vector<uint8_t> ref_q_mxfp8;
  std::vector<uint8_t> ref_k_buf_mxfp8;
  std::vector<uint8_t> ref_v_buf_mxfp8;
  std::vector<uint8_t> ref_sfq;
  std::vector<uint8_t> ref_sfk;
  std::vector<uint8_t> ref_sfv;
  int q_stride = 0;
  int k_stride = 0;
  int v_stride = 0;
  int q_mxfp8_stride = 0;
  int kv_buf_stride = 0;
  int kv_slots = 0;
  int pages = 0;
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
void load_block32(Element const* ptr, float (&values)[kMXFP8Block]) {
#pragma unroll
  for (int chunk = 0; chunk < kMXFP8Block / kVecElems; ++chunk) {
    float tmp[kVecElems];
    load_vec8(ptr + chunk * kVecElems, tmp);
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      values[chunk * kVecElems + j] = tmp[j];
    }
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
void store_block32(Element* ptr, float const (&values)[kMXFP8Block]) {
#pragma unroll
  for (int chunk = 0; chunk < kMXFP8Block / kVecElems; ++chunk) {
    float tmp[kVecElems];
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      tmp[j] = values[chunk * kVecElems + j];
    }
    store_vec8(ptr + chunk * kVecElems, tmp);
  }
}

CUTLASS_HOST_DEVICE
float hd_fabs(float x) {
  return x < 0.0f ? -x : x;
}

CUTLASS_HOST_DEVICE
float hd_floor(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::floor(x);
#else
  return std::floor(x);
#endif
}

CUTLASS_HOST_DEVICE
float hd_ceil(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::ceil(x);
#else
  return std::ceil(x);
#endif
}

CUTLASS_HOST_DEVICE
float hd_sqrt(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::sqrt(x);
#else
  return std::sqrt(x);
#endif
}

CUTLASS_HOST_DEVICE
float hd_log2(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::log2(x);
#else
  return std::log2(x);
#endif
}

CUTLASS_HOST_DEVICE
float hd_exp2(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::exp2(x);
#else
  return std::exp2(x);
#endif
}

CUTLASS_HOST_DEVICE
uint32_t float_bits(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::bit_cast<uint32_t>(x);
#else
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
#endif
}

CUTLASS_HOST_DEVICE
int round_even_positive(float x) {
  float floor_x = hd_floor(x);
  int base = static_cast<int>(floor_x);
  float frac = x - floor_x;
  if (frac > 0.5f || (frac == 0.5f && (base & 1))) {
    return base + 1;
  }
  return base;
}

CUTLASS_HOST_DEVICE
uint8_t float_to_e4m3fn_byte(float x) {
  if (!(x == x)) {
    return 0x7fu;
  }
  uint8_t sign = x < 0.0f ? 0x80u : 0u;
  float ax = hd_fabs(x);
  if (ax == 0.0f) {
    return sign;
  }
  if (ax >= kE4M3Max) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  constexpr float kMinNormal = 0x1p-6f;
  if (ax < kMinNormal) {
    int mant = round_even_positive(ax * 512.0f);
    if (mant <= 0) {
      return sign;
    }
    if (mant >= 8) {
      return static_cast<uint8_t>(sign | 0x08u);
    }
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(mant));
  }

  uint32_t bits = float_bits(ax);
  int e = static_cast<int>((bits >> 23) & 0xffu) - 127;
  if (e > 8) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  uint32_t frac = bits & 0x007fffffu;
  int mant = static_cast<int>(frac >> 20);
  uint32_t rem = frac & ((1u << 20) - 1u);
  uint32_t halfway = 1u << 19;
  if (rem > halfway || (rem == halfway && (mant & 1))) {
    ++mant;
  }
  if (mant == 8) {
    mant = 0;
    ++e;
  }
  if (e > 8) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  int exp_field = e + 7;
  if (exp_field >= 15 && mant > 6) {
    exp_field = 15;
    mant = 6;
  }
  return static_cast<uint8_t>(sign | static_cast<uint8_t>((exp_field << 3) | mant));
}

float e4m3fn_byte_to_float(uint8_t byte) {
  if ((byte & 0x7fu) == 0) {
    return (byte & 0x80u) ? -0.0f : 0.0f;
  }
  float sign = (byte & 0x80u) ? -1.0f : 1.0f;
  int exp_field = (byte >> 3) & 0x0f;
  int mant = byte & 0x07;
  if (exp_field == 0) {
    return sign * static_cast<float>(mant) * 0x1p-9f;
  }
  if (exp_field == 15 && mant == 7) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float significand = 1.0f + static_cast<float>(mant) * 0.125f;
  return sign * std::ldexp(significand, exp_field - 7);
}

float e8m0_scale_to_float(uint8_t byte) {
  return std::ldexp(1.0f, static_cast<int>(byte) - 127);
}

CUTLASS_HOST_DEVICE
uint8_t mxfp8_scale_byte(float amax, float& descale) {
  float safe_amax = amax > 1.0e-30f ? amax : 1.0e-30f;
  float biased = hd_ceil(hd_log2(safe_amax / kE4M3Max)) + 127.0f;
  if (biased < 0.0f) {
    biased = 0.0f;
  }
  if (biased > 254.0f) {
    biased = 254.0f;
  }
  int byte = static_cast<int>(biased);
  descale = hd_exp2(static_cast<float>(byte - 127));
  return static_cast<uint8_t>(byte);
}

CUTLASS_HOST_DEVICE
int64_t kv_scale_offset(int64_t kv_slot, int ch, int dkv, int page_size) {
  int hkv = dkv / kHeadDim;
  int page_chunks = page_size / kMXFP8Block;
  int sf_dim = kHeadDim / kMXFP8Block;
  int64_t page = kv_slot / page_size;
  int64_t po = kv_slot % page_size;
  int head = ch / kHeadDim;
  int block = (ch % kHeadDim) / kMXFP8Block;
  return ((page * hkv + head) * (kMXFP8Block * page_chunks * sf_dim))
      + ((po % kMXFP8Block) * (page_chunks * sf_dim))
      + ((po / kMXFP8Block) * sf_dim)
      + block;
}

CUTLASS_HOST_DEVICE
void store_mxfp8_block(float const (&values)[kMXFP8Block], uint8_t* dst, uint8_t* sf) {
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kMXFP8Block; ++j) {
    float ax = hd_fabs(values[j]);
    amax = ax > amax ? ax : amax;
  }
  float descale = 1.0f;
  *sf = mxfp8_scale_byte(amax, descale);

#pragma unroll
  for (int chunk = 0; chunk < kMXFP8Block / kVecElems; ++chunk) {
    uint64_t raw = 0;
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      float scaled = values[chunk * kVecElems + j] / descale;
      if (scaled > kE4M3Max) {
        scaled = kE4M3Max;
      }
      if (scaled < -kE4M3Max) {
        scaled = -kE4M3Max;
      }
      raw |= static_cast<uint64_t>(float_to_e4m3fn_byte(scaled)) << (8 * j);
    }
    *reinterpret_cast<uint64_t*>(dst + chunk * kVecElems) = raw;
  }
}

CUTLASS_DEVICE
void store_mxfp8_vec8(float const (&values)[kVecElems], uint8_t* dst, float descale) {
  uint64_t raw = 0;
#pragma unroll
  for (int j = 0; j < kVecElems; ++j) {
    float scaled = values[j] / descale;
    if (scaled > kE4M3Max) {
      scaled = kE4M3Max;
    }
    if (scaled < -kE4M3Max) {
      scaled = -kE4M3Max;
    }
    raw |= static_cast<uint64_t>(float_to_e4m3fn_byte(scaled)) << (8 * j);
  }
  *reinterpret_cast<uint64_t*>(dst) = raw;
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

template <typename Element, bool UseTau, bool DoStore>
class Mxfp8StoreTauKernel {
 public:
  Mxfp8StoreTauParams<Element> params;
  sycl::local_accessor<float, 1> scratch;

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int t = static_cast<int>(item.get_group(0));
    int lane = static_cast<int>(item.get_local_id(0));
    int q_lanes = params.dq / kVecElems;
    int kv_lanes = params.dkv / kVecElems;
    int total_lanes = q_lanes + 2 * kv_lanes;
    bool active = lane < total_lanes;
    bool is_q = active && lane < q_lanes;
    bool is_k = active && lane >= q_lanes && lane < q_lanes + kv_lanes;
    bool is_v = active && lane >= q_lanes + kv_lanes;
    int role_lane = is_q ? lane : (is_k ? lane - q_lanes : lane - q_lanes - kv_lanes);
    int ch = role_lane * kVecElems;

    float values[kVecElems];
#pragma unroll
    for (int j = 0; j < kVecElems; ++j) {
      values[j] = 0.0f;
    }

    float ss = 0.0f;
    if (is_q) {
      load_vec8(params.q + static_cast<int64_t>(t) * params.q_stride + ch, values);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        ss += values[j] * values[j];
      }
    } else if (is_k) {
      load_vec8(params.k + static_cast<int64_t>(t) * params.k_stride + ch, values);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        ss += values[j] * values[j];
      }
    } else if (is_v) {
      load_vec8(params.v + static_cast<int64_t>(t) * params.v_stride + ch, values);
    }
    scratch[lane] = ss;
    item.barrier(sycl::access::fence_space::local_space);

    if ((is_q || is_k) && ((role_lane & (kHeadDim / kVecElems - 1)) == 0)) {
      float head_ss = 0.0f;
      int group_start = lane;
#pragma unroll
      for (int i = 0; i < kHeadDim / kVecElems; ++i) {
        head_ss += scratch[group_start + i];
      }
      scratch[group_start] = 1.0f / sycl::sqrt(head_ss / static_cast<float>(kHeadDim) + params.eps);
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (is_q) {
      int group_start = lane - (role_lane & (kHeadDim / kVecElems - 1));
      float inv = scratch[group_start];
      float tau = 1.0f;
      if constexpr (UseTau) {
        tau = params.tau[t];
      }
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        float gamma = to_float(params.q_gamma[(ch + j) % kHeadDim]);
        Element rounded = Element(values[j] * inv * gamma);
        if constexpr (UseTau) {
          rounded = Element(to_float(rounded) * tau);
        }
        values[j] = to_float(rounded);
      }
    } else if (is_k) {
      int group_start = lane - (role_lane & (kHeadDim / kVecElems - 1));
      float inv = scratch[group_start];
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        float gamma = to_float(params.k_gamma[(ch + j) % kHeadDim]);
        Element rounded = Element(values[j] * inv * gamma);
        values[j] = to_float(rounded);
      }
      store_vec8(params.k_out + static_cast<int64_t>(t) * params.k_out_stride + ch, values);
    } else if (is_v) {
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        values[j] = to_float(Element(values[j]));
      }
      store_vec8(params.v_out + static_cast<int64_t>(t) * params.v_out_stride + ch, values);
    }

    float local_amax = 0.0f;
    if (is_q || (DoStore && (is_k || is_v))) {
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        float ax = values[j] < 0.0f ? -values[j] : values[j];
        local_amax = ax > local_amax ? ax : local_amax;
      }
    }
    scratch[lane] = local_amax;
    item.barrier(sycl::access::fence_space::local_space);

    bool scale_leader = active && ((role_lane & (kMXFP8Block / kVecElems - 1)) == 0);
    if (scale_leader && (is_q || (DoStore && (is_k || is_v)))) {
      int group_start = lane;
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kMXFP8Block / kVecElems; ++i) {
        float candidate = scratch[group_start + i];
        amax = candidate > amax ? candidate : amax;
      }
      float descale = 1.0f;
      uint8_t scale_byte = mxfp8_scale_byte(amax, descale);
      scratch[group_start] = descale;
      if (is_q) {
        int q_blocks = params.dq / kMXFP8Block;
        params.sfq[t * q_blocks + role_lane / (kMXFP8Block / kVecElems)] = scale_byte;
      } else if constexpr (DoStore) {
        int64_t kv_slot = params.loc[t];
        if (kv_slot >= 0) {
          int64_t sf_off = kv_scale_offset(kv_slot, ch, params.dkv, params.page_size);
          uint8_t* sf = is_k ? params.sfk : params.sfv;
          sf[sf_off] = scale_byte;
        }
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (is_q) {
      int group_start = lane - (role_lane & (kMXFP8Block / kVecElems - 1));
      float descale = scratch[group_start];
      store_mxfp8_vec8(values, params.q_mxfp8 + static_cast<int64_t>(t) * params.q_mxfp8_stride + ch, descale);
    } else if constexpr (DoStore) {
      if (is_k || is_v) {
        int64_t kv_slot = params.loc[t];
        if (kv_slot >= 0) {
          int group_start = lane - (role_lane & (kMXFP8Block / kVecElems - 1));
          float descale = scratch[group_start];
          uint8_t* buf = is_k ? params.k_buf_mxfp8 : params.v_buf_mxfp8;
          store_mxfp8_vec8(values, buf + kv_slot * params.kv_buf_stride + ch, descale);
        }
      }
    }
  }
};

template <typename Element, bool UseTau, bool DoStore>
sycl::event launch_mxfp8_store_tau_static(
    sycl::queue& q,
    Mxfp8StoreTauParams<Element> const& params,
    int local_size) {
  if (params.T == 0) {
    return sycl::event{};
  }
  int global = params.T * local_size;
  return q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(static_cast<std::size_t>(local_size)), cgh);
    Mxfp8StoreTauKernel<Element, UseTau, DoStore> kernel{params, scratch};
    cgh.parallel_for<Mxfp8StoreTauKernel<Element, UseTau, DoStore>>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(local_size))),
        kernel);
  });
}

template <typename Element, bool UseTau>
sycl::event launch_store_selected(
    sycl::queue& q,
    Mxfp8StoreTauParams<Element> const& params,
    int local_size,
    bool do_store) {
  if (do_store) {
    return launch_mxfp8_store_tau_static<Element, UseTau, true>(q, params, local_size);
  }
  return launch_mxfp8_store_tau_static<Element, UseTau, false>(q, params, local_size);
}

template <typename Element>
sycl::event launch_mxfp8_store_tau(
    sycl::queue& q,
    Mxfp8StoreTauParams<Element> const& params,
    bool use_tau,
    bool do_store) {
  if (params.dq % kHeadDim != 0 || params.dkv % kHeadDim != 0) {
    throw std::invalid_argument("dq and dkv must be multiples of 128");
  }
  if (params.page_size <= 0 || params.page_size % kMXFP8Block != 0) {
    throw std::invalid_argument("page_size must be a positive multiple of 32");
  }
  if (params.q_stride % kVecElems != 0 || params.k_stride % kVecElems != 0 ||
      params.v_stride % kVecElems != 0 || params.k_out_stride % kVecElems != 0 ||
      params.v_out_stride % kVecElems != 0 || params.q_mxfp8_stride % kMXFP8Block != 0 ||
      params.kv_buf_stride % kMXFP8Block != 0) {
    throw std::invalid_argument("vectorized strides must be aligned");
  }
  int lanes = params.dq / kVecElems + 2 * (params.dkv / kVecElems);
  int local_size = std::max(kMinLocalSize, round_up(lanes, kMinLocalSize));
  if (local_size > kMaxLanes) {
    throw std::invalid_argument("MXFP8 store/tau lanes exceed 1024 work-items");
  }
  if (use_tau) {
    return launch_store_selected<Element, true>(q, params, local_size, do_store);
  }
  return launch_store_selected<Element, false>(q, params, local_size, do_store);
}

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  if (cfg.tokens < 0 || cfg.dq <= 0 || cfg.dkv <= 0) {
    throw std::invalid_argument("invalid non-positive shape");
  }
  if (cfg.dq % kHeadDim != 0 || cfg.dkv % kHeadDim != 0) {
    throw std::invalid_argument("dq and dkv must be multiples of 128");
  }
  if (cfg.page_size <= 0 || cfg.page_size % kMXFP8Block != 0) {
    throw std::invalid_argument("page_size must be a positive multiple of 32");
  }

  HostTensors<Element> h;
  h.q_stride = round_up(cfg.dq + cfg.row_padding, kVecElems);
  h.k_stride = round_up(cfg.dkv + cfg.row_padding, kVecElems);
  h.v_stride = round_up(cfg.dkv + cfg.row_padding, kVecElems);
  h.q_mxfp8_stride = round_up(cfg.dq, kMXFP8Block);
  h.kv_buf_stride = round_up(cfg.dkv + cfg.kv_padding, kMXFP8Block);
  h.kv_slots = std::max(1, cfg.tokens + cfg.extra_slots);
  h.pages = round_up(h.kv_slots, cfg.page_size) / cfg.page_size;

  int q_blocks = cfg.dq / kMXFP8Block;
  int hkv = cfg.dkv / kHeadDim;
  int sfkv_count = h.pages * hkv * kMXFP8Block * (cfg.page_size / kMXFP8Block) * kBlocksPerHead;

  h.q.resize(static_cast<std::size_t>(cfg.tokens) * h.q_stride);
  h.k.resize(static_cast<std::size_t>(cfg.tokens) * h.k_stride);
  h.v.resize(static_cast<std::size_t>(cfg.tokens) * h.v_stride);
  h.q_gamma.resize(kHeadDim);
  h.k_gamma.resize(kHeadDim);
  h.tau.resize(cfg.tokens);
  h.loc.resize(cfg.tokens);
  h.k_out.resize(static_cast<std::size_t>(cfg.tokens) * cfg.dkv);
  h.v_out.resize(static_cast<std::size_t>(cfg.tokens) * cfg.dkv);
  h.q_mxfp8.resize(static_cast<std::size_t>(cfg.tokens) * h.q_mxfp8_stride);
  h.k_buf_mxfp8.resize(static_cast<std::size_t>(h.kv_slots) * h.kv_buf_stride);
  h.v_buf_mxfp8.resize(static_cast<std::size_t>(h.kv_slots) * h.kv_buf_stride);
  h.sfq.resize(static_cast<std::size_t>(cfg.tokens) * q_blocks);
  h.sfk.resize(sfkv_count);
  h.sfv.resize(sfkv_count);

  std::mt19937 gen(20260720u + static_cast<unsigned>(cfg.tokens * 13 + cfg.dq + cfg.dkv));
  std::uniform_real_distribution<float> x_dist(cfg.stress_values ? -3.0f : -0.9f, cfg.stress_values ? 3.0f : 0.9f);
  std::uniform_real_distribution<float> g_dist(0.75f, 1.25f);
  std::uniform_real_distribution<float> tau_dist(0.35f, cfg.stress_values ? 9.0f : 1.75f);

  for (auto& x : h.q) {
    x = Element(x_dist(gen));
  }
  for (auto& x : h.k) {
    x = Element(x_dist(gen));
  }
  for (auto& x : h.v) {
    x = Element(x_dist(gen));
  }
  for (auto& x : h.q_gamma) {
    x = Element(g_dist(gen));
  }
  for (auto& x : h.k_gamma) {
    x = Element(g_dist(gen));
  }
  for (int t = 0; t < cfg.tokens; ++t) {
    h.tau[t] = cfg.use_tau ? tau_dist(gen) : 1.0f;
    h.loc[t] = (cfg.include_negative_loc && (t % 17 == 5)) ? -1 : static_cast<int64_t>(t + 4);
  }

  std::fill(h.k_out.begin(), h.k_out.end(), Element(0.0f));
  std::fill(h.v_out.begin(), h.v_out.end(), Element(0.0f));
  std::fill(h.q_mxfp8.begin(), h.q_mxfp8.end(), 0xa5u);
  std::fill(h.k_buf_mxfp8.begin(), h.k_buf_mxfp8.end(), 0xc3u);
  std::fill(h.v_buf_mxfp8.begin(), h.v_buf_mxfp8.end(), 0x3cu);
  std::fill(h.sfq.begin(), h.sfq.end(), 0u);
  std::fill(h.sfk.begin(), h.sfk.end(), 0x7fu);
  std::fill(h.sfv.begin(), h.sfv.end(), 0x7fu);

  h.ref_k_out = h.k_out;
  h.ref_v_out = h.v_out;
  h.ref_q_mxfp8 = h.q_mxfp8;
  h.ref_k_buf_mxfp8 = h.k_buf_mxfp8;
  h.ref_v_buf_mxfp8 = h.v_buf_mxfp8;
  h.ref_sfq = h.sfq;
  h.ref_sfk = h.sfk;
  h.ref_sfv = h.sfv;
  return h;
}

template <typename Element>
void reference_case(CaseConfig const& cfg, HostTensors<Element>& h) {
  constexpr float eps = 1.0e-5f;
  int q_blocks = cfg.dq / kMXFP8Block;
  int k_blocks = cfg.dkv / kMXFP8Block;

  for (int t = 0; t < cfg.tokens; ++t) {
    for (int head = 0; head < cfg.dq / kHeadDim; ++head) {
      float partial[kBlocksPerHead];
      for (int block = 0; block < kBlocksPerHead; ++block) {
        float block_ss = 0.0f;
        for (int j = 0; j < kMXFP8Block; ++j) {
          int c = head * kHeadDim + block * kMXFP8Block + j;
          float x = to_float(h.q[static_cast<std::size_t>(t) * h.q_stride + c]);
          block_ss += x * x;
        }
        partial[block] = block_ss;
      }
      float ss = 0.0f;
      for (int block = 0; block < kBlocksPerHead; ++block) {
        ss += partial[block];
      }
      float inv = 1.0f / hd_sqrt(ss / static_cast<float>(kHeadDim) + eps);
      for (int block = 0; block < kBlocksPerHead; ++block) {
        int ch = head * kHeadDim + block * kMXFP8Block;
        float values[kMXFP8Block];
        for (int j = 0; j < kMXFP8Block; ++j) {
          int c = ch + j;
          float gamma = to_float(h.q_gamma[c % kHeadDim]);
          Element rounded = Element(to_float(h.q[static_cast<std::size_t>(t) * h.q_stride + c]) * inv * gamma);
          if (cfg.use_tau) {
            rounded = Element(to_float(rounded) * h.tau[t]);
          }
          values[j] = to_float(rounded);
        }
        int sf_idx = t * q_blocks + ch / kMXFP8Block;
        store_mxfp8_block(
            values,
            h.ref_q_mxfp8.data() + static_cast<std::size_t>(t) * h.q_mxfp8_stride + ch,
            h.ref_sfq.data() + sf_idx);
      }
    }

    for (int head = 0; head < cfg.dkv / kHeadDim; ++head) {
      float partial[kBlocksPerHead];
      for (int block = 0; block < kBlocksPerHead; ++block) {
        float block_ss = 0.0f;
        for (int j = 0; j < kMXFP8Block; ++j) {
          int c = head * kHeadDim + block * kMXFP8Block + j;
          float x = to_float(h.k[static_cast<std::size_t>(t) * h.k_stride + c]);
          block_ss += x * x;
        }
        partial[block] = block_ss;
      }
      float ss = 0.0f;
      for (int block = 0; block < kBlocksPerHead; ++block) {
        ss += partial[block];
      }
      float inv = 1.0f / hd_sqrt(ss / static_cast<float>(kHeadDim) + eps);
      for (int block = 0; block < kBlocksPerHead; ++block) {
        int ch = head * kHeadDim + block * kMXFP8Block;
        float values[kMXFP8Block];
        for (int j = 0; j < kMXFP8Block; ++j) {
          int c = ch + j;
          float gamma = to_float(h.k_gamma[c % kHeadDim]);
          Element rounded = Element(to_float(h.k[static_cast<std::size_t>(t) * h.k_stride + c]) * inv * gamma);
          values[j] = to_float(rounded);
          h.ref_k_out[static_cast<std::size_t>(t) * cfg.dkv + c] = rounded;
        }
        if (cfg.do_store && h.loc[t] >= 0) {
          int64_t sf_off = kv_scale_offset(h.loc[t], ch, cfg.dkv, cfg.page_size);
          store_mxfp8_block(
              values,
              h.ref_k_buf_mxfp8.data() + h.loc[t] * h.kv_buf_stride + ch,
              h.ref_sfk.data() + sf_off);
        }
      }
    }

    for (int block = 0; block < k_blocks; ++block) {
      int ch = block * kMXFP8Block;
      float values[kMXFP8Block];
      for (int j = 0; j < kMXFP8Block; ++j) {
        int c = ch + j;
        Element rounded = Element(to_float(h.v[static_cast<std::size_t>(t) * h.v_stride + c]));
        values[j] = to_float(rounded);
        h.ref_v_out[static_cast<std::size_t>(t) * cfg.dkv + c] = rounded;
      }
      if (cfg.do_store && h.loc[t] >= 0) {
        int64_t sf_off = kv_scale_offset(h.loc[t], ch, cfg.dkv, cfg.page_size);
        store_mxfp8_block(
            values,
            h.ref_v_buf_mxfp8.data() + h.loc[t] * h.kv_buf_stride + ch,
            h.ref_sfv.data() + sf_off);
      }
    }
  }
}

template <typename Element>
Mxfp8StoreTauParams<Element> make_params(HostTensors<Element> const& h, CaseConfig const& cfg) {
  Mxfp8StoreTauParams<Element> params;
  params.q = nullptr;
  params.k = nullptr;
  params.v = nullptr;
  params.q_gamma = nullptr;
  params.k_gamma = nullptr;
  params.tau = nullptr;
  params.loc = nullptr;
  params.k_out = nullptr;
  params.v_out = nullptr;
  params.q_mxfp8 = nullptr;
  params.k_buf_mxfp8 = nullptr;
  params.v_buf_mxfp8 = nullptr;
  params.sfq = nullptr;
  params.sfk = nullptr;
  params.sfv = nullptr;
  params.eps = 1.0e-5f;
  params.T = cfg.tokens;
  params.dq = cfg.dq;
  params.dkv = cfg.dkv;
  params.q_stride = h.q_stride;
  params.k_stride = h.k_stride;
  params.v_stride = h.v_stride;
  params.k_out_stride = cfg.dkv;
  params.v_out_stride = cfg.dkv;
  params.q_mxfp8_stride = h.q_mxfp8_stride;
  params.kv_buf_stride = h.kv_buf_stride;
  params.page_size = cfg.page_size;
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

VerifyResult compare_bytes(std::vector<uint8_t> const& got, std::vector<uint8_t> const& ref) {
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double abs_err = std::abs(static_cast<int>(got[i]) - static_cast<int>(ref[i]));
    result.max_abs = std::max(result.max_abs, abs_err);
    if (got[i] != ref[i] && result.passed) {
      result.passed = false;
      result.bad_index = static_cast<int>(i);
    }
  }
  return result;
}

void update_dequant_result(
    VerifyResult& result,
    float got_value,
    float ref_value,
    double atol,
    double rtol,
    int index) {
  double abs_err = std::abs(static_cast<double>(got_value) - static_cast<double>(ref_value));
  double rel_err = abs_err / std::max(1.0e-6, std::abs(static_cast<double>(ref_value)));
  result.max_abs = std::max(result.max_abs, abs_err);
  result.max_rel = std::max(result.max_rel, rel_err);
  if (abs_err > atol + rtol * std::abs(static_cast<double>(ref_value)) && result.passed) {
    result.passed = false;
    result.bad_index = index;
  }
}

VerifyResult compare_q_mxfp8_dequant(
    std::vector<uint8_t> const& got,
    std::vector<uint8_t> const& got_sf,
    std::vector<uint8_t> const& ref,
    std::vector<uint8_t> const& ref_sf,
    int T,
    int dq,
    int stride,
    double atol,
    double rtol) {
  VerifyResult result;
  int blocks = dq / kMXFP8Block;
  for (int t = 0; t < T; ++t) {
    for (int block = 0; block < blocks; ++block) {
      float got_scale = e8m0_scale_to_float(got_sf[static_cast<std::size_t>(t) * blocks + block]);
      float ref_scale = e8m0_scale_to_float(ref_sf[static_cast<std::size_t>(t) * blocks + block]);
      std::size_t base = static_cast<std::size_t>(t) * stride + block * kMXFP8Block;
      for (int j = 0; j < kMXFP8Block; ++j) {
        float got_value = e4m3fn_byte_to_float(got[base + j]) * got_scale;
        float ref_value = e4m3fn_byte_to_float(ref[base + j]) * ref_scale;
        update_dequant_result(result, got_value, ref_value, atol, rtol, static_cast<int>(base + j));
      }
    }
  }
  return result;
}

VerifyResult compare_kv_mxfp8_dequant(
    std::vector<uint8_t> const& got,
    std::vector<uint8_t> const& got_sf,
    std::vector<uint8_t> const& ref,
    std::vector<uint8_t> const& ref_sf,
    int kv_slots,
    int dkv,
    int stride,
    int page_size,
    double atol,
    double rtol) {
  VerifyResult result;
  int blocks = dkv / kMXFP8Block;
  for (int slot = 0; slot < kv_slots; ++slot) {
    for (int block = 0; block < blocks; ++block) {
      int ch = block * kMXFP8Block;
      int64_t sf_off = kv_scale_offset(slot, ch, dkv, page_size);
      float got_scale = e8m0_scale_to_float(got_sf[sf_off]);
      float ref_scale = e8m0_scale_to_float(ref_sf[sf_off]);
      std::size_t base = static_cast<std::size_t>(slot) * stride + ch;
      for (int j = 0; j < kMXFP8Block; ++j) {
        float got_value = e4m3fn_byte_to_float(got[base + j]) * got_scale;
        float ref_value = e4m3fn_byte_to_float(ref[base + j]) * ref_scale;
        update_dequant_result(result, got_value, ref_value, atol, rtol, static_cast<int>(base + j));
      }
    }
  }
  return result;
}

void print_verify_result(std::string const& label, VerifyResult const& result) {
  std::cout << "    " << std::setw(12) << label
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
  double q_read = T * cfg.dq * elem;
  double kv_read = T * 2.0 * cfg.dkv * elem;
  double gamma = T * (cfg.dq + cfg.dkv) * elem;
  double tau = cfg.use_tau ? T * 4.0 : 0.0;
  double loc = cfg.do_store ? T * 8.0 : 0.0;
  double q_store = T * cfg.dq;
  double q_scale = T * (cfg.dq / kMXFP8Block);
  double kv_out = T * 2.0 * cfg.dkv * elem;
  double kv_store = cfg.do_store ? T * 2.0 * cfg.dkv : 0.0;
  double kv_scale = cfg.do_store ? T * 2.0 * (cfg.dkv / kMXFP8Block) : 0.0;
  return q_read + kv_read + gamma + tau + loc + q_store + q_scale + kv_out + kv_store + kv_scale;
}

double estimate_flops(CaseConfig const& cfg) {
  double T = static_cast<double>(cfg.tokens);
  double q_norm = T * cfg.dq * (4.0 + (cfg.use_tau ? 1.0 : 0.0));
  double k_norm = T * cfg.dkv * 4.0;
  double quant = T * (cfg.dq + (cfg.do_store ? 2.0 * cfg.dkv : 0.0)) * 10.0;
  return q_norm + k_norm + quant;
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
    reference_case(cfg, h);
  }

  DeviceBuffer<Element> d_q(q, h.q.size());
  DeviceBuffer<Element> d_k(q, h.k.size());
  DeviceBuffer<Element> d_v(q, h.v.size());
  DeviceBuffer<Element> d_q_gamma(q, h.q_gamma.size());
  DeviceBuffer<Element> d_k_gamma(q, h.k_gamma.size());
  DeviceBuffer<float> d_tau(q, h.tau.size());
  DeviceBuffer<int64_t> d_loc(q, h.loc.size());
  DeviceBuffer<Element> d_k_out(q, h.k_out.size());
  DeviceBuffer<Element> d_v_out(q, h.v_out.size());
  DeviceBuffer<uint8_t> d_q_mxfp8(q, h.q_mxfp8.size());
  DeviceBuffer<uint8_t> d_k_buf_mxfp8(q, h.k_buf_mxfp8.size());
  DeviceBuffer<uint8_t> d_v_buf_mxfp8(q, h.v_buf_mxfp8.size());
  DeviceBuffer<uint8_t> d_sfq(q, h.sfq.size());
  DeviceBuffer<uint8_t> d_sfk(q, h.sfk.size());
  DeviceBuffer<uint8_t> d_sfv(q, h.sfv.size());

  d_q.copy_from(h.q);
  d_k.copy_from(h.k);
  d_v.copy_from(h.v);
  d_q_gamma.copy_from(h.q_gamma);
  d_k_gamma.copy_from(h.k_gamma);
  d_tau.copy_from(h.tau);
  d_loc.copy_from(h.loc);
  d_k_out.copy_from(h.k_out);
  d_v_out.copy_from(h.v_out);
  d_q_mxfp8.copy_from(h.q_mxfp8);
  d_k_buf_mxfp8.copy_from(h.k_buf_mxfp8);
  d_v_buf_mxfp8.copy_from(h.v_buf_mxfp8);
  d_sfq.copy_from(h.sfq);
  d_sfk.copy_from(h.sfk);
  d_sfv.copy_from(h.sfv);

  Mxfp8StoreTauParams<Element> params = make_params(h, cfg);
  params.q = d_q.get();
  params.k = d_k.get();
  params.v = d_v.get();
  params.q_gamma = d_q_gamma.get();
  params.k_gamma = d_k_gamma.get();
  params.tau = d_tau.get();
  params.loc = d_loc.get();
  params.k_out = d_k_out.get();
  params.v_out = d_v_out.get();
  params.q_mxfp8 = d_q_mxfp8.get();
  params.k_buf_mxfp8 = d_k_buf_mxfp8.get();
  params.v_buf_mxfp8 = d_v_buf_mxfp8.get();
  params.sfq = d_sfq.get();
  params.sfk = d_sfk.get();
  params.sfv = d_sfv.get();

  auto launch = [&]() {
    return launch_mxfp8_store_tau(q, params, cfg.use_tau, cfg.do_store);
  };

  launch().wait_and_throw();

  bool passed = true;
  if (verify) {
    d_k_out.copy_to(h.k_out);
    d_v_out.copy_to(h.v_out);
    d_q_mxfp8.copy_to(h.q_mxfp8);
    d_k_buf_mxfp8.copy_to(h.k_buf_mxfp8);
    d_v_buf_mxfp8.copy_to(h.v_buf_mxfp8);
    d_sfq.copy_to(h.sfq);
    d_sfk.copy_to(h.sfk);
    d_sfv.copy_to(h.sfv);

    double atol = std::is_same_v<Element, cutlass::bfloat16_t> ? 3.5e-2 : 4.0e-3;
    double rtol = std::is_same_v<Element, cutlass::bfloat16_t> ? 3.5e-2 : 4.0e-3;
    VerifyResult k_result = compare_close(h.k_out, h.ref_k_out, atol, rtol);
    VerifyResult v_result = compare_close(h.v_out, h.ref_v_out, atol, rtol);
    VerifyResult q8_result = compare_q_mxfp8_dequant(
        h.q_mxfp8, h.sfq, h.ref_q_mxfp8, h.ref_sfq, cfg.tokens, cfg.dq, h.q_mxfp8_stride, 8.0e-2, 3.0e-1);
    VerifyResult k8_result = compare_kv_mxfp8_dequant(
        h.k_buf_mxfp8, h.sfk, h.ref_k_buf_mxfp8, h.ref_sfk, h.kv_slots, cfg.dkv, h.kv_buf_stride, cfg.page_size,
        8.0e-2, 3.0e-1);
    VerifyResult v8_result = compare_kv_mxfp8_dequant(
        h.v_buf_mxfp8, h.sfv, h.ref_v_buf_mxfp8, h.ref_sfv, h.kv_slots, cfg.dkv, h.kv_buf_stride, cfg.page_size,
        8.0e-2, 3.0e-1);
    passed = k_result.passed && v_result.passed && q8_result.passed && k8_result.passed &&
        v8_result.passed;
    if (!passed) {
      print_verify_result("k_out", k_result);
      print_verify_result("v_out", v_result);
      print_verify_result("q_dequant", q8_result);
      print_verify_result("k_dequant", k8_result);
      print_verify_result("v_dequant", v8_result);
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
            << std::left << std::setw(32) << cfg.name << std::right
            << " T=" << cfg.tokens
            << " dq=" << cfg.dq
            << " dkv=" << cfg.dkv
            << " page=" << cfg.page_size
            << " tau=" << bool_text(cfg.use_tau)
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
      {"tiny_tau_store", 5, 128, 128, 128, 0, 0, 16, true, true, false, false},
      {"page_tail_negative_loc", 37, 256, 256, 128, 8, 32, 16, true, true, true, false},
      {"no_tau_no_store", 19, 128, 256, 128, 0, 0, 16, false, false, false, false},
      {"stress_values", 65, 384, 128, 128, 16, 0, 16, true, true, true, true},
  };
}

// Inkling MXFP8 store+tau: head_dim=128, num_kv_heads=4. dq = 128 * num_heads/tp,
// dkv = 128 * max(1, 4/tp). hidden_size=1536 → num_heads=12 (TP∈{1,2,4});
// hidden_size=6144 → num_heads=48 (TP∈{1,2,4,8}). MXFP8 KV pool uses page_size=128
// (upstream_model_attn.py:430). Token counts pick verify-like (batch*draft_token_num=9),
// extend-like (chunked prefill up to 16384), and decode-like (one token per seq).
std::vector<CaseConfig> inkling_suite() {
  return {
      // hidden_size=1536 (config defaults) — verify-like batch*9 tokens
      {"mxfp8_verify_h1536_tp1_dq1536_dkv512",  288, 1536, 512, 128, 0,  0, 16, true,  true,  false, false},
      {"mxfp8_verify_h1536_tp2_dq768_dkv256",   288,  768, 256, 128, 0,  0, 16, true,  true,  false, false},
      {"mxfp8_verify_h1536_tp4_dq384_dkv128",   576,  384, 128, 128, 0,  0, 16, true,  true,  false, false},
      // hidden_size=6144 (production checkpoint) — verify-like batch*9 tokens
      {"mxfp8_verify_h6144_tp1_dq6144_dkv512",  288, 6144, 512, 128, 0,  0, 16, true,  true,  false, false},
      {"mxfp8_verify_h6144_tp2_dq3072_dkv256",  288, 3072, 256, 128, 0,  0, 16, true,  true,  false, false},
      {"mxfp8_verify_h6144_tp4_dq1536_dkv128",  576, 1536, 128, 128, 0,  0, 16, true,  true,  false, false},
      {"mxfp8_verify_h6144_tp8_dq768_dkv128",  1152,  768, 128, 128, 0,  0, 16, true,  true,  false, false},
      // Decode-like: one token per active sequence
      {"mxfp8_decode_h1536_tp1_dq1536_dkv512",  512, 1536, 512, 128, 0, 64, 16, true,  true,  false, false},
      {"mxfp8_decode_h6144_tp1_dq6144_dkv512",  256, 6144, 512, 128, 0, 64, 16, true,  true,  false, false},
      // Extend-like: chunked prefill up to max_prefill_tokens=16384; use tail-page shape
      {"mxfp8_extend_h1536_tp1_dq1536_dkv512", 8191, 1536, 512, 128, 8, 32, 16, true,  true,  false, false},
      {"mxfp8_extend_h6144_tp2_dq3072_dkv256", 8191, 3072, 256, 128, 8, 32, 16, true,  true,  false, false},
      // Behavior variants at a real shape
      {"mxfp8_no_tau_h1536",                    288, 1536, 512, 128, 0,  0, 16, false, true,  false, false},
      {"mxfp8_no_store_swa_h1536",              288, 1536, 512, 128, 0,  0, 16, true,  false, false, false},
      {"mxfp8_negloc_h1536",                    288, 1536, 512, 128, 0,  0, 16, true,  true,  true,  false},
      {"mxfp8_stress_h1536",                    288, 1536, 512, 128, 0,  0, 16, true,  true,  false, true },
  };
}

// Perf-only sweep: sustained working sets beyond kMinSustainedTargetBytes = 32 MB.
// Token counts scale to cover chunked prefill (up to 16384) and decode (long batches).
std::vector<CaseConfig> perf_suite() {
  return {
      {"perf_h1536_tp1_dq1536_dkv512_t8192",  8192, 1536, 512, 128, 0, 64, 16, true, true, false, false, 135.0},
      {"perf_h1536_tp2_dq768_dkv256_t16384", 16384,  768, 256, 128, 0, 64, 16, true, true, false, false, 140.0},
      {"perf_h1536_tp4_dq384_dkv128_t16384", 16384,  384, 128, 128, 0, 64, 16, true, true, false, false, 140.0},
      {"perf_h6144_tp1_dq6144_dkv512_t4096",  4096, 6144, 512, 128, 0, 64, 16, true, true, false, false, 140.0},
      {"perf_h6144_tp2_dq3072_dkv256_t8192",  8192, 3072, 256, 128, 0, 64, 16, true, true, false, false, 145.0},
      {"perf_h6144_tp4_dq1536_dkv128_t16384",16384, 1536, 128, 128, 0, 64, 16, true, true, false, false, 130.0},
      {"perf_h6144_tp8_dq768_dkv128_t16384", 16384,  768, 128, 128, 0, 64, 16, true, true, false, false, 130.0},
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
      if (key == "T") {
        cfg.tokens = std::stoi(value);
      } else if (key == "dq") {
        cfg.dq = std::stoi(value);
      } else if (key == "dkv") {
        cfg.dkv = std::stoi(value);
      } else if (key == "page") {
        cfg.page_size = std::stoi(value);
      } else if (key == "rowpad") {
        cfg.row_padding = std::stoi(value);
      } else if (key == "kvpad") {
        cfg.kv_padding = std::stoi(value);
      } else if (key == "tau") {
        if (!parse_bool_value(value, cfg.use_tau)) {
          return false;
        }
      } else if (key == "store") {
        if (!parse_bool_value(value, cfg.do_store)) {
          return false;
        }
      } else if (key == "negloc") {
        if (!parse_bool_value(value, cfg.include_negative_loc)) {
          return false;
        }
      } else if (key == "stress") {
        if (!parse_bool_value(value, cfg.stress_values)) {
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

}  // namespace cutlass::examples::attn_prologue_mxfp8

int main(int argc, char const** argv) {
  using namespace cutlass::examples::attn_prologue_mxfp8;

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
          << "Inkling MXFP8 KV-store and log-tau prologue example\n\n"
          << "Options:\n"
          << "  --suite=<quick|inkling|perf>    Built-in shape suite (default: quick)\n"
          << "  --shape=T=...,dq=...,dkv=...,page=128,tau=1,store=1,negloc=0\n"
          << "                                  Single custom shape; overrides suite\n"
          << "  --dtype=<all|bf16|fp16>         Element dtype (default: all)\n"
          << "  --iterations=<int>              Timed kernel iterations\n"
          << "  --verify=<0|1>                  Run CPU reference comparison\n"
          << "  --target-gbps=<float>           Override sustained effective GB/s gate; 0 disables\n\n"
          << "Examples:\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_mxfp8_store_tau --suite=quick\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_mxfp8_store_tau --suite=inkling --dtype=bf16\n"
          << "  ./examples/15_bmg_attn_prologue/15_bmg_attn_prologue_mxfp8_store_tau --suite=perf --verify=0 --iterations=100\n";
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
