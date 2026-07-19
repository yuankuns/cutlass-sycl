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
    \brief Inkling packed SConv cache update example for CUTLASS SYCL.

    Semantics match modeltune/inkling/01_sconv/01_02_update_sconv_cache:

      old_state = has_initial_state[b] ? cache[cache_indices[b]] : zeros([W-1, D])
      history = concat(old_state, x[query_start_loc[b]:query_start_loc[b+1]])
      cache[cache_indices[b]] = history[-(W-1):]

    cache_indices[b] == -1 and empty query ranges leave the cache slot unchanged.
*/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/util/command_line.h"

#include <algorithm>
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

constexpr int kPadSlot = -1;
constexpr int kCopyBytes = 16;
constexpr int kCopyWords = kCopyBytes / static_cast<int>(sizeof(uint32_t));
constexpr int kPacksPerLane = 1;
constexpr int kThreads = 256;
// Smaller shapes are useful launch/cache smoke tests, but their fixed-address
// working sets can stay resident after warmup. Apply sustained-bandwidth gates
// only to cases with enough effective traffic to exceed typical on-chip cache.
constexpr double kMinSustainedTargetBytes = 32.0 * 1024.0 * 1024.0;

enum class DType {
  kBf16,
  kFp16
};

template <typename Element_>
struct UpdateSconvCacheParams {
  using Element = Element_;

  Element const* __restrict__ x;
  Element* __restrict__ cache;
  int32_t const* __restrict__ cache_indices;
  uint8_t const* __restrict__ has_initial_state;
  int32_t const* __restrict__ query_start_loc;
  int x_stride_t;
  int cache_stride_slot;
  int cache_stride_w;
  int batch;
  int width_minus_one;
  int channels;
};

template <typename Element, int StaticWidthMinusOne>
class UpdateSconvCacheKernel {
 public:
  UpdateSconvCacheParams<Element> params;
  int lanes_per_batch;
  int vec_count;
  int pack_elems;

  CUTLASS_DEVICE
  void copy_zero_pack(Element* row_base, int pack_idx) const {
    using pack_t = sycl::vec<uint32_t, kCopyWords>;
    pack_t zero(0u);
#pragma unroll
    for (int i = 0; i < kPacksPerLane; ++i) {
      zero.store(pack_idx + i, reinterpret_cast<uint32_t*>(row_base));
    }
  }

  CUTLASS_DEVICE
  void copy_pack(Element const* src_row_base, Element* dst_row_base, int pack_idx) const {
    using pack_t = sycl::vec<uint32_t, kCopyWords>;
#pragma unroll
    for (int i = 0; i < kPacksPerLane; ++i) {
      pack_t value;
      value.load(pack_idx + i, reinterpret_cast<uint32_t const*>(src_row_base));
      value.store(pack_idx + i, reinterpret_cast<uint32_t*>(dst_row_base));
    }
  }

  CUTLASS_DEVICE
  void copy_zero_scalar(Element* row_base, int channel) const {
    row_base[channel] = Element(0.0f);
  }

  CUTLASS_DEVICE
  void copy_scalar(Element const* src_row_base, Element* dst_row_base, int channel) const {
    dst_row_base[channel] = src_row_base[channel];
  }

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    constexpr bool kStaticWidth = StaticWidthMinusOne > 0;
    int width_minus_one = kStaticWidth ? StaticWidthMinusOne : params.width_minus_one;
    int linear = static_cast<int>(item.get_global_linear_id());
    int total_lanes = params.batch * lanes_per_batch;
    if (linear >= total_lanes) {
      return;
    }

    int b = linear / lanes_per_batch;
    int lane = linear - b * lanes_per_batch;

    int cache_slot = params.cache_indices[b];
    int start = params.query_start_loc[b];
    int end = params.query_start_loc[b + 1];
    int qlen = end - start;
    if (cache_slot == kPadSlot || qlen <= 0) {
      return;
    }

    bool has_state = params.has_initial_state[b] != 0;
    bool is_vec_lane = lane < vec_count;
    int lane_elems = pack_elems * kPacksPerLane;
    int channel = is_vec_lane ? lane * lane_elems : vec_count * lane_elems + (lane - vec_count);
    int slot_base = cache_slot * params.cache_stride_slot;

    // Low-to-high stores are RAW-safe: shifted cache sources are at w + qlen > w.
#pragma unroll
    for (int w = 0; w < width_minus_one; ++w) {
      Element* dst_row = params.cache + slot_base + w * params.cache_stride_w;
      if (qlen >= width_minus_one - w) {
        int x_idx = end - width_minus_one + w;
        Element const* src_row = params.x + x_idx * params.x_stride_t;
        if (is_vec_lane) {
          copy_pack(src_row, dst_row, lane * kPacksPerLane);
        } else {
          copy_scalar(src_row, dst_row, channel);
        }
      } else if (has_state) {
        Element const* src_row = params.cache + slot_base + (w + qlen) * params.cache_stride_w;
        if (is_vec_lane) {
          copy_pack(src_row, dst_row, lane * kPacksPerLane);
        } else {
          copy_scalar(src_row, dst_row, channel);
        }
      } else {
        if (is_vec_lane) {
          copy_zero_pack(dst_row, lane * kPacksPerLane);
        } else {
          copy_zero_scalar(dst_row, channel);
        }
      }
    }
  }
};

template <typename Element, int StaticWidthMinusOne>
sycl::event launch_update_sconv_cache_static(sycl::queue& q, UpdateSconvCacheParams<Element> const& params) {
  if (params.batch == 0 || params.width_minus_one == 0 || params.channels == 0) {
    return sycl::event{};
  }

  int pack_elems = kCopyBytes / static_cast<int>(sizeof(Element));
  int lane_elems = pack_elems * kPacksPerLane;
  auto aligned_elems = [pack_elems](int stride) {
    return stride % pack_elems == 0;
  };
  bool aligned = aligned_elems(params.x_stride_t) &&
      aligned_elems(params.cache_stride_slot) &&
      aligned_elems(params.cache_stride_w) &&
      (reinterpret_cast<std::uintptr_t>(params.x) % kCopyBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.cache) % kCopyBytes == 0);

  int vec_count = aligned ? params.channels / lane_elems : 0;
  int scalar_tail = params.channels - vec_count * lane_elems;
  int lanes_per_batch = vec_count + scalar_tail;
  int total_lanes = params.batch * lanes_per_batch;
  if (total_lanes == 0) {
    return sycl::event{};
  }

  // Roofline: this update has effectively zero arithmetic intensity. It is
  // memory-bound, so the optimized path uses wide copies with scalar lanes only
  // for tails or misaligned/strided layouts.
  int global = ((total_lanes + kThreads - 1) / kThreads) * kThreads;
  UpdateSconvCacheKernel<Element, StaticWidthMinusOne> kernel{params, lanes_per_batch, vec_count, pack_elems};
  return q.parallel_for<UpdateSconvCacheKernel<Element, StaticWidthMinusOne>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename Element>
sycl::event launch_update_sconv_cache(sycl::queue& q, UpdateSconvCacheParams<Element> const& params) {
  switch (params.width_minus_one) {
    case 1:
      return launch_update_sconv_cache_static<Element, 1>(q, params);
    case 2:
      return launch_update_sconv_cache_static<Element, 2>(q, params);
    case 3:
      return launch_update_sconv_cache_static<Element, 3>(q, params);
    case 5:
      return launch_update_sconv_cache_static<Element, 5>(q, params);
    case 7:
      return launch_update_sconv_cache_static<Element, 7>(q, params);
    case 8:
      return launch_update_sconv_cache_static<Element, 8>(q, params);
    default:
      return launch_update_sconv_cache_static<Element, 0>(q, params);
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
  int batch = 1;
  int width_minus_one = 3;
  int channels = 128;
  int qlen = 1;
  bool mixed_lengths = false;
  bool include_pad = false;
  bool include_empty = false;
  int x_padding = 0;
  int cache_padding = 0;
  int slot_padding = 0;
  int x_offset = 0;
  int cache_offset = 0;
  bool random_metadata = false;
  unsigned seed = 0;
};

template <typename Element_>
struct HostTensors {
  using Element = Element_;

  std::vector<Element> x;
  std::vector<Element> cache;
  std::vector<Element> ref;
  std::vector<int32_t> cache_indices;
  std::vector<uint8_t> has_initial_state;
  std::vector<int32_t> query_start_loc;
  int slots = 0;
  int x_stride_t = 0;
  int cache_stride_w = 0;
  int cache_stride_slot = 0;
};

template <typename Element>
float to_float(Element x) {
  return static_cast<float>(x);
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

std::string bool_text(bool value) {
  return value ? "true" : "false";
}

std::vector<int> make_lengths(CaseConfig const& cfg) {
  std::vector<int> lengths(cfg.batch, cfg.qlen);
  if (cfg.random_metadata) {
    std::mt19937 gen(cfg.seed);
    std::uniform_int_distribution<int> len_dist(0, std::max(cfg.qlen, cfg.width_minus_one + 5));
    for (int b = 0; b < cfg.batch; ++b) {
      lengths[b] = len_dist(gen);
    }
  } else if (cfg.mixed_lengths) {
    for (int b = 0; b < cfg.batch; ++b) {
      int pattern = b % 6;
      if (pattern == 0 && cfg.include_empty) {
        lengths[b] = 0;
      } else if (pattern == 1) {
        lengths[b] = 1;
      } else if (pattern == 2) {
        lengths[b] = std::max(1, cfg.width_minus_one - 1);
      } else if (pattern == 3) {
        lengths[b] = cfg.width_minus_one;
      } else if (pattern == 4) {
        lengths[b] = cfg.width_minus_one + 3;
      } else {
        lengths[b] = cfg.qlen;
      }
    }
  } else if (cfg.include_empty && cfg.batch > 0) {
    lengths[0] = 0;
  }
  return lengths;
}

template <typename Element>
HostTensors<Element> initialize_case(CaseConfig const& cfg) {
  HostTensors<Element> h;
  h.slots = std::max(cfg.batch + 8, 8);
  h.x_stride_t = cfg.channels + cfg.x_padding;
  h.cache_stride_w = cfg.channels + cfg.cache_padding;
  h.cache_stride_slot = cfg.width_minus_one * h.cache_stride_w + cfg.slot_padding;
  h.cache_indices.resize(cfg.batch);
  h.has_initial_state.resize(cfg.batch);
  h.query_start_loc.resize(cfg.batch + 1);

  auto lengths = make_lengths(cfg);
  h.query_start_loc[0] = 0;
  for (int b = 0; b < cfg.batch; ++b) {
    h.query_start_loc[b + 1] = h.query_start_loc[b] + lengths[b];
  }
  int total_tokens = h.query_start_loc.back();
  std::size_t x_storage_size = static_cast<std::size_t>(cfg.x_offset);
  if (total_tokens > 0) {
    x_storage_size += static_cast<std::size_t>(total_tokens - 1) * h.x_stride_t + cfg.channels;
  }
  std::size_t cache_storage_size = static_cast<std::size_t>(cfg.cache_offset);
  if (h.slots > 0 && cfg.width_minus_one > 0) {
    cache_storage_size += static_cast<std::size_t>(h.slots - 1) * h.cache_stride_slot
        + static_cast<std::size_t>(cfg.width_minus_one - 1) * h.cache_stride_w + cfg.channels;
  }
  h.x.resize(std::max<std::size_t>(1, x_storage_size));
  h.cache.resize(std::max<std::size_t>(1, cache_storage_size));
  h.ref.resize(h.cache.size());

  unsigned seed = cfg.seed == 0
      ? 20260718u + static_cast<unsigned>(cfg.batch * 13 + cfg.channels * 5 + cfg.width_minus_one)
      : cfg.seed;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> c_dist(-0.25f, 0.25f);

  for (auto& v : h.x) {
    v = Element(x_dist(gen));
  }
  for (auto& v : h.cache) {
    v = Element(c_dist(gen));
  }

  for (int b = 0; b < cfg.batch; ++b) {
    h.cache_indices[b] = b;
    h.has_initial_state[b] = static_cast<uint8_t>((b % 4) != 1);
  }

  if (cfg.random_metadata) {
    std::uniform_int_distribution<int> pad_dist(0, 5);
    std::uniform_int_distribution<int> state_dist(0, 1);
    for (int b = 0; b < cfg.batch; ++b) {
      h.cache_indices[b] = (cfg.include_pad && pad_dist(gen) == 0) ? kPadSlot : b;
      h.has_initial_state[b] = static_cast<uint8_t>(state_dist(gen));
    }
  } else {
    for (int b = 0; b < cfg.batch; ++b) {
      h.cache_indices[b] = (cfg.include_pad && b % 7 == 3) ? kPadSlot : b;
    }
  }

  return h;
}

template <typename Element>
void reference_update(CaseConfig const& cfg, HostTensors<Element>& h) {
  h.ref = h.cache;
  for (int b = 0; b < cfg.batch; ++b) {
    int slot = h.cache_indices[b];
    int start = h.query_start_loc[b];
    int end = h.query_start_loc[b + 1];
    int qlen = end - start;
    if (slot == kPadSlot || qlen <= 0) {
      continue;
    }

    bool has_state = h.has_initial_state[b] != 0;
    for (int w = 0; w < cfg.width_minus_one; ++w) {
      for (int d = 0; d < cfg.channels; ++d) {
        std::size_t dst = static_cast<std::size_t>(cfg.cache_offset)
            + static_cast<std::size_t>(slot) * h.cache_stride_slot
            + static_cast<std::size_t>(w) * h.cache_stride_w + d;
        if (qlen >= cfg.width_minus_one - w) {
          int x_idx = end - cfg.width_minus_one + w;
          std::size_t src = static_cast<std::size_t>(cfg.x_offset)
              + static_cast<std::size_t>(x_idx) * h.x_stride_t + d;
          h.ref[dst] = h.x[src];
        } else if (has_state) {
          int src_w = w + qlen;
          std::size_t src = static_cast<std::size_t>(cfg.cache_offset)
              + static_cast<std::size_t>(slot) * h.cache_stride_slot
              + static_cast<std::size_t>(src_w) * h.cache_stride_w + d;
          h.ref[dst] = h.cache[src];
        } else {
          h.ref[dst] = Element(0.0f);
        }
      }
    }
  }
}

struct VerifyResult {
  bool passed = true;
  int index = 0;
  float got = 0.0f;
  float expected = 0.0f;
};

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
      return result;
    }
  }
  return result;
}

template <typename Element>
double effective_bytes(CaseConfig const& cfg, HostTensors<Element> const& h) {
  double bytes = 0.0;
  for (int b = 0; b < cfg.batch; ++b) {
    int slot = h.cache_indices[b];
    int start = h.query_start_loc[b];
    int end = h.query_start_loc[b + 1];
    int qlen = end - start;
    if (slot == kPadSlot || qlen <= 0) {
      continue;
    }
    bool has_state = h.has_initial_state[b] != 0;
    for (int w = 0; w < cfg.width_minus_one; ++w) {
      bytes += static_cast<double>(cfg.channels) * sizeof(Element);
      if (qlen >= cfg.width_minus_one - w || has_state) {
        bytes += static_cast<double>(cfg.channels) * sizeof(Element);
      }
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
    double target_gbps) {
  HostTensors<Element> h = initialize_case<Element>(cfg);
  if (verify) {
    reference_update<Element>(cfg, h);
  }

  DeviceBuffer<Element> d_x(q, h.x.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<uint8_t> d_has_initial_state(q, h.has_initial_state.size());
  DeviceBuffer<int32_t> d_query_start_loc(q, h.query_start_loc.size());

  d_x.copy_from(h.x);
  d_cache.copy_from(h.cache);
  d_cache_indices.copy_from(h.cache_indices);
  d_has_initial_state.copy_from(h.has_initial_state);
  d_query_start_loc.copy_from(h.query_start_loc);

  UpdateSconvCacheParams<Element> params{
      d_x.get() + cfg.x_offset,
      d_cache.get() + cfg.cache_offset,
      d_cache_indices.get(),
      d_has_initial_state.get(),
      d_query_start_loc.get(),
      h.x_stride_t,
      h.cache_stride_slot,
      h.cache_stride_w,
      cfg.batch,
      cfg.width_minus_one,
      cfg.channels};

  auto launch = [&]() {
    return launch_update_sconv_cache<Element>(q, params);
  };

  launch();
  q.wait_and_throw();

  bool passed = true;
  VerifyResult vr;
  if (verify) {
    std::vector<Element> got(h.cache.size());
    d_cache.copy_to(got);
    vr = verify_cache<Element>(got, h.ref);
    passed = vr.passed;
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
  double bytes = effective_bytes<Element>(cfg, h);
  double gbps = (bytes / 1.0e9) / avg_s;
  bool applies_target = target_gbps > 0.0 && bytes >= kMinSustainedTargetBytes;
  if (applies_target && gbps < target_gbps) {
    passed = false;
  }

  std::cout << std::left << std::setw(30) << cfg.name
            << " B=" << std::setw(5) << cfg.batch
            << " W-1=" << std::setw(3) << cfg.width_minus_one
            << " D=" << std::setw(6) << cfg.channels
            << " qlen=" << std::setw(5) << cfg.qlen
            << " mixed=" << bool_text(cfg.mixed_lengths)
            << "  " << std::fixed << std::setprecision(3)
            << (avg_s * 1000.0) << " ms"
            << "  " << gbps << " GB/s";

  if (applies_target) {
    std::cout << "  target=" << target_gbps << " GB/s";
  } else if (target_gbps > 0.0) {
    std::cout << "  target=skipped-cache-smoke";
  }
  if (verify) {
    std::cout << "  " << (passed ? "passed" : "failed");
    if (!passed) {
      std::cout << " index=" << vr.index
                << " got=" << vr.got
                << " expected=" << vr.expected;
    }
  } else {
    std::cout << "  verification skipped";
  }
  std::cout << "\n";

  return passed;
}

std::vector<CaseConfig> quick_suite() {
  // Inkling calls update with W-1 = sconv_kernel_size - 1 = 3 and D chosen from
  // {hidden_size=1536, hidden_size/tp=768/384/192, head_dim*num_tp_kv_heads=512/256/128}.
  return {
      {"decode_b5_w3_d7", 5, 3, 7, 1, false, true, false},
      {"decode_b32_w3_d128", 32, 3, 128, 1, false, true, false},
      {"extend_mixed_b8_w7_d257", 8, 7, 257, 8, true, true, true},
      {"extend_b32_w7_d4096_q8", 32, 7, 4096, 8, false, false, false},
      {"edge_w1_b6_d15", 6, 1, 15, 2, true, true, true},
      {"empty_all_b4_w3_d64", 4, 3, 64, 0, false, false, false},
      {"padded_aligned_b7_w5_d130", 7, 5, 130, 6, true, true, true, 14, 14, 64},
      {"misaligned_scalar_b6_w3_d19", 6, 3, 19, 2, true, true, true, 3, 5, 11, 1, 1},
      {"random_meta_b17_w7_d263", 17, 7, 263, 9, false, true, true, 4, 9, 17, 0, 0, true, 12345u},
      {"inkling_verify_b16_w3_d1536_q9", 16, 3, 1536, 9, false, true, false},
      {"inkling_verify_b16_w3_d512_q9", 16, 3, 512, 9, false, true, false},
      {"inkling_extend_mixed_b8_w3_d768_q128", 8, 3, 768, 128, true, true, true},
      {"inkling_extend_mixed_b8_w3_d384_q128", 8, 3, 384, 128, true, true, true},
      {"inkling_extend_mixed_b8_w3_d192_q128", 8, 3, 192, 128, true, true, true},
      {"inkling_extend_mixed_b8_w3_d256_q128", 8, 3, 256, 128, true, true, true},
  };
}

std::vector<CaseConfig> stress_suite() {
  return {
      {"stress_00_b3_w2_d1", 3, 2, 1, 4, false, true, true, 1, 2, 3, 1, 0, true, 1000u},
      {"stress_01_b9_w3_d31", 9, 3, 31, 5, false, true, true, 5, 7, 11, 0, 1, true, 1001u},
      {"stress_02_b11_w5_d64", 11, 5, 64, 6, false, true, true, 8, 8, 16, 0, 0, true, 1002u},
      {"stress_03_b13_w7_d129", 13, 7, 129, 9, false, true, true, 3, 5, 19, 1, 1, true, 1003u},
      {"stress_04_b15_w8_d257", 15, 8, 257, 10, false, true, true, 7, 11, 23, 0, 0, true, 1004u},
      {"stress_05_b17_w1_d512", 17, 1, 512, 3, false, true, true, 0, 0, 0, 0, 0, true, 1005u},
      {"stress_06_b19_w3_d770", 19, 3, 770, 12, false, true, true, 14, 14, 32, 0, 0, true, 1006u},
      {"stress_07_b23_w7_d1025", 23, 7, 1025, 11, false, true, true, 1, 3, 5, 1, 1, true, 1007u},
  };
}

std::vector<CaseConfig> perf_suite() {
  // Inkling always calls update with W-1 = sconv_kernel_size - 1 = 3. Per-layer
  // dims across TP configs:
  //   attn/mlp non-scattered : D=1536
  //   attn/mlp scattered     : D=1536/tp = 1536, 768, 384, 192
  //   k/v_sconv              : D=head_dim*num_tp_kv_heads = 512, 256, 128, 128
  return {
      {"decode_launch_smoke_b1_w3_d4096", 1, 3, 4096, 1, false, false, false},
      {"extend_cache_smoke_b32_w7_d4096_q8", 32, 7, 4096, 8, false, false, false},
      {"extend_b256_w7_d7168_q256", 256, 7, 7168, 256, false, false, false},
      {"inkling_extend_b64_w3_d1536_q1024", 64, 3, 1536, 1024, false, false, false},
      {"inkling_extend_b64_w3_d768_q1024", 64, 3, 768, 1024, false, false, false},
      {"inkling_extend_b64_w3_d512_q1024", 64, 3, 512, 1024, false, false, false},
      {"inkling_extend_b64_w3_d384_q1024", 64, 3, 384, 1024, false, false, false},
      {"inkling_extend_b64_w3_d256_q1024", 64, 3, 256, 1024, false, false, false},
      {"inkling_extend_b64_w3_d192_q1024", 64, 3, 192, 1024, false, false, false},
      {"inkling_extend_b64_w3_d128_q1024", 64, 3, 128, 1024, false, false, false},
      {"inkling_verify_b128_w3_d1536_q9", 128, 3, 1536, 9, false, false, false},
      {"inkling_verify_b128_w3_d512_q9", 128, 3, 512, 9, false, false, false},
      {"inkling_verify_b128_w3_d192_q9", 128, 3, 192, 9, false, false, false},
      {"inkling_verify_b128_w3_d128_q9", 128, 3, 128, 9, false, false, false},
      {"inkling_extend_b256_w3_d1536_q256", 256, 3, 1536, 256, false, false, false},
  };
}

bool parse_single_shape(std::string const& text, CaseConfig& cfg) {
  if (text.empty()) {
    return false;
  }

  cfg = {"custom", 1, 3, 128, 1, false, false, false};
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
    } else if (key == "B") {
      cfg.batch = std::stoi(value);
    } else if (key == "Wm1") {
      cfg.width_minus_one = std::stoi(value);
    } else if (key == "D") {
      cfg.channels = std::stoi(value);
    } else if (key == "qlen") {
      cfg.qlen = std::stoi(value);
    } else if (key == "mixed") {
      cfg.mixed_lengths = std::stoi(value) != 0;
    } else if (key == "pad") {
      cfg.include_pad = std::stoi(value) != 0;
    } else if (key == "empty") {
      cfg.include_empty = std::stoi(value) != 0;
    } else if (key == "xpad") {
      cfg.x_padding = std::stoi(value);
    } else if (key == "cachepad") {
      cfg.cache_padding = std::stoi(value);
    } else if (key == "slotpad") {
      cfg.slot_padding = std::stoi(value);
    } else if (key == "xoff") {
      cfg.x_offset = std::stoi(value);
    } else if (key == "cacheoff") {
      cfg.cache_offset = std::stoi(value);
    } else if (key == "random") {
      cfg.random_metadata = std::stoi(value) != 0;
    } else if (key == "seed") {
      cfg.seed = static_cast<unsigned>(std::stoul(value));
    } else {
      return false;
    }
  }

  return cfg.batch > 0 && cfg.width_minus_one > 0 && cfg.channels > 0 && cfg.qlen >= 0 &&
      cfg.x_padding >= 0 && cfg.cache_padding >= 0 && cfg.slot_padding >= 0 &&
      cfg.x_offset >= 0 && cfg.cache_offset >= 0;
}

struct Options {
  bool help = false;
  bool valid = true;
  bool verify = true;
  int iterations = 20;
  std::string suite = "quick";
  std::string shape;
  std::string dtype_name = "bf16";
  DType dtype = DType::kBf16;
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
    cmd.get_cmd_line_argument("dtype", dtype_name, std::string("bf16"));
    cmd.get_cmd_line_argument("target-gbps", target_gbps, 0.0);
    if (!parse_dtype(dtype_name, dtype)) {
      valid = false;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Inkling BMG SConv Cache Update Example\n\n"
        << "Options:\n"
        << "  --help                         Print this message\n"
        << "  --suite=<quick|stress|perf>     Built-in shape suite (default: quick)\n"
        << "  --shape=<k=v,...>               Run one custom shape instead of a suite\n"
        << "                                  Keys: name,B,Wm1,D,qlen,mixed,pad,empty,\n"
        << "                                        xpad,cachepad,slotpad,xoff,cacheoff,random,seed\n"
        << "  --dtype=<bf16|fp16>             Input/cache dtype (default: bf16)\n"
        << "  --iterations=<int>              Timed kernel iterations\n"
        << "  --verify=<0|1>                  Run CPU reference comparison\n"
        << "  --target-gbps=<float>           Fail if any timed case is below this effective GB/s\n\n"
        << "Examples:\n"
        << "  ./examples/14_bmg_sconv/14_bmg_update_sconv_cache --suite=quick\n"
        << "  ./examples/14_bmg_sconv/14_bmg_update_sconv_cache --suite=quick --dtype=fp16\n"
        << "  ./examples/14_bmg_sconv/14_bmg_update_sconv_cache --suite=perf --verify=0 --iterations=100\n"
        << "  ./examples/14_bmg_sconv/14_bmg_update_sconv_cache --shape=B=32,Wm1=7,D=4096,qlen=8\n"
        << "  ./examples/14_bmg_sconv/14_bmg_update_sconv_cache --shape=B=6,Wm1=3,D=19,qlen=2,xoff=1,cacheoff=1,xpad=3,cachepad=5\n";
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
      if (options.dtype == DType::kBf16) {
        all_passed &= run_case<cutlass::bfloat16_t>(
            q, cfg, options.iterations, options.verify, options.target_gbps);
      } else {
        all_passed &= run_case<cutlass::half_t>(
            q, cfg, options.iterations, options.verify, options.target_gbps);
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
