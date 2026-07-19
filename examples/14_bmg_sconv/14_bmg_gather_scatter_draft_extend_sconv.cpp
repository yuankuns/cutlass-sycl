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
    \brief Inkling gather/scatter and draft-extend SConv cache update example for CUTLASS SYCL.

    Semantics match modeltune/inkling/01_sconv/01_04_gather_scatter_and_draft_extend:

      gather/scatter:
        if mask[b]:
          cache[dst[b], w, :] = hidden[track_idx[b, w], :]

      draft extend:
        virtual = concat(cache[cache_indices[b]], hidden[b, 0:draft_token_num])
        cache[cache_indices[b], w, :] = virtual[num_accepted[b] + w, :]
        if tracking and crossed[b]:
          cache[track_indices[b], w, :] = virtual[track_step[b] + w, :]

    Roofline: both kernels are pure copy/select operations with no useful
    arithmetic. Arithmetic intensity is effectively 0 ops/byte, so they are
    memory-bound. The optimized path uses 16-byte vector copies when base
    pointers and row/slot strides are aligned, with scalar lanes only for tails
    and deliberately misaligned layouts. Performance reporting therefore uses
    effective GB/s rather than TOPS.
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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cutlass::examples::sconv {

constexpr int kPadSlot = -1;
constexpr int kThreads = 256;
constexpr int kCopyBytes = 16;
constexpr int kCopyWords = kCopyBytes / static_cast<int>(sizeof(uint32_t));
constexpr int kPacksPerLane = 1;
constexpr int kMaxWindow = 16;
constexpr double kMinSustainedTargetBytes = 32.0 * 1024.0 * 1024.0;

enum class DType {
  kAll,
  kBf16,
  kFp16
};

enum class Op {
  kAll,
  kGather,
  kDraft
};

template <typename Element_>
struct GatherScatterParams {
  using Element = Element_;

  Element const* __restrict__ hidden;
  Element* __restrict__ cache;
  int32_t const* __restrict__ track_idx;
  uint8_t const* __restrict__ mask;
  int64_t const* __restrict__ dst;
  int hidden_stride_t;
  int cache_stride_slot;
  int cache_stride_w;
  int track_stride_b;
  int track_stride_w;
  int dst_stride_b;
  int batch;
  int width_minus_one;
  int channels;
  int pad_slot_id;
};

template <typename Element_>
struct DraftExtendParams {
  using Element = Element_;

  Element const* __restrict__ hidden;
  Element* __restrict__ cache;
  int32_t const* __restrict__ cache_indices;
  int32_t const* __restrict__ num_accepted;
  uint8_t const* __restrict__ crossed;
  int32_t const* __restrict__ track_step;
  int64_t const* __restrict__ track_indices;
  int hidden_stride_b;
  int hidden_stride_t;
  int cache_stride_slot;
  int cache_stride_w;
  int batch;
  int width_minus_one;
  int channels;
  int pad_slot_id;
};

template <typename Element>
CUTLASS_HOST_DEVICE
float to_float(Element x) {
  return static_cast<float>(x);
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

std::string op_text(Op op) {
  switch (op) {
    case Op::kAll:
      return "all";
    case Op::kGather:
      return "gather";
    case Op::kDraft:
      return "draft";
  }
  return "unknown";
}

bool parse_op(std::string const& text, Op& op) {
  if (text == "all") {
    op = Op::kAll;
    return true;
  }
  if (text == "gather") {
    op = Op::kGather;
    return true;
  }
  if (text == "draft") {
    op = Op::kDraft;
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

template <typename Element>
CUTLASS_HOST
bool exact_equal(Element a, Element b) {
  return to_float(a) == to_float(b);
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

template <typename Element, int StaticWidthMinusOne>
class GatherScatterSconvKernel {
 public:
  GatherScatterParams<Element> params;
  int lanes_per_batch;
  int vec_count;
  int pack_elems;

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
  void copy_scalar(Element const* src_row_base, Element* dst_row_base, int channel) const {
    dst_row_base[channel] = src_row_base[channel];
  }

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int width = StaticWidthMinusOne > 0 ? StaticWidthMinusOne : params.width_minus_one;
    int linear = static_cast<int>(item.get_global_linear_id());
    int total = params.batch * lanes_per_batch;
    if (linear >= total) {
      return;
    }

    int b = linear / lanes_per_batch;
    int lane = linear - b * lanes_per_batch;
    if (params.mask[b] == 0) {
      return;
    }

    int64_t dst_slot = params.dst[static_cast<std::size_t>(b) * params.dst_stride_b];
    if (dst_slot == params.pad_slot_id) {
      return;
    }

    bool is_vec_lane = lane < vec_count;
    int lane_elems = pack_elems * kPacksPerLane;
    int channel = is_vec_lane ? lane * lane_elems : vec_count * lane_elems + (lane - vec_count);
    int64_t cache_base = dst_slot * static_cast<int64_t>(params.cache_stride_slot);
    int track_base = b * params.track_stride_b;

#pragma unroll
    for (int w = 0; w < width; ++w) {
      int src_t = params.track_idx[track_base + w * params.track_stride_w];
      Element const* src_row = params.hidden + static_cast<std::size_t>(src_t) * params.hidden_stride_t;
      Element* dst_row = params.cache + cache_base + static_cast<int64_t>(w) * params.cache_stride_w;
      if (is_vec_lane) {
        copy_pack(src_row, dst_row, lane * kPacksPerLane);
      } else {
        copy_scalar(src_row, dst_row, channel);
      }
    }
  }
};

template <typename Element, int StaticWidthMinusOne>
sycl::event launch_gather_scatter_static(
    sycl::queue& q,
    GatherScatterParams<Element> const& params) {
  if (params.batch == 0 || params.width_minus_one == 0 || params.channels == 0) {
    return sycl::event{};
  }
  if (params.width_minus_one > kMaxWindow) {
    throw std::invalid_argument("gather_scatter width_minus_one exceeds kMaxWindow");
  }

  int pack_elems = kCopyBytes / static_cast<int>(sizeof(Element));
  int lane_elems = pack_elems * kPacksPerLane;
  auto aligned_elems = [pack_elems](int stride) {
    return stride % pack_elems == 0;
  };
  bool aligned = aligned_elems(params.hidden_stride_t) &&
      aligned_elems(params.cache_stride_slot) &&
      aligned_elems(params.cache_stride_w) &&
      (reinterpret_cast<std::uintptr_t>(params.hidden) % kCopyBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.cache) % kCopyBytes == 0);

  int vec_count = aligned ? params.channels / lane_elems : 0;
  int scalar_tail = params.channels - vec_count * lane_elems;
  int lanes_per_batch = vec_count + scalar_tail;
  int total_lanes = params.batch * lanes_per_batch;
  int global = ((total_lanes + kThreads - 1) / kThreads) * kThreads;

  GatherScatterSconvKernel<Element, StaticWidthMinusOne> kernel{
      params, lanes_per_batch, vec_count, pack_elems};
  return q.parallel_for<GatherScatterSconvKernel<Element, StaticWidthMinusOne>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename Element>
sycl::event launch_gather_scatter(
    sycl::queue& q,
    GatherScatterParams<Element> const& params) {
  switch (params.width_minus_one) {
    case 1:
      return launch_gather_scatter_static<Element, 1>(q, params);
    case 2:
      return launch_gather_scatter_static<Element, 2>(q, params);
    case 3:
      return launch_gather_scatter_static<Element, 3>(q, params);
    case 5:
      return launch_gather_scatter_static<Element, 5>(q, params);
    case 7:
      return launch_gather_scatter_static<Element, 7>(q, params);
    case 8:
      return launch_gather_scatter_static<Element, 8>(q, params);
    default:
      return launch_gather_scatter_static<Element, 0>(q, params);
  }
}

template <typename Element, int StaticWidthMinusOne, bool DoTrack>
class DraftExtendSconvKernel {
 public:
  DraftExtendParams<Element> params;
  int lanes_per_batch;
  int vec_count;
  int pack_elems;

  using pack_t = sycl::vec<uint32_t, kCopyWords>;

  CUTLASS_DEVICE
  pack_t load_pack(Element const* row_base, int pack_idx) const {
    pack_t value;
    value.load(pack_idx, reinterpret_cast<uint32_t const*>(row_base));
    return value;
  }

  CUTLASS_DEVICE
  void store_pack(Element* row_base, int pack_idx, pack_t value) const {
    value.store(pack_idx, reinterpret_cast<uint32_t*>(row_base));
  }

  CUTLASS_DEVICE
  void operator()(sycl::nd_item<1> item) const {
    int width = StaticWidthMinusOne > 0 ? StaticWidthMinusOne : params.width_minus_one;
    int linear = static_cast<int>(item.get_global_linear_id());
    int total = params.batch * lanes_per_batch;
    if (linear >= total) {
      return;
    }

    int b = linear / lanes_per_batch;
    int lane = linear - b * lanes_per_batch;
    int n_acc = params.num_accepted[b];
    int cache_slot = params.cache_indices[b];
    if (n_acc < 0 || cache_slot == params.pad_slot_id) {
      return;
    }

    bool is_vec_lane = lane < vec_count;
    int lane_elems = pack_elems * kPacksPerLane;
    int channel = is_vec_lane ? lane * lane_elems : vec_count * lane_elems + (lane - vec_count);
    int64_t src_slot_base = static_cast<int64_t>(cache_slot) * params.cache_stride_slot;
    Element const* hidden_base = params.hidden + static_cast<std::size_t>(b) * params.hidden_stride_b;
    bool do_track_write = false;
    int track_at = 0;
    int64_t track_slot = 0;
    if constexpr (DoTrack) {
      do_track_write = params.crossed[b] != 0;
      if (do_track_write) {
        track_at = params.track_step[b];
        track_slot = params.track_indices[b];
        do_track_write = track_slot != params.pad_slot_id;
      }
    }
    bool need_init = n_acc < width || (do_track_write && track_at < width);

    if (is_vec_lane) {
      pack_t init[kMaxWindow][kPacksPerLane];
      int pack_idx = lane * kPacksPerLane;
      if (need_init) {
#pragma unroll
        for (int w = 0; w < width; ++w) {
          Element const* row = params.cache + src_slot_base + static_cast<int64_t>(w) * params.cache_stride_w;
#pragma unroll
          for (int p = 0; p < kPacksPerLane; ++p) {
            init[w][p] = load_pack(row, pack_idx + p);
          }
        }
      }

      if constexpr (DoTrack) {
        if (do_track_write) {
          int64_t dst_base = track_slot * static_cast<int64_t>(params.cache_stride_slot);
#pragma unroll
          for (int w = 0; w < width; ++w) {
            int pos = track_at + w;
            Element* dst_row = params.cache + dst_base + static_cast<int64_t>(w) * params.cache_stride_w;
            if (pos < width) {
#pragma unroll
              for (int p = 0; p < kPacksPerLane; ++p) {
                store_pack(dst_row, pack_idx + p, init[pos][p]);
              }
            } else {
              Element const* row = hidden_base + static_cast<std::size_t>(pos - width) * params.hidden_stride_t;
#pragma unroll
              for (int p = 0; p < kPacksPerLane; ++p) {
                store_pack(dst_row, pack_idx + p, load_pack(row, pack_idx + p));
              }
            }
          }
        }
      }

#pragma unroll
      for (int w = 0; w < width; ++w) {
        int pos = n_acc + w;
        Element* dst_row = params.cache + src_slot_base + static_cast<int64_t>(w) * params.cache_stride_w;
        if (pos < width) {
#pragma unroll
          for (int p = 0; p < kPacksPerLane; ++p) {
            store_pack(dst_row, pack_idx + p, init[pos][p]);
          }
        } else {
          Element const* row = hidden_base + static_cast<std::size_t>(pos - width) * params.hidden_stride_t;
#pragma unroll
          for (int p = 0; p < kPacksPerLane; ++p) {
            store_pack(dst_row, pack_idx + p, load_pack(row, pack_idx + p));
          }
        }
      }
    } else {
      Element init[kMaxWindow];
      if (need_init) {
#pragma unroll
        for (int w = 0; w < width; ++w) {
          Element const* row = params.cache + src_slot_base + static_cast<int64_t>(w) * params.cache_stride_w;
          init[w] = row[channel];
        }
      }

      if constexpr (DoTrack) {
        if (do_track_write) {
          int64_t dst_base = track_slot * static_cast<int64_t>(params.cache_stride_slot);
#pragma unroll
          for (int w = 0; w < width; ++w) {
            int pos = track_at + w;
            Element value = pos < width
                ? init[pos]
                : hidden_base[static_cast<std::size_t>(pos - width) * params.hidden_stride_t + channel];
            params.cache[dst_base + static_cast<int64_t>(w) * params.cache_stride_w + channel] = value;
          }
        }
      }

#pragma unroll
      for (int w = 0; w < width; ++w) {
        int pos = n_acc + w;
        Element value = pos < width
            ? init[pos]
            : hidden_base[static_cast<std::size_t>(pos - width) * params.hidden_stride_t + channel];
        params.cache[src_slot_base + static_cast<int64_t>(w) * params.cache_stride_w + channel] = value;
      }
    }
  }
};

template <typename Element, int StaticWidthMinusOne, bool DoTrack>
sycl::event launch_draft_extend_static(
    sycl::queue& q,
    DraftExtendParams<Element> const& params) {
  if (params.batch == 0 || params.width_minus_one == 0 || params.channels == 0) {
    return sycl::event{};
  }
  if (params.width_minus_one > kMaxWindow) {
    throw std::invalid_argument("draft_extend width_minus_one exceeds kMaxWindow");
  }

  int pack_elems = kCopyBytes / static_cast<int>(sizeof(Element));
  int lane_elems = pack_elems * kPacksPerLane;
  auto aligned_elems = [pack_elems](int stride) {
    return stride % pack_elems == 0;
  };
  bool aligned = aligned_elems(params.hidden_stride_b) &&
      aligned_elems(params.hidden_stride_t) &&
      aligned_elems(params.cache_stride_slot) &&
      aligned_elems(params.cache_stride_w) &&
      (reinterpret_cast<std::uintptr_t>(params.hidden) % kCopyBytes == 0) &&
      (reinterpret_cast<std::uintptr_t>(params.cache) % kCopyBytes == 0);

  int vec_count = aligned ? params.channels / lane_elems : 0;
  int scalar_tail = params.channels - vec_count * lane_elems;
  int lanes_per_batch = vec_count + scalar_tail;
  int total_lanes = params.batch * lanes_per_batch;
  int global = ((total_lanes + kThreads - 1) / kThreads) * kThreads;

  DraftExtendSconvKernel<Element, StaticWidthMinusOne, DoTrack> kernel{
      params, lanes_per_batch, vec_count, pack_elems};
  return q.parallel_for<DraftExtendSconvKernel<Element, StaticWidthMinusOne, DoTrack>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kThreads)), kernel);
}

template <typename Element, int StaticWidthMinusOne>
sycl::event launch_draft_extend_track_selected(
    sycl::queue& q,
    DraftExtendParams<Element> const& params,
    bool do_track) {
  if (do_track) {
    return launch_draft_extend_static<Element, StaticWidthMinusOne, true>(q, params);
  }
  return launch_draft_extend_static<Element, StaticWidthMinusOne, false>(q, params);
}

template <typename Element>
sycl::event launch_draft_extend(
    sycl::queue& q,
    DraftExtendParams<Element> const& params,
    bool do_track) {
  switch (params.width_minus_one) {
    case 1:
      return launch_draft_extend_track_selected<Element, 1>(q, params, do_track);
    case 2:
      return launch_draft_extend_track_selected<Element, 2>(q, params, do_track);
    case 3:
      return launch_draft_extend_track_selected<Element, 3>(q, params, do_track);
    case 5:
      return launch_draft_extend_track_selected<Element, 5>(q, params, do_track);
    case 7:
      return launch_draft_extend_track_selected<Element, 7>(q, params, do_track);
    case 8:
      return launch_draft_extend_track_selected<Element, 8>(q, params, do_track);
    default:
      return launch_draft_extend_track_selected<Element, 0>(q, params, do_track);
  }
}

struct GatherCaseConfig {
  std::string name;
  int batch = 1;
  int width_minus_one = 3;
  int channels = 128;
  int total_tokens = 32;
  bool include_masked = false;
  bool random_metadata = false;
  int hidden_padding = 0;
  int cache_padding = 0;
  int slot_padding = 0;
  int track_padding = 0;
  int dst_stride_b = 1;
  int hidden_offset = 0;
  int cache_offset = 0;
  unsigned seed = 0;
};

template <typename Element_>
struct GatherHostTensors {
  using Element = Element_;

  std::vector<Element> hidden;
  std::vector<Element> cache;
  std::vector<Element> ref;
  std::vector<int32_t> track_idx;
  std::vector<uint8_t> mask;
  std::vector<int64_t> dst;
  int slots = 0;
  int hidden_stride_t = 0;
  int cache_stride_w = 0;
  int cache_stride_slot = 0;
  int track_stride_b = 0;
};

template <typename Element>
GatherHostTensors<Element> initialize_gather_case(GatherCaseConfig const& cfg) {
  GatherHostTensors<Element> h;
  h.slots = cfg.batch + 8;
  h.hidden_stride_t = cfg.channels + cfg.hidden_padding;
  h.cache_stride_w = cfg.channels + cfg.cache_padding;
  h.cache_stride_slot = cfg.width_minus_one * h.cache_stride_w + cfg.slot_padding;
  h.track_stride_b = cfg.width_minus_one + cfg.track_padding;

  std::size_t hidden_storage = static_cast<std::size_t>(cfg.hidden_offset)
      + static_cast<std::size_t>(std::max(0, cfg.total_tokens - 1)) * h.hidden_stride_t
      + cfg.channels;
  std::size_t cache_storage = static_cast<std::size_t>(cfg.cache_offset)
      + static_cast<std::size_t>(h.slots - 1) * h.cache_stride_slot
      + static_cast<std::size_t>(cfg.width_minus_one - 1) * h.cache_stride_w
      + cfg.channels;
  h.hidden.resize(std::max<std::size_t>(1, hidden_storage));
  h.cache.resize(std::max<std::size_t>(1, cache_storage));
  h.ref.resize(h.cache.size());
  h.track_idx.resize(static_cast<std::size_t>(cfg.batch - 1) * h.track_stride_b + cfg.width_minus_one);
  h.mask.resize(cfg.batch);
  h.dst.resize(static_cast<std::size_t>(cfg.batch - 1) * cfg.dst_stride_b + 1);

  unsigned seed = cfg.seed == 0
      ? 20260718u + static_cast<unsigned>(cfg.batch * 11 + cfg.channels * 7 + cfg.width_minus_one)
      : cfg.seed;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> h_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> c_dist(-0.35f, 0.35f);
  std::uniform_int_distribution<int> token_dist(0, std::max(0, cfg.total_tokens - 1));

  for (auto& v : h.hidden) {
    v = Element(h_dist(gen));
  }
  for (auto& v : h.cache) {
    v = Element(c_dist(gen));
  }

  for (int b = 0; b < cfg.batch; ++b) {
    h.mask[b] = static_cast<uint8_t>(!cfg.include_masked || (b % 5 != 2));
    int64_t slot = b == 0 ? h.slots - 1 : (b == 1 ? 0 : b + 2);
    h.dst[static_cast<std::size_t>(b) * cfg.dst_stride_b] = slot;
    for (int w = 0; w < cfg.width_minus_one; ++w) {
      int src_t = cfg.random_metadata ? token_dist(gen) : ((b * cfg.width_minus_one + w * 7 + 3) % cfg.total_tokens);
      h.track_idx[static_cast<std::size_t>(b) * h.track_stride_b + w] = src_t;
    }
  }

  return h;
}

template <typename Element>
void reference_gather(GatherCaseConfig const& cfg, GatherHostTensors<Element>& h) {
  h.ref = h.cache;
  for (int b = 0; b < cfg.batch; ++b) {
    if (h.mask[b] == 0) {
      continue;
    }
    int64_t dst_slot = h.dst[static_cast<std::size_t>(b) * cfg.dst_stride_b];
    if (dst_slot == kPadSlot) {
      continue;
    }
    for (int w = 0; w < cfg.width_minus_one; ++w) {
      int src_t = h.track_idx[static_cast<std::size_t>(b) * h.track_stride_b + w];
      for (int d = 0; d < cfg.channels; ++d) {
        std::size_t src = static_cast<std::size_t>(cfg.hidden_offset)
            + static_cast<std::size_t>(src_t) * h.hidden_stride_t + d;
        std::size_t dst = static_cast<std::size_t>(cfg.cache_offset)
            + static_cast<std::size_t>(dst_slot) * h.cache_stride_slot
            + static_cast<std::size_t>(w) * h.cache_stride_w + d;
        h.ref[dst] = h.hidden[src];
      }
    }
  }
}

struct DraftCaseConfig {
  std::string name;
  int batch = 1;
  int width_minus_one = 3;
  int channels = 128;
  int draft_tokens = 9;
  bool do_track = false;
  bool include_crossed = false;
  bool include_skips = false;
  bool random_metadata = false;
  int hidden_padding = 0;
  int hidden_batch_padding = 0;
  int cache_padding = 0;
  int slot_padding = 0;
  int hidden_offset = 0;
  int cache_offset = 0;
  unsigned seed = 0;
};

template <typename Element_>
struct DraftHostTensors {
  using Element = Element_;

  std::vector<Element> hidden;
  std::vector<Element> cache;
  std::vector<Element> ref;
  std::vector<int32_t> cache_indices;
  std::vector<int32_t> num_accepted;
  std::vector<uint8_t> crossed;
  std::vector<int32_t> track_step;
  std::vector<int64_t> track_indices;
  int slots = 0;
  int hidden_stride_t = 0;
  int hidden_stride_b = 0;
  int cache_stride_w = 0;
  int cache_stride_slot = 0;
};

template <typename Element>
DraftHostTensors<Element> initialize_draft_case(DraftCaseConfig const& cfg) {
  DraftHostTensors<Element> h;
  h.slots = cfg.do_track ? (cfg.batch * 2 + 16) : (cfg.batch + 8);
  h.hidden_stride_t = cfg.channels + cfg.hidden_padding;
  h.hidden_stride_b = cfg.draft_tokens * h.hidden_stride_t + cfg.hidden_batch_padding;
  h.cache_stride_w = cfg.channels + cfg.cache_padding;
  h.cache_stride_slot = cfg.width_minus_one * h.cache_stride_w + cfg.slot_padding;

  std::size_t hidden_storage = static_cast<std::size_t>(cfg.hidden_offset)
      + static_cast<std::size_t>(cfg.batch - 1) * h.hidden_stride_b
      + static_cast<std::size_t>(cfg.draft_tokens - 1) * h.hidden_stride_t
      + cfg.channels;
  std::size_t cache_storage = static_cast<std::size_t>(cfg.cache_offset)
      + static_cast<std::size_t>(h.slots - 1) * h.cache_stride_slot
      + static_cast<std::size_t>(cfg.width_minus_one - 1) * h.cache_stride_w
      + cfg.channels;
  h.hidden.resize(std::max<std::size_t>(1, hidden_storage));
  h.cache.resize(std::max<std::size_t>(1, cache_storage));
  h.ref.resize(h.cache.size());
  h.cache_indices.resize(cfg.batch);
  h.num_accepted.resize(cfg.batch);
  h.crossed.resize(cfg.batch);
  h.track_step.resize(cfg.batch);
  h.track_indices.resize(cfg.batch);

  unsigned seed = cfg.seed == 0
      ? 20260718u + static_cast<unsigned>(cfg.batch * 17 + cfg.channels * 3 + cfg.draft_tokens)
      : cfg.seed;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> h_dist(-0.9f, 0.9f);
  std::uniform_real_distribution<float> c_dist(-0.30f, 0.30f);
  std::uniform_int_distribution<int> accept_dist(0, cfg.draft_tokens);

  for (auto& v : h.hidden) {
    v = Element(h_dist(gen));
  }
  for (auto& v : h.cache) {
    v = Element(c_dist(gen));
  }

  for (int b = 0; b < cfg.batch; ++b) {
    int working_slot = b == 0 ? h.slots - 1 : (b == 1 ? 0 : b);
    h.cache_indices[b] = working_slot;

    int pattern = b % 6;
    int accepted = 0;
    if (cfg.random_metadata) {
      accepted = accept_dist(gen);
    } else if (pattern == 0) {
      accepted = 0;
    } else if (pattern == 1) {
      accepted = std::min(1, cfg.draft_tokens);
    } else if (pattern == 2) {
      accepted = std::min(cfg.width_minus_one - 1, cfg.draft_tokens);
    } else if (pattern == 3) {
      accepted = std::min(cfg.width_minus_one, cfg.draft_tokens);
    } else if (pattern == 4) {
      accepted = cfg.draft_tokens;
    } else {
      accepted = (b * 3) % (cfg.draft_tokens + 1);
    }
    if (cfg.include_skips && b % 11 == 4) {
      accepted = -1;
    }
    h.num_accepted[b] = accepted;

    bool crossed = cfg.do_track && cfg.include_crossed && accepted >= 0 && (b % 3 != 1);
    h.crossed[b] = static_cast<uint8_t>(crossed);
    int step = cfg.random_metadata ? accept_dist(gen) : ((b == 0) ? 0 : ((b == 1) ? cfg.draft_tokens : (b * 2) % (cfg.draft_tokens + 1)));
    h.track_step[b] = step;
    h.track_indices[b] = cfg.do_track ? static_cast<int64_t>(cfg.batch + 8 + b) : 0;
  }

  return h;
}

template <typename Element>
void reference_draft(DraftCaseConfig const& cfg, DraftHostTensors<Element>& h) {
  h.ref = h.cache;

  auto load_virtual = [&](int b, int slot, int pos, int d) -> Element {
    if (pos < cfg.width_minus_one) {
      std::size_t src = static_cast<std::size_t>(cfg.cache_offset)
          + static_cast<std::size_t>(slot) * h.cache_stride_slot
          + static_cast<std::size_t>(pos) * h.cache_stride_w + d;
      return h.cache[src];
    }
    std::size_t src = static_cast<std::size_t>(cfg.hidden_offset)
        + static_cast<std::size_t>(b) * h.hidden_stride_b
        + static_cast<std::size_t>(pos - cfg.width_minus_one) * h.hidden_stride_t + d;
    return h.hidden[src];
  };

  auto store_window = [&](int b, int src_slot, int at, int64_t dst_slot) {
    if (dst_slot == kPadSlot) {
      return;
    }
    for (int w = 0; w < cfg.width_minus_one; ++w) {
      int pos = at + w;
      for (int d = 0; d < cfg.channels; ++d) {
        std::size_t dst = static_cast<std::size_t>(cfg.cache_offset)
            + static_cast<std::size_t>(dst_slot) * h.cache_stride_slot
            + static_cast<std::size_t>(w) * h.cache_stride_w + d;
        h.ref[dst] = load_virtual(b, src_slot, pos, d);
      }
    }
  };

  for (int b = 0; b < cfg.batch; ++b) {
    int slot = h.cache_indices[b];
    int n_acc = h.num_accepted[b];
    if (n_acc < 0 || slot == kPadSlot) {
      continue;
    }
    if (cfg.do_track && h.crossed[b] != 0) {
      store_window(b, slot, h.track_step[b], h.track_indices[b]);
    }
    store_window(b, slot, n_acc, slot);
  }
}

struct VerifyResult {
  bool passed = true;
  int index = 0;
  float got = 0.0f;
  float expected = 0.0f;
};

template <typename Element>
VerifyResult verify_exact(std::vector<Element> const& got, std::vector<Element> const& ref) {
  VerifyResult result;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!exact_equal(got[i], ref[i])) {
      result.passed = false;
      result.index = static_cast<int>(i);
      result.got = to_float(got[i]);
      result.expected = to_float(ref[i]);
      return result;
    }
  }
  return result;
}

template <typename Element>
double gather_effective_bytes(GatherCaseConfig const& cfg, GatherHostTensors<Element> const& h) {
  double bytes = 0.0;
  for (int b = 0; b < cfg.batch; ++b) {
    if (h.mask[b] == 0 || h.dst[static_cast<std::size_t>(b) * cfg.dst_stride_b] == kPadSlot) {
      continue;
    }
    bytes += 2.0 * static_cast<double>(cfg.width_minus_one) * cfg.channels * sizeof(Element);
  }
  return bytes;
}

template <typename Element>
double draft_effective_bytes(DraftCaseConfig const& cfg, DraftHostTensors<Element> const& h) {
  double bytes = 0.0;
  double row_bytes = static_cast<double>(cfg.channels) * sizeof(Element);
  for (int b = 0; b < cfg.batch; ++b) {
    int slot = h.cache_indices[b];
    int n_acc = h.num_accepted[b];
    if (n_acc < 0 || slot == kPadSlot) {
      continue;
    }

    bool do_track = cfg.do_track && h.crossed[b] != 0 && h.track_indices[b] != kPadSlot;
    bool need_init = n_acc < cfg.width_minus_one ||
        (do_track && h.track_step[b] < cfg.width_minus_one);
    if (need_init) {
      bytes += static_cast<double>(cfg.width_minus_one) * row_bytes; // initial cache loads
    }
    bytes += static_cast<double>(cfg.width_minus_one) * row_bytes; // accepted stores
    for (int w = 0; w < cfg.width_minus_one; ++w) {
      if (n_acc + w >= cfg.width_minus_one) {
        bytes += row_bytes; // hidden load
      }
    }

    if (do_track) {
      bytes += static_cast<double>(cfg.width_minus_one) * row_bytes; // tracking stores
      int step = h.track_step[b];
      for (int w = 0; w < cfg.width_minus_one; ++w) {
        if (step + w >= cfg.width_minus_one) {
          bytes += row_bytes; // hidden load
        }
      }
    }
  }
  return bytes;
}

template <typename Element>
bool run_gather_case(
    sycl::queue& q,
    GatherCaseConfig const& cfg,
    int iterations,
    bool verify,
    double target_gbps) {
  GatherHostTensors<Element> h = initialize_gather_case<Element>(cfg);
  if (verify) {
    reference_gather<Element>(cfg, h);
  }

  DeviceBuffer<Element> d_hidden(q, h.hidden.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<int32_t> d_track_idx(q, h.track_idx.size());
  DeviceBuffer<uint8_t> d_mask(q, h.mask.size());
  DeviceBuffer<int64_t> d_dst(q, h.dst.size());

  d_hidden.copy_from(h.hidden);
  d_cache.copy_from(h.cache);
  d_track_idx.copy_from(h.track_idx);
  d_mask.copy_from(h.mask);
  d_dst.copy_from(h.dst);

  GatherScatterParams<Element> params{
      d_hidden.get() + cfg.hidden_offset,
      d_cache.get() + cfg.cache_offset,
      d_track_idx.get(),
      d_mask.get(),
      d_dst.get(),
      h.hidden_stride_t,
      h.cache_stride_slot,
      h.cache_stride_w,
      h.track_stride_b,
      1,
      cfg.dst_stride_b,
      cfg.batch,
      cfg.width_minus_one,
      cfg.channels,
      kPadSlot};

  auto launch = [&]() {
    return launch_gather_scatter<Element>(q, params);
  };

  launch();
  q.wait_and_throw();

  bool passed = true;
  VerifyResult vr;
  if (verify) {
    std::vector<Element> got(h.cache.size());
    d_cache.copy_to(got);
    vr = verify_exact<Element>(got, h.ref);
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
  double bytes = gather_effective_bytes<Element>(cfg, h);
  double gbps = (bytes / 1.0e9) / avg_s;
  bool applies_target = target_gbps > 0.0 && bytes >= kMinSustainedTargetBytes;
  if (applies_target && gbps < target_gbps) {
    passed = false;
  }

  std::cout << std::left << std::setw(36) << cfg.name
            << " op=gather dtype=" << std::setw(4) << element_dtype_text<Element>()
            << " B=" << std::setw(6) << cfg.batch
            << " W-1=" << std::setw(3) << cfg.width_minus_one
            << " D=" << std::setw(5) << cfg.channels
            << " masked=" << bool_text(cfg.include_masked)
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

template <typename Element>
bool run_draft_case(
    sycl::queue& q,
    DraftCaseConfig const& cfg,
    int iterations,
    bool verify,
    double target_gbps) {
  DraftHostTensors<Element> h = initialize_draft_case<Element>(cfg);
  if (verify) {
    reference_draft<Element>(cfg, h);
  }

  DeviceBuffer<Element> d_hidden(q, h.hidden.size());
  DeviceBuffer<Element> d_cache(q, h.cache.size());
  DeviceBuffer<int32_t> d_cache_indices(q, h.cache_indices.size());
  DeviceBuffer<int32_t> d_num_accepted(q, h.num_accepted.size());
  DeviceBuffer<uint8_t> d_crossed(q, h.crossed.size());
  DeviceBuffer<int32_t> d_track_step(q, h.track_step.size());
  DeviceBuffer<int64_t> d_track_indices(q, h.track_indices.size());

  d_hidden.copy_from(h.hidden);
  d_cache.copy_from(h.cache);
  d_cache_indices.copy_from(h.cache_indices);
  d_num_accepted.copy_from(h.num_accepted);
  d_crossed.copy_from(h.crossed);
  d_track_step.copy_from(h.track_step);
  d_track_indices.copy_from(h.track_indices);

  DraftExtendParams<Element> params{
      d_hidden.get() + cfg.hidden_offset,
      d_cache.get() + cfg.cache_offset,
      d_cache_indices.get(),
      d_num_accepted.get(),
      d_crossed.get(),
      d_track_step.get(),
      d_track_indices.get(),
      h.hidden_stride_b,
      h.hidden_stride_t,
      h.cache_stride_slot,
      h.cache_stride_w,
      cfg.batch,
      cfg.width_minus_one,
      cfg.channels,
      kPadSlot};

  auto launch = [&]() {
    return launch_draft_extend<Element>(q, params, cfg.do_track);
  };

  launch();
  q.wait_and_throw();

  bool passed = true;
  VerifyResult vr;
  if (verify) {
    std::vector<Element> got(h.cache.size());
    d_cache.copy_to(got);
    vr = verify_exact<Element>(got, h.ref);
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
  double bytes = draft_effective_bytes<Element>(cfg, h);
  double gbps = (bytes / 1.0e9) / avg_s;
  bool applies_target = target_gbps > 0.0 && bytes >= kMinSustainedTargetBytes;
  if (applies_target && gbps < target_gbps) {
    passed = false;
  }

  std::cout << std::left << std::setw(36) << cfg.name
            << " op=draft  dtype=" << std::setw(4) << element_dtype_text<Element>()
            << " B=" << std::setw(6) << cfg.batch
            << " W-1=" << std::setw(3) << cfg.width_minus_one
            << " D=" << std::setw(5) << cfg.channels
            << " T=" << std::setw(3) << cfg.draft_tokens
            << " track=" << bool_text(cfg.do_track)
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

std::vector<GatherCaseConfig> gather_quick_suite() {
  return {
      {"gs_tiny_reference_b2_w3_d4", 2, 3, 4, 5, false, false},
      {"gs_masked_odd_b9_w3_d7", 9, 3, 7, 47, true, false},
      {"gs_padded_misaligned_b6_w5_d19", 6, 5, 19, 83, true, false, 3, 5, 11, 2, 2, 1, 1},
      {"gs_dynamic_w4_b11_d33", 11, 4, 33, 97, true, true, 1, 3, 7, 1, 1, 0, 0, 1234u},
      {"gs_inkling_verify_b16_w3_d1536", 16, 3, 1536, 256, true, false},
      {"gs_inkling_kv_verify_b16_w3_d512", 16, 3, 512, 256, false, false},
      {"gs_inkling_scattered_b16_w3_d192", 16, 3, 192, 256, true, false},
  };
}

std::vector<DraftCaseConfig> draft_quick_suite() {
  return {
      {"draft_tiny_reference_b2_w3_d4", 2, 3, 4, 3, false, false, false},
      {"draft_track_odd_b9_w3_d7", 9, 3, 7, 5, true, true, true},
      {"draft_padded_misaligned_b6_w5_d19", 6, 5, 19, 6, true, true, true, false, 3, 5, 5, 11, 1, 1},
      {"draft_dynamic_w4_b11_d33", 11, 4, 33, 9, true, true, true, true, 1, 3, 3, 7, 0, 0, 4321u},
      {"draft_inkling_verify_b16_w3_d1536", 16, 3, 1536, 9, true, true, true},
      {"draft_inkling_kv_verify_b16_w3_d512", 16, 3, 512, 9, false, false, false},
      {"draft_inkling_scattered_b16_w3_d192", 16, 3, 192, 9, true, true, true},
  };
}

std::vector<GatherCaseConfig> gather_stress_suite() {
  return {
      {"gs_stress_w1_b3_d1", 3, 1, 1, 17, true, true, 1, 2, 3, 1, 1, 1, 0, 1000u},
      {"gs_stress_w2_b7_d31", 7, 2, 31, 89, true, true, 5, 7, 11, 2, 2, 0, 1, 1001u},
      {"gs_stress_w3_b13_d64", 13, 3, 64, 233, true, true, 8, 8, 16, 0, 1, 0, 0, 1002u},
      {"gs_stress_w7_b17_d129", 17, 7, 129, 307, true, true, 3, 5, 19, 1, 2, 1, 1, 1003u},
      {"gs_stress_w8_b19_d770", 19, 8, 770, 521, true, true, 14, 14, 32, 3, 1, 0, 0, 1004u},
  };
}

std::vector<DraftCaseConfig> draft_stress_suite() {
  return {
      {"draft_stress_w1_b3_d1", 3, 1, 1, 2, true, true, true, true, 1, 2, 2, 3, 1, 0, 2000u},
      {"draft_stress_w2_b7_d31", 7, 2, 31, 5, true, true, true, true, 5, 7, 7, 11, 0, 1, 2001u},
      {"draft_stress_w3_b13_d64", 13, 3, 64, 9, false, false, true, true, 8, 0, 8, 16, 0, 0, 2002u},
      {"draft_stress_w7_b17_d129", 17, 7, 129, 11, true, true, true, true, 3, 5, 5, 19, 1, 1, 2003u},
      {"draft_stress_w8_b19_d770", 19, 8, 770, 13, true, true, true, true, 14, 14, 14, 32, 0, 0, 2004u},
  };
}

std::vector<GatherCaseConfig> gather_perf_suite() {
  return {
      {"gs_perf_b8192_w3_d1536", 8192, 3, 1536, 8192 * 3 + 17, false, false},
      {"gs_perf_b8192_w3_d768", 8192, 3, 768, 8192 * 3 + 17, false, false},
      {"gs_perf_b8192_w3_d512", 8192, 3, 512, 8192 * 3 + 17, false, false},
      {"gs_perf_b16384_w3_d384", 16384, 3, 384, 16384 * 3 + 17, false, false},
      {"gs_perf_b32768_w3_d192", 32768, 3, 192, 32768 * 3 + 17, false, false},
      {"gs_perf_b32768_w3_d128", 32768, 3, 128, 32768 * 3 + 17, false, false},
  };
}

std::vector<DraftCaseConfig> draft_perf_suite() {
  return {
      {"draft_perf_b4096_w3_d1536", 4096, 3, 1536, 9, true, true, false},
      {"draft_perf_b8192_w3_d768", 8192, 3, 768, 9, true, true, false},
      {"draft_perf_b8192_w3_d512", 8192, 3, 512, 9, false, false, false},
      {"draft_perf_b16384_w3_d384", 16384, 3, 384, 9, true, true, false},
      {"draft_perf_b32768_w3_d192", 32768, 3, 192, 9, true, true, false},
      {"draft_perf_b32768_w3_d128", 32768, 3, 128, 9, false, false, false},
  };
}

struct Options {
  bool help = false;
  bool valid = true;
  bool verify = true;
  int iterations = 20;
  std::string suite = "quick";
  std::string dtype_name = "all";
  std::string op_name = "all";
  DType dtype = DType::kAll;
  Op op = Op::kAll;
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
    cmd.get_cmd_line_argument("dtype", dtype_name, std::string("all"));
    cmd.get_cmd_line_argument("op", op_name, std::string("all"));
    cmd.get_cmd_line_argument("target-gbps", target_gbps, 0.0);
    if (!parse_dtype(dtype_name, dtype) || !parse_op(op_name, op)) {
      valid = false;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Inkling BMG Gather/Scatter + Draft Extend SConv Cache Example\n\n"
        << "Options:\n"
        << "  --help                         Print this message\n"
        << "  --suite=<quick|stress|perf>     Built-in shape suite (default: quick)\n"
        << "  --op=<all|gather|draft>         Run both kernels or one kernel family\n"
        << "  --dtype=<all|bf16|fp16>         Cache/hidden dtype (default: all)\n"
        << "  --iterations=<int>              Timed kernel iterations\n"
        << "  --verify=<0|1>                  Run CPU reference comparison\n"
        << "  --target-gbps=<float>           Fail if large working-set cases are below this effective GB/s\n\n"
        << "Examples:\n"
        << "  ./examples/14_bmg_sconv/14_bmg_gather_scatter_draft_extend_sconv --suite=quick\n"
        << "  ./examples/14_bmg_sconv/14_bmg_gather_scatter_draft_extend_sconv --suite=stress --dtype=fp16\n"
        << "  ./examples/14_bmg_sconv/14_bmg_gather_scatter_draft_extend_sconv --suite=perf --verify=0 --iterations=100\n";
    return out;
  }
};

template <typename Element>
bool run_typed(
    sycl::queue& q,
    Options const& options,
    std::vector<GatherCaseConfig> const& gather_cases,
    std::vector<DraftCaseConfig> const& draft_cases) {
  bool all_passed = true;
  if (options.op == Op::kAll || options.op == Op::kGather) {
    for (auto const& cfg : gather_cases) {
      all_passed &= run_gather_case<Element>(
          q, cfg, options.iterations, options.verify, options.target_gbps);
    }
  }
  if (options.op == Op::kAll || options.op == Op::kDraft) {
    for (auto const& cfg : draft_cases) {
      all_passed &= run_draft_case<Element>(
          q, cfg, options.iterations, options.verify, options.target_gbps);
    }
  }
  return all_passed;
}

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
    std::cerr << "Unsupported dtype or op: dtype=" << options.dtype_name
              << " op=" << options.op_name << "\n";
    options.print_usage(std::cerr);
    return 1;
  }

  std::vector<GatherCaseConfig> gather_cases;
  std::vector<DraftCaseConfig> draft_cases;
  if (options.suite == "quick") {
    gather_cases = gather_quick_suite();
    draft_cases = draft_quick_suite();
  } else if (options.suite == "stress") {
    gather_cases = gather_stress_suite();
    draft_cases = draft_stress_suite();
  } else if (options.suite == "perf") {
    gather_cases = gather_perf_suite();
    draft_cases = draft_perf_suite();
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
    std::cout << "Suite: " << options.suite
              << ", op=" << op_text(options.op)
              << ", dtype=" << dtype_text(options.dtype)
              << ", iterations=" << options.iterations
              << ", verify=" << bool_text(options.verify) << "\n";

    bool all_passed = true;
    if (options.dtype == DType::kAll || options.dtype == DType::kBf16) {
      all_passed &= run_typed<cutlass::bfloat16_t>(q, options, gather_cases, draft_cases);
    }
    if (options.dtype == DType::kAll || options.dtype == DType::kFp16) {
      all_passed &= run_typed<cutlass::half_t>(q, options, gather_cases, draft_cases);
    }
    return all_passed ? 0 : 2;
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
  } catch (std::exception const& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  return 1;
}
